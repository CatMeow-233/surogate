#!/usr/bin/env python3
"""Controlled evaluation for assembled ZH/EN/bridge expert packs.
Records exact generations, route traces, language-heuristic and exact anchors.
Not a substitute for chrF/COMET, blind human translation review, or reasoning benchmarks.
"""
import argparse
import json
import re
from collections import defaultdict

import torch

from moe_pilot import Config, RoutedLM, get_device, get_tokenizer, load_ckpt, maybe_autocast, route_from_tags


def language_heuristic(text, lang):
    # Heuristic only; English technical words inside Chinese and proper nouns are legitimate.
    chinese = len(re.findall(r'[\u4e00-\u9fff]', text))
    latin = len(re.findall(r'[A-Za-z]', text))
    total = chinese + latin
    if total == 0:
        return None
    if lang == 'zh':
        return chinese / total >= 0.45
    if lang == 'en':
        return latin / total >= 0.85 and chinese <= 2
    return None


@torch.inference_mode()
def complete(model, tok, prompt, device, length, default='zh'):
    ids = tok.encode(prompt).ids
    initial_len = len(ids)
    traces = []
    eos = tok.token_to_id('<eos>')
    for _ in range(length):
        # Recompute full prefix to avoid adding an untested, route-aware KV cache.
        # This is intentionally slower than deployable inference.
        suffix = ids[-model.cfg.max_seq:]
        routes = route_from_tags(suffix, tok, model.names, default=default)
        x = torch.tensor([suffix], dtype=torch.long, device=device)
        r = torch.tensor([routes], dtype=torch.long, device=device)
        with maybe_autocast(device):
            pred = model(x, r)[:, -1].argmax(-1).item()
        traces.append(model.names[routes[-1]])
        ids.append(int(pred))
        if pred == eos:
            break
    return tok.decode(ids[initial_len:], skip_special_tokens=False), traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-new', type=int, default=64)
    args = ap.parse_args()
    pack = load_ckpt(args.model)
    cfg = Config(**pack['config'])
    device = get_device()
    model = RoutedLM(cfg, pack['names'])
    model.load_state_dict(pack['state'])
    model.to(device).eval()
    tok = get_tokenizer(args.tokenizer)
    aggregate = defaultdict(list)
    with open(args.out, 'w', encoding='utf-8') as out:
        with open(args.cases, encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                case = json.loads(line)
                required_expert = case.get('needs_expert')
                if required_expert and required_expert not in model.names:
                    continue
                completion, traces = complete(model, tok, case['prompt'], device,
                                              case.get('max_new', args.max_new),
                                              default=case.get('default', 'zh'))
                # Control tags are not prose. Strip them for rough language checks.
                prose = re.sub(r'<\|(?:zh|en|bridge)\|>|<eos>', '', completion)
                lang = language_heuristic(prose, case.get('target_language'))
                anchors = all(str(fragment) in completion for fragment in case.get('must_contain', []))
                forbidden = all(str(fragment) not in completion for fragment in case.get('must_not_contain', []))
                result = {'id': case.get('id'), 'group': case.get('group'),
                          'output': completion, 'route_trace': traces, 'language_heuristic': lang,
                          'anchors_present': anchors, 'forbidden_absent': forbidden}
                out.write(json.dumps(result, ensure_ascii=False) + '\n')
                aggregate[case.get('group', 'other')].append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
    print('\nSUMMARY (diagnostic, not a research result):')
    for group, items in aggregate.items():
        lang = [i['language_heuristic'] for i in items if i['language_heuristic'] is not None]
        print(group, {'n': len(items),
                      'language_fraction': sum(lang)/len(lang) if lang else None,
                      'anchor_fraction': sum(i['anchors_present'] for i in items)/len(items)})


if __name__ == '__main__':
    main()
