#!/usr/bin/env python3
"""Single-GPU Chinese/English/bridge independent-FFN MoE pilot.

Dependencies: torch, numpy, tokenizers (tokenizers only for tokenize/pack/generate).
Architecture is the 24-layer x 512, 32k vocabulary, 1728-intermediate
subset of the proposed ~2B/100M-active model. No performance claims.

No external downloads are performed by the script. Supply licensed corpora.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

TAGS = ["<unk>", "<eos>", "<|zh|>", "<|en|>", "<|bridge|>"]

@dataclass
class Config:
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 24
    n_heads: int = 8
    n_kv_heads: int = 2
    intermediate: int = 1728
    max_seq: int = 1024

    def validate(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0


def cfg_for_profile(profile, vocab_size):
    if profile == 'pilot':
        cfg = Config(vocab_size=vocab_size)
    elif profile == 'smoke':
        cfg = Config(vocab_size=vocab_size, d_model=64, n_layers=2,
                     n_heads=4, n_kv_heads=2, intermediate=128, max_seq=128)
    else:
        raise ValueError(profile)
    cfg.validate()
    return cfg


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xx = x.float()
        normed = xx * torch.rsqrt(xx.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.to(x.dtype) * self.weight


def rotary(x, cos, sin):
    # x = [batch, heads, time, head_dim]; cos/sin = [1, 1, time, head_dim/2]
    x0, x1 = x[..., ::2], x[..., 1::2]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack((out0, out1), dim=-1).flatten(-2)


class CausalAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * self.hd, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        inv = 1 / (10000 ** (torch.arange(0, self.hd, 2, device=x.device,
                                          dtype=torch.float32) / self.hd))
        ang = torch.outer(torch.arange(T, device=x.device, dtype=torch.float32), inv)
        cos, sin = ang.cos()[None, None], ang.sin()[None, None]
        q = rotary(q.float(), cos, sin).to(x.dtype)
        k = rotary(k.float(), cos, sin).to(x.dtype)
        k = k.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, i = cfg.d_model, cfg.intermediate
        self.gate = nn.Linear(d, i, bias=False)
        self.up = nn.Linear(d, i, bias=False)
        self.down = nn.Linear(i, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SharedLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = CausalAttention(cfg)
        self.norm_ffn = RMSNorm(cfg.d_model)


class RoutedLM(nn.Module):
    def __init__(self, cfg, names):
        super().__init__()
        self.cfg = cfg
        self.names = tuple(names)
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([SharedLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model)
        self.experts = nn.ModuleDict({name: nn.ModuleList([SwiGLU(cfg) for _ in range(cfg.n_layers)])
                                      for name in names})
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _ffn(self, h, routes, layer_index):
        b, t, d = h.shape
        if routes is None:
            return self.experts[self.names[0]][layer_index](h)
        assert routes.shape == (b, t)
        flat = h.reshape(-1, d)
        route = routes.reshape(-1)
        output = torch.zeros_like(flat)
        for idx, name in enumerate(self.names):
            positions = (route == idx).nonzero(as_tuple=False).flatten()
            if positions.numel():
                selected = flat.index_select(0, positions)
                vals = self.experts[name][layer_index](selected)
                output = output.index_add(0, positions, vals)
        return output.reshape(b, t, d)

    def forward(self, token_ids, routes=None, checkpointing=False):
        h = self.emb(token_ids)
        for layer_index, layer in enumerate(self.layers):
            def block(x, layer=layer, layer_index=layer_index):
                x = x + layer.attn(layer.norm_attn(x))
                x = x + self._ffn(layer.norm_ffn(x), routes, layer_index)
                return x
            if checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return F.linear(self.final_norm(h), self.emb.weight)  # tied output head

    def shared_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith('experts.')}

    def expert_state(self, name):
        return {k: v.detach().cpu().clone() for k, v in self.experts[name].state_dict().items()}

    def active_counts(self):
        shared = sum(p.numel() for key, p in self.named_parameters() if not key.startswith('experts.'))
        experts = {n: sum(p.numel() for p in self.experts[n].parameters()) for n in self.names}
        return shared, experts, shared + sum(experts.values()), shared + max(experts.values())


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_ckpt(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def prepare_seed(path):
    pack = load_ckpt(path)
    cfg = Config(**pack['config'])
    model = RoutedLM(cfg, ['seed'])
    model.load_state_dict(pack['model'])
    return model, cfg


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def maybe_autocast(device):
    # BF16 on a 5090; on CPU, use FP32 for predictable smoke tests.
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.autocast('cpu', enabled=False)


def random_batch(mm, batch_size, seq_len, device):
    if len(mm) < seq_len + 2:
        raise ValueError('Corpus too small for seq_len; use a larger packed .bin')
    starts = np.random.randint(0, len(mm) - seq_len - 1, size=batch_size)
    segments = np.stack([np.asarray(mm[int(i):int(i) + seq_len + 1], dtype=np.int64)
                         for i in starts])
    x = torch.from_numpy(segments[:, :-1].copy()).to(device)
    y = torch.from_numpy(segments[:, 1:].copy()).to(device)
    return x, y


def train_loop(args, model, pack_fn, trainable_parameters):
    device = get_device()
    model.to(device)
    model.train()
    dtype = np.uint16 if args.dtype == 'uint16' else np.uint32
    mm = np.memmap(args.data, mode='r', dtype=dtype)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=0.1)
    step_start = 0
    if args.resume:
        res = load_ckpt(args.resume)
        model.load_state_dict(res['model_state'])
        optimizer.load_state_dict(res['optimizer'])
        step_start = int(res['step'])
    for step in range(step_start, args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_scalar = 0.0
        for _ in range(args.accum):
            x, y = random_batch(mm, args.batch, args.seq, device)
            with maybe_autocast(device):
                logits = model(x, checkpointing=args.checkpoint)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                       y.reshape(-1)) / args.accum
            loss.backward()
            loss_scalar += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        optimizer.step()
        if (step + 1) % args.log_every == 0 or step == 0:
            mem = (torch.cuda.max_memory_allocated() / 2**30) if device.type == 'cuda' else 0
            print(json.dumps({'step': step + 1, 'loss': round(loss_scalar, 5),
                              'peak_allocated_gib': round(mem, 3)}, ensure_ascii=False), flush=True)
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            out = pack_fn(model, step + 1)
            path = Path(args.out)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out, path)
            # The resumable optimizer state is intentionally stored separately.
            if args.save_optimizer:
                torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'step': step + 1}, str(path) + '.resume.pt')
    print('Saved:', args.out)


def train_seed(args):
    cfg = cfg_for_profile(args.profile, args.vocab)
    model = RoutedLM(cfg, ['seed'])
    print('Seed parameters:', model.active_counts()[2])
    train_loop(args, model, lambda m, step: {'phase': 'seed', 'config': asdict(cfg),
                                            'model': m.state_dict(), 'step': step},
               list(model.parameters()))


def train_expert(args):
    model, cfg = prepare_seed(args.seed_checkpoint)
    model.requires_grad_(False)
    model.experts['seed'].requires_grad_(True)
    pars = list(model.experts['seed'].parameters())
    print('Trainable expert parameters:', sum(p.numel() for p in pars))
    seed_hash = fingerprint(args.seed_checkpoint)
    def pack(m, step):
        return {'phase': 'expert', 'name': args.name, 'config': asdict(cfg),
                'seed_sha256': seed_hash, 'expert': m.expert_state('seed'),
                'step': step}
    train_loop(args, model, pack, pars)


def assemble(args):
    seed_model, cfg = prepare_seed(args.seed_checkpoint)
    checkpoints = {}
    seed_hash = fingerprint(args.seed_checkpoint)
    for name, path in [pair.split('=', 1) for pair in args.expert]:
        if name in checkpoints:
            raise ValueError('Duplicate expert: ' + name)
        if name not in {'zh', 'en', 'bridge'}:
            raise ValueError('Use expert names zh, en, bridge')
        payload = load_ckpt(path)
        if payload['seed_sha256'] != seed_hash:
            raise ValueError(f'Seed mismatch for {name}: refusing incompatible composition')
        checkpoints[name] = payload['expert']
    if not {'zh', 'en'}.issubset(checkpoints):
        raise ValueError('The experiment needs at least zh and en expert packs')
    m = RoutedLM(cfg, list(checkpoints))
    m.load_state_dict(seed_model.shared_state(), strict=False)
    for name, params in checkpoints.items():
        m.experts[name].load_state_dict(params)
    shared, experts, total, active = m.active_counts()
    print(json.dumps({'shared': shared, 'experts': experts,
                      'total': total, 'active_top1': active}, indent=2))
    torch.save({'phase': 'assembled', 'seed_sha256': seed_hash, 'config': asdict(cfg),
                'names': m.names, 'state': m.state_dict()}, args.out)
    print('Saved:', args.out)


@torch.inference_mode()
def verify_assembly(args):
    """True isolated-vs-assembled logit identity test for real exported experts."""
    pack = load_ckpt(args.model)
    base, cfg = prepare_seed(args.seed_checkpoint)
    if pack['seed_sha256'] != fingerprint(args.seed_checkpoint):
        raise ValueError('Seed mismatch')
    merged = RoutedLM(cfg, pack['names'])
    merged.load_state_dict(pack['state'])
    device = get_device()
    merged.to(device).eval()
    base.to(device).eval()
    torch.manual_seed(9)
    x = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device=device)
    for name, path in [pair.split('=', 1) for pair in args.expert]:
        standalone = load_ckpt(path)
        if standalone['seed_sha256'] != pack['seed_sha256']:
            raise ValueError('Expert seed mismatch')
        base.experts['seed'].load_state_dict(standalone['expert'])
        r = torch.full_like(x, pack['names'].index(name))
        with maybe_autocast(device):
            a = merged(x, r)
            b = base(x)
        diff = float((a.float()-b.float()).abs().max())
        print(name, 'max_abs_logit_difference', diff, flush=True)
        torch.testing.assert_close(a.float(), b.float(), atol=args.atol, rtol=args.rtol)
    print('PASS: separately exported experts exactly reproduce assembled forced-route logits.')


def get_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def train_tokenizer(args):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, min_frequency=2, special_tokens=TAGS)
    def lines():
        for path in args.inputs:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield line.rstrip('\n')
    tok.train_from_iterator(lines(), trainer=trainer)
    tok.save(args.out)
    print('Vocabulary:', tok.get_vocab_size(), 'Saved:', args.out)


def pack_data(args):
    tok = get_tokenizer(args.tokenizer)
    vocab_size = tok.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError('Pilot uses uint16 packed IDs; reduce vocabulary below 65536')
    eos = tok.token_to_id('<eos>')
    count = 0
    with open(args.out, 'wb') as out:
        for path in args.inputs:
            with open(path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ids = tok.encode(line).ids + [eos]
                        np.asarray(ids, dtype=np.uint16).tofile(out)
                        count += len(ids)
    print('Packed tokens:', count, 'Saved:', args.out)


def route_from_tags(ids, tok, names, default='zh'):
    """Explicit/oracle route. A tag changes the current route from that position onward."""
    lookup = {tok.token_to_id('<|' + name + '|>'): names.index(name) for name in names}
    if default not in names:
        default = names[0]
    current = names.index(default)
    result = []
    for i in ids:
        if i in lookup:
            current = lookup[i]
        result.append(current)
    return result


@torch.inference_mode()
def generate(args):
    data = load_ckpt(args.checkpoint)
    cfg = Config(**data['config'])
    m = RoutedLM(cfg, data['names'])
    m.load_state_dict(data['state'])
    device = get_device()
    m.to(device).eval()
    tok = get_tokenizer(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    assert max(ids) < cfg.vocab_size
    print('Initial routes:', [data['names'][i] for i in route_from_tags(ids, tok, m.names)[-12:]])
    for _ in range(args.max_new):
        seq = ids[-cfg.max_seq:]
        route = route_from_tags(seq, tok, m.names, default=args.default)
        x = torch.tensor([seq], dtype=torch.long, device=device)
        r = torch.tensor([route], dtype=torch.long, device=device)
        with maybe_autocast(device):
            logits = m(x, r)[:, -1]
        pred = int(logits.argmax(-1).item())
        ids.append(pred)
        if pred == tok.token_to_id('<eos>'):
            break
    print(tok.decode(ids, skip_special_tokens=False))
    print('Final route:', m.names[route_from_tags(ids, tok, m.names, default=args.default)[-1]])


def smoke():
    torch.manual_seed(7)
    cfg = cfg_for_profile('smoke', 256)
    m = RoutedLM(cfg, ['zh', 'en', 'bridge']).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 12))
    routes_zh = torch.zeros_like(x)
    with torch.no_grad():
        alone = m(x)
        routed = m(x, routes_zh)
    assert torch.allclose(alone, routed, rtol=1e-5, atol=1e-5), 'Top-1 vs isolated mismatch'
    old_shared = m.emb.weight.detach().clone()
    old_en = m.experts['en'][0].gate.weight.detach().clone()
    m.requires_grad_(False)
    m.experts['zh'].requires_grad_(True)
    opt = torch.optim.AdamW(m.experts['zh'].parameters(), lr=1e-3)
    m.train()
    out = m(x, routes_zh)
    loss = F.cross_entropy(out[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
    loss.backward()
    opt.step()
    assert torch.equal(old_shared, m.emb.weight), 'Frozen shared weights changed'
    assert torch.equal(old_en, m.experts['en'][0].gate.weight), 'English expert changed during Chinese training'
    seq = torch.tensor([[0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
                        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]])
    assert m(x, seq).shape == (2, 12, cfg.vocab_size)
    print('PASS: isolated vs Top-1 routing, independent expert update, shape and route switching.')
    print('SMOKE ONLY: these tests do not establish Chinese/English language quality.')


def add_train_args(parser):
    parser.add_argument('--data', required=True, help='Packed uint16 token ids')
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--seq', type=int, default=512)
    parser.add_argument('--accum', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', action='store_true', default=True)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--resume')
    parser.add_argument('--save-optimizer', action='store_true')
    parser.add_argument('--dtype', choices=['uint16', 'uint32'], default='uint16')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('smoke')
    t = sub.add_parser('tokenizer')
    t.add_argument('--inputs', nargs='+', required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--vocab', type=int, default=32768)
    p = sub.add_parser('pack')
    p.add_argument('--inputs', nargs='+', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--out', required=True)
    s = sub.add_parser('train-seed')
    add_train_args(s)
    s.add_argument('--profile', choices=['pilot', 'smoke'], default='pilot')
    s.add_argument('--vocab', type=int, default=32768)
    e = sub.add_parser('train-expert')
    add_train_args(e)
    e.add_argument('--seed-checkpoint', required=True)
    e.add_argument('--name', choices=['zh', 'en', 'bridge'], required=True)
    a = sub.add_parser('assemble')
    a.add_argument('--seed-checkpoint', required=True)
    a.add_argument('--expert', nargs='+', required=True, help='zh=path en=path [bridge=path]')
    a.add_argument('--out', required=True)
    v = sub.add_parser('verify')
    v.add_argument('--model', required=True)
    v.add_argument('--seed-checkpoint', required=True)
    v.add_argument('--expert', nargs='+', required=True)
    v.add_argument('--seq', type=int, default=16)
    v.add_argument('--batch', type=int, default=1)
    v.add_argument('--atol', type=float, default=2e-4)
    v.add_argument('--rtol', type=float, default=2e-4)
    g = sub.add_parser('generate')
    g.add_argument('--checkpoint', required=True)
    g.add_argument('--tokenizer', required=True)
    g.add_argument('--prompt', required=True)
    g.add_argument('--default', choices=['zh', 'en', 'bridge'], default='zh')
    g.add_argument('--max-new', type=int, default=48)
    args = ap.parse_args()
    if args.cmd == 'smoke': smoke()
    elif args.cmd == 'tokenizer': train_tokenizer(args)
    elif args.cmd == 'pack': pack_data(args)
    elif args.cmd == 'train-seed': train_seed(args)
    elif args.cmd == 'train-expert': train_expert(args)
    elif args.cmd == 'assemble': assemble(args)
    elif args.cmd == 'verify': verify_assembly(args)
    elif args.cmd == 'generate': generate(args)


if __name__ == '__main__':
    main()
