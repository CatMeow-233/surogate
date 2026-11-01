"""Host cost of the GRPO training-path rollout server (ledger topics T11, T12), without a GPU.

Three steps, each one command (run from any directory with an interpreter that has
numpy, torch, tokenizers and transformers, e.g. FORK_PY; nothing is installed):

  tokenizer DIR    train a Qwen2-style ByteLevel BPE (32k, Qwen2 split regex, chat and
                   tool specials) on the checkout's markdown and surogate/ sources, save to DIR
  per-token DIR    time SharedModelServer._generate per generated token with a scripted
                   sampler (no scheduler, no GPU) at 1k/2k/4k tokens (--lengths), with and
                   without stop strings; prints a digest of each response so modes can be
                   compared
  rollout DIR MS   16 requests on real threads through generate() and the real
                   DecodeScheduler; the GPU round is time.sleep(MS) with the GIL released,
                   as the native decode call releases it; prints rollout tok/s

Set SUROGATE_GRPO_DETOK and SUROGATE_GRPO_LIVE_DISPATCH to choose the modes. The results
in fork/docs/performance.md were measured with these commands; its T11 section says how to
reproduce the column measured before T11. fork/patches/tests/grpo/test_shared_model_detok.py
runs per-token and rollout at short lengths.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
SPLIT = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
DOCS = ["fork/docs/performance.md", "README.md", "fork/README.md", "surogate/grpo/shared_model.py", "docs/index.md"]


def build_tokenizer(out: str) -> None:
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    files = sorted(glob.glob(str(ROOT / "**/*.md"), recursive=True)) + sorted(
        glob.glob(str(ROOT / "surogate/**/*.py"), recursive=True)
    )
    files = [f for f in files if "/.venv" not in f]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.train(
        files,
        trainers.BpeTrainer(
            vocab_size=32768,
            special_tokens=SPECIAL,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>", clean_up_tokenization_spaces=False
    ).save_pretrained(out)
    print(f"{len(files)} files, vocabulary {tok.get_vocab_size()} -> {out}")


def completion_ids(tok, think: bool) -> list[int]:
    text = "\n".join((ROOT / d).read_text() for d in DOCS if (ROOT / d).exists())
    return tok.encode(("<think>\n" if think else "") + text, add_special_tokens=False)


def per_token(tokdir: str, reps: int, lengths=(1024, 2048, 4096)) -> None:
    from transformers import AutoTokenizer

    from surogate.grpo.shared_model import SharedModelServer

    tok = AutoTokenizer.from_pretrained(tokdir)
    completion = completion_ids(tok, think=True)
    server = SharedModelServer.__new__(SharedModelServer)  # _generate and prepare only: no HTTP, no trainer
    server.tokenizer, server.config, server.vocab, server.context = tok, {}, len(tok), 1 << 20
    server.eos, server.persistent_decode, server.adapter = {tok.convert_tokens_to_ids("<|im_end|>")}, True, "policy"
    server.trainer = type("Trainer", (), dict(batch_size=1, seq_length=1 << 20, decode_batch_sample=None))()
    try:  # T28's _generate calls admission.check_client() every step; code before T28 has neither
        from surogate._fork.shared_admission import Admission
    except ImportError:
        pass
    else:
        # The constructor builds this; with no recorded connection check_client() returns at once.
        server.admission = Admission(dict(max_concurrency=1))
    mode = os.environ.get("SUROGATE_GRPO_DETOK", "final")
    for stop in ([], ["</answer>", "\n\n\n\n"]):
        for length in lengths:
            walls, cpus = [], []
            for _ in range(reps):
                request = server.prepare(
                    dict(model="policy", tokens=[1, 2, 3], max_tokens=length, logprobs=True, stop=stop)
                )
                script = iter(completion[:length])
                wall, cpu = time.perf_counter(), time.thread_time()
                result = server._generate(
                    request,
                    None,
                    lambda *args, script=script: dict(
                        status=0, token=next(script), logprob=-0.5, top_ids=[], top_logprobs=[]
                    ),
                )
                walls.append((time.perf_counter() - wall) / length * 1e6)
                cpus.append((time.thread_time() - cpu) / length * 1e6)
            choice = result["choices"][0]
            assert choice["token_ids"] == completion[:length]
            digest = hashlib.sha1(json.dumps(choice, sort_keys=True).encode()).hexdigest()[:10]
            print(
                f"mode={mode:11s} stop={'yes' if stop else 'no':3s} L={length:5d} "
                f"wall={statistics.median(walls):7.2f} us/token cpu={statistics.median(cpus):7.2f} us/token "
                f"response={digest} finish={choice['finish_reason']}"
            )


def rollout(tokdir: str, gpu_ms: float, reps: int, length: int) -> None:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from surogate.grpo.shared_model import SharedModelServer

    tok = AutoTokenizer.from_pretrained(tokdir)
    completion = completion_ids(tok, think=False)

    class Trainer:
        batch_size, seq_length = 1, 1 << 16

        def __init__(self):
            self.steps, self.rounds, self.rows = {}, 0, 0

        def get_shared_base_weights(self):
            return dict(weight=torch.zeros(8, dtype=torch.bfloat16))

        def decode_batch_logits(self, *args):
            raise AssertionError("rollouts sample on the device")

        def reset_decode_state(self):
            pass

        def release_decode_sessions(self, ids):
            for session in ids:
                self.steps.pop(int(session), None)

        def decode_batch_sample(self, ids, tokens, offsets, resets, sampling):
            time.sleep(gpu_ms / 1000)  # one weight-bound round, GIL released
            self.rounds += 1
            self.rows += len(ids)
            rows = []
            for session in ids.tolist():
                k = self.steps.get(session, 0)
                self.steps[session] = k + 1
                rows.append(dict(status=0, token=completion[k], logprob=-0.5, top_ids=[], top_logprobs=[]))
            return rows

    def run(lengths):
        trainer = Trainer()
        server = SharedModelServer(
            trainer,
            tok,
            dict(vocab_size=len(tok)),
            dict(host="127.0.0.1", port=0, model="base", max_context=1 << 16, max_concurrency=16, eos_token_id=[0]),
        )
        try:
            requests = [
                server.prepare(dict(model="base", tokens=[5, 6, 7], max_tokens=n, logprobs=True)) for n in lengths
            ]
            results = [None] * len(lengths)
            threads = [
                threading.Thread(target=lambda i=i: results.__setitem__(i, server.generate(requests[i])))
                for i in range(len(lengths))
            ]
            start = time.perf_counter()
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            wall = time.perf_counter() - start
            assert all(r["choices"][0]["token_ids"] == completion[:n] for r, n in zip(results, lengths, strict=True))
            return sum(lengths) / wall, trainer.rounds, trainer.rows / trainer.rounds
        finally:
            server.close()

    spread = [int(n) for n in np.random.default_rng(0).integers(length // 4, length + 1, 16)]
    for name, lengths in (("equal", [length] * 16), ("spread", spread)):
        runs = [run(lengths) for _ in range(reps)]
        rate = statistics.median(r[0] for r in runs)
        ideal = sum(lengths) / (max(lengths) * gpu_ms / 1000)  # every live request in every round
        print(
            f"gpu={gpu_ms:5.1f}ms lengths={name:6s} detok={os.environ.get('SUROGATE_GRPO_DETOK', 'final'):11s} "
            f"live_dispatch={os.environ.get('SUROGATE_GRPO_LIVE_DISPATCH', '0')} tok/s={rate:6.0f} "
            f"({rate / ideal:5.1%} of {ideal:5.0f}) rounds={statistics.median(r[1] for r in runs):5.0f} "
            f"mean_batch={statistics.median(r[2] for r in runs):5.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("tokenizer").add_argument("dir")
    p = sub.add_parser("per-token")
    p.add_argument("dir")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    p = sub.add_parser("rollout")
    p.add_argument("dir")
    p.add_argument("gpu_ms", type=float)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--length", type=int, default=1024)
    args = parser.parse_args()
    if args.command == "tokenizer":
        build_tokenizer(args.dir)
    elif args.command == "per-token":
        per_token(args.dir, args.reps, args.lengths)
    else:
        rollout(args.dir, args.gpu_ms, args.reps, args.length)


if __name__ == "__main__":
    main()
