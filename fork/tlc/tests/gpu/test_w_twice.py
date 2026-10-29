"""Scouting for the shared-weight depth loop: one weight used twice in a layer (tests/gpu/twice_probe.py), one native
forward and backward, and its gradient against PyTorch. The outcome - ok, nan, garbage (gradient cosine below
0.99), error, crash or timeout - goes to $TLC_RUNS/env/w-twice.json. Every outcome passes: a failure is what P0
expects, and the record is the input for the later phase. The probe runs in a child process, so that a crash is
recorded too. A child that measured and then died or hung in teardown keeps its measurement, with the exit next to
it under "exit"; error, crash and timeout mean the probe never reported.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHILD = """
import json, shutil, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import save_file
sys.path.insert(0, sys.argv[1])
import twice_probe  # registers the probe model
from surogate_tlc import init, ir
from surogate_tlc.native import native_trainer

runs, out = Path(sys.argv[2]), Path(sys.argv[3])
tokenizer = runs / "tokenizer"
vocab = json.loads((runs / "data" / "manifest.json").read_text())["tokenizer"]["vocab_size"]
cfg = twice_probe.probe_config(vocab, pad_token_id=0, bos_token_id=1, eos_token_id=2)
model = out / "model"
model.mkdir(parents=True)
for name in ("tokenizer.json", "tokenizer_config.json"):
    shutil.copy(tokenizer / name, model / name)
(model / "config.json").write_text(json.dumps(cfg))
weights = init.init_state_dict(cfg, ir.compile_ir(cfg), seed=0, rules={**init.RULES, twice_probe.FIELD: "linear"})
save_file(weights, model / "model.safetensors", metadata={"format": "pt"})
ids = np.random.default_rng(0).integers(0, vocab, size=(2, 512), dtype=np.int32)
targets = np.concatenate([ids[:, 1:], np.full((2, 1), -100, np.int32)], 1)
positions = np.ascontiguousarray(np.broadcast_to(np.arange(512, dtype=np.int32), ids.shape))
trainer = native_trainer(model, runs / "data", out / "native", batch=2, seq_len=512).trainer
trainer.step(ids, targets, positions)
grads = {k: torch.from_dlpack(v).float().cpu() for k, v in trainer.get_gradients(0).items()}
name = f"blocks[0].{twice_probe.FIELD}"
got = grads[name].double().flatten()
want = twice_probe.reference_gradient(weights, cfg, torch.from_numpy(ids).long()).double().flatten()
cosine = float(got @ want / (got.norm() * want.norm()))
outcome = "nan" if not torch.isfinite(got).all() else ("ok" if cosine >= 0.99 else "garbage")
print(json.dumps({"outcome": outcome, "cosine": cosine, "norm_ratio": float(got.norm() / want.norm())}), flush=True)
"""
TIMEOUT = 1800  # seconds


def outcome_of(stdout: str, stderr: str, returncode: int | None) -> dict:
    """The record of one child run; returncode None: it timed out."""
    lines = [line for line in stdout.splitlines() if line.startswith('{"outcome"')]
    if lines:  # the probe measured; a non-zero exit after that is a teardown failure
        outcome = json.loads(lines[-1])
        if returncode != 0:
            outcome["exit"] = {"returncode": returncode, "stderr": stderr[-4000:]}
        return outcome
    kind = "timeout" if returncode is None else "crash" if returncode < 0 else "error"
    return {"outcome": kind, "returncode": returncode, "stderr": stderr[-4000:]}


def test_w_used_twice_is_recorded(runs, tmp_path):
    record = runs / "env" / "w-twice.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-c", CHILD, str(HERE), str(runs), str(tmp_path)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=TIMEOUT, env=dict(os.environ))
        outcome = outcome_of(result.stdout, result.stderr, result.returncode)
    except subprocess.TimeoutExpired as expired:  # its output may arrive as bytes despite text=True
        output = (expired.stdout, expired.stderr)
        stdout, stderr = (o.decode(errors="replace") if isinstance(o, bytes) else o or "" for o in output)
        outcome = outcome_of(stdout, stderr, None)
    record.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: outcome[key] for key in outcome if key != "stderr"}))
    assert outcome["outcome"] in {"ok", "nan", "garbage", "error", "crash", "timeout"}
