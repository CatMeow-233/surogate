"""The binding-probe path on the native trainer (docs/training.md: 绑定探针; docs/runbook.md: G3 基线): for A0 on
short items of the synthetic dev suite, the native candidate scores equal those of prototypes/ref_mamba2.py (bf16
storage) within the G1(b) tolerance per scored token, through the same rows and sums (surogate_tlc.eval.probes).
Needs $TLC_RUNS with models/A0, data/, goldens/A0.npz and suites/synth/ from the runbook.
"""

import json
import sys
from pathlib import Path

import numpy as np
from surogate_tlc.data.items import read_items
from surogate_tlc.data.tokenizer import BOS, CODE_START, PAD, load, special_ids
from surogate_tlc.eval import probes
from surogate_tlc.native import native_trainer

TLC = Path(__file__).resolve().parents[2]
SEQ_LEN, BATCH, ITEMS = 256, 8, 12


def test_probe_scores_match_the_reference(runs, tmp_path):
    import torch

    sys.path.insert(0, str(TLC / "prototypes"))
    import ref_mamba2

    model = runs / "models" / "A0"
    tokenizer = load(model, ordinary=True)
    special = special_ids(tokenizer)

    def encode(text):
        encoding = tokenizer.encode(text, add_special_tokens=False)
        return encoding.ids, encoding.offsets

    rows, firsts, owners = [], [], []
    for item in read_items(runs / "suites" / "synth" / "dev.jsonl"):
        prepared = probes.rows_of(item, encode, [special[BOS], special[CODE_START]], SEQ_LEN)
        if "rows" in prepared and len(item.candidates) <= 6 and len(set(owners)) < ITEMS:
            rows += prepared["rows"]
            firsts += prepared["first"]
            owners += [item.id] * len(prepared["rows"])
    assert len(set(owners)) == ITEMS

    wrapper = native_trainer(model, runs / "data", tmp_path / "native", batch=BATCH, seq_len=SEQ_LEN)

    def native(inputs, targets, positions):
        return np.asarray(wrapper.trainer.compute_logprobs(inputs, targets, use_lora=False, position_ids=positions))

    reference = ref_mamba2.Reference(model, bf16=True)

    def oracle(inputs, targets, positions):
        with torch.no_grad():
            return reference.logprobs(torch.from_numpy(inputs).long(), torch.from_numpy(targets).long()).numpy()

    kwargs = {"batch": BATCH, "seq_len": SEQ_LEN, "pad_id": special[PAD]}
    got, want = (probes.score_rows(rows, firsts, s, **kwargs) for s in (native, oracle))
    tolerance = float(np.load(runs / "goldens" / "A0.npz")["tolerance"])
    bounds = [tolerance * (len(row) - first) for row, first in zip(rows, firsts, strict=True)]
    errors = [abs(a - b) for a, b in zip(got, want, strict=True)]
    print(json.dumps({"probe_max_error": max(errors), "probe_max_bound": max(bounds), "rows": len(rows)}))
    assert all(e <= b for e, b in zip(errors, bounds, strict=True))
    for item_id in set(owners):  # the choice is the reference's where its margin exceeds both bounds
        mine = [i for i, owner in enumerate(owners) if owner == item_id]
        order = sorted(mine, key=lambda i: want[i], reverse=True)
        if want[order[0]] - want[order[1]] > bounds[order[0]] + bounds[order[1]]:
            assert max(mine, key=lambda i: got[i]) == order[0], item_id
