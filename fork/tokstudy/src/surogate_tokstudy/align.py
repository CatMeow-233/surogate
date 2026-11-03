"""Student/teacher token alignment (`tokstudy-align/1`): how exactly a teacher can score a student by byte-prefix
marginals (BPM). A CPU screen; TLC's G-BPM remains the authoritative check.

Each document's text segments are encoded by the student and by the teacher's canonical tokenization of the same
bytes (the teacher's normaliser removed, so both see identical bytes; the documents it would have changed are
counted). Every student token [s, e) is classified against the teacher token holding its first byte and the one
holding its last byte, as the notebook's alignment prototype measures it:

- `one_to_one`: the two spans coincide;
- `head`: it starts where a teacher token starts and ends inside it (the first of the N student tokens of one
  teacher token: BPM's 1:N head, mapped by φ);
- `interior`: it starts inside a teacher token and ends inside or at its end (1:N interior, derived by the student's
  BPE);
- `cross_boundary`: its first and last byte lie in different teacher tokens, i.e. a teacher boundary falls strictly
  inside it (BPM masks these positions). A student token covering several whole teacher tokens is cross-boundary.

The non-identity all-whitespace rows are the student tokens made only of spaces and tabs that are not one_to_one:
the rows the proposed whitespace-mask rule counts, as a share of the code domain's student tokens (README: 命令行).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .canonical import TokstudyError, atomic_write_json
from .codecs import Codec, SubwordCodec, parse_tokenizer_arg
from .docs import EvalDoc
from .intrinsic import add_codec_arguments, add_docs_arguments, codecs_from, docs_from

FORMAT = "tokstudy-align/1"
CLASSES = ("one_to_one", "head", "interior", "cross_boundary")
WHITESPACE = frozenset(b" \t")


def classify(student_ends: np.ndarray, teacher_ends: np.ndarray) -> np.ndarray:
    """The class index (into CLASSES) of every student token; both end arrays cover the same bytes."""
    s_ends = np.asarray(student_ends, dtype=np.int64)
    t_ends = np.asarray(teacher_ends, dtype=np.int64)
    if s_ends.size == 0:
        return np.zeros(0, dtype=np.int64)
    if t_ends.size == 0 or s_ends[-1] != t_ends[-1]:
        raise TokstudyError("student and teacher tokens cover different bytes")
    s_starts = np.concatenate([[0], s_ends[:-1]])
    t_starts = np.concatenate([[0], t_ends[:-1]])
    first = np.searchsorted(t_ends, s_starts, side="right")  # teacher token holding the first byte
    last = np.searchsorted(t_ends, s_ends - 1, side="right")  # ... and the last byte
    out = np.full(s_ends.size, 3, dtype=np.int64)
    inside = first == last
    starts_with = s_starts == t_starts[np.minimum(first, t_ends.size - 1)]
    ends_with = s_ends == t_ends[np.minimum(first, t_ends.size - 1)]
    out[inside & starts_with & ends_with] = 0
    out[inside & starts_with & ~ends_with] = 1
    out[inside & ~starts_with] = 2
    return out


def measure(student: Codec, teacher: SubwordCodec, docs: Iterable[EvalDoc]) -> dict[str, dict[str, Any]]:
    """Per domain: student and teacher tokens, the student tokens of each class, and the whitespace rows."""
    out: dict[str, dict[str, Any]] = {}
    for doc in docs:
        d = out.setdefault(
            doc.domain,
            {
                "docs": 0,
                "bytes": 0,
                "student_tokens": 0,
                "teacher_tokens": 0,
                **{c: 0 for c in CLASSES},
                "whitespace_tokens": 0,
                "non_identity_whitespace": 0,
                "docs_changed_by_teacher_normalizer": 0,
            },
        )
        d["docs"] += 1
        changed = False
        for text in doc.texts():
            s_ids, s_ends = student.encode_text(text)
            _, t_ends = teacher.encode_text(text)
            changed |= teacher.changed_by_dropped_normalizer(text)
            classes = classify(s_ends, t_ends)
            d["bytes"] += int(s_ends[-1]) if s_ends.size else 0
            d["student_tokens"] += int(s_ids.size)
            d["teacher_tokens"] += int(t_ends.size)
            for k, name in enumerate(CLASSES):
                d[name] += int((classes == k).sum())
            for token_id, cls in zip(s_ids.tolist(), classes.tolist(), strict=True):
                b = student.token_bytes(token_id)
                if b and set(b) <= WHITESPACE:
                    d["whitespace_tokens"] += 1
                    d["non_identity_whitespace"] += cls != 0
        d["docs_changed_by_teacher_normalizer"] += changed
    for d in out.values():
        n = d["student_tokens"]
        d["rates"] = {name: (d[name] / n if n else None) for name in CLASSES}
        d["rates"]["non_identity_whitespace"] = d["non_identity_whitespace"] / n if n else None
    return dict(sorted(out.items()))


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_tokstudy align", description=__doc__.split("\n")[0])
    parser.add_argument("--teacher", required=True, metavar="PATH@SHA256", help="the teacher's tokenizer.json")
    add_codec_arguments(parser)
    add_docs_arguments(parser)
    parser.add_argument("--out", type=Path, required=True, help="the tokstudy-align/1 JSON to write")
    args = parser.parse_args(list(argv))
    try:
        _, teacher_path, teacher_sha = parse_tokenizer_arg(f"teacher={args.teacher}")
        teacher = SubwordCodec.from_tokenizer_json(
            teacher_path, teacher_sha, name="teacher", drop_normalizer=True, check=args.check
        )
        students, skipped = codecs_from(args)
        docs, inputs = docs_from(args)
        report = {
            "format": FORMAT,
            "teacher": teacher.identity(),
            "inputs": inputs,
            "students": {n: {"codec": c.identity(), "domains": measure(c, teacher, docs)} for n, c in students.items()},
            "skipped_hnet_arms": skipped,
            "classes": list(CLASSES),
        }
    except (TokstudyError, OSError, ValueError) as err:
        print(f"align: {err}", file=sys.stderr)
        return 1
    atomic_write_json(args.out, report)
    for name, student in report["students"].items():
        for domain, d in student["domains"].items():
            r = {k: "-" if v is None else f"{v:.4f}" for k, v in d["rates"].items()}
            print(
                f"{name}\t{domain}\t1:1 {r['one_to_one']}\tcross {r['cross_boundary']}\t"
                f"ws-non-identity {r['non_identity_whitespace']}"
            )
    return 0
