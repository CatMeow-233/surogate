"""Binding probe items: the record every probe suite holds (T1, T2 from data.synth; T3 from data.natural).

An item asks for one continuation of a prompt: eval.probes scores every candidate after the prompt with the model
and counts the item correct when the gold candidate scores best. Behavioural scoring treats every architecture
alike and reads no internal state, so the same items compare Mamba, attention and hybrid rows. Every item carries
the predictions of the heuristic floors (data.floors for T1/T2, data.natural for T3), against which eval.probes
reports the model.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..files import write_text_atomic

SUITE_VERSION = 1
TASKS = ("T1", "T2", "T3")
HALVES = ("dev", "test")
SUITE_FILES = {half: f"{half}.jsonl" for half in HALVES}


@dataclasses.dataclass
class Item:
    id: str
    task: str
    prompt: str
    candidates: list[str]
    gold: str
    tail: str  # the text after the candidate to the end of its line, the same for every candidate
    heuristics: dict[str, str | None]
    strata: dict[str, Any]
    anchor: int | None = None  # offset in prompt of the binding the gold comes from (distance buckets)
    query: tuple[int, int] | None = None  # [start, end) in prompt of the identifier asked about (piece count)
    start: int = 0  # offset of prompt in its document; 0: the prompt starts the document (framing tokens)
    program: str | None = None  # T1/T2: the whole program, its readout completed with the gold
    hash: str | None = None  # T1/T2: data.decontam.ast_hash of the program

    def problems(self) -> list[str]:
        out = []
        if self.task not in TASKS:
            out.append(f"task {self.task!r}")
        if self.gold not in self.candidates or len(set(self.candidates)) != len(self.candidates):
            out.append("the gold must be one of distinct candidates")
        if not self.tail:
            out.append("empty tail")
        return [f"{self.id}: {p}" for p in out]


def write_items(path: Path, items: Iterable[Item]) -> int:
    lines = [json.dumps(dataclasses.asdict(item), ensure_ascii=False) for item in items]
    write_text_atomic(Path(path), "".join(line + "\n" for line in lines))
    return len(lines)


def read_items(path: Path) -> list[Item]:
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("query") is not None:
                record["query"] = tuple(record["query"])
            items.append(Item(**record))
    return items
