"""Synthetic people and E3-ctx group codes (MS §6.2).

People. A name is a given name and a surname, each built from two invented syllables; it is written in the Latin
alphabet in both languages, so cross-language tests never mix in "transliterate a name never seen" (SCI-01). Names
are drawn without replacement, then the people are shuffled and dealt into the partitions P_seed, P_zh, P_en and
P_par; each of the six attributes is drawn uniformly and independently of the name and of the partition.

Sub-groups (all [建议] defaults, set in examples/track_s.toml):

- P_zh and P_en: `qa_train` (their QA is in the expert's training data, the mixed-training setting of Physics of
  LMs 3.1) and `heldout` (no QA in any training data; E1, E2 and E3-par ask about them). `heldout` is split into
  `dev` and `test` people.
- P_par: `train` (parallel biographies in D_parallel) and `dev` / `test` (E4 items only, never in training).
- P_seed: all `train`.

Group codes. Every E3-ctx entity is a fresh code such as "KQZ-47", drawn without replacement from 26^3 x 90 codes
through an affine permutation of the code space, so no code appears in two items or in any training document.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .values import ATTRIBUTES

GIVEN_HEADS = (
    "Al", "Bren", "Cal", "Dar", "El", "Fen", "Gar", "Hal", "Ith", "Jor", "Kel", "Lor", "Mar", "Nor", "Or", "Per",
    "Quin", "Ren", "Sel", "Tor", "Ul", "Val", "Wen", "Yor", "Zan", "Bel", "Cor", "Dal", "Ev", "Fal",
)  # fmt: skip
GIVEN_TAILS = ("a", "en", "ion", "is", "or", "an", "eth", "ia", "o", "us", "ard", "wyn")
SURNAME_HEADS = (
    "Ash", "Brem", "Cor", "Dun", "Ell", "Fair", "Grim", "Hart", "Ing", "Kess", "Lind", "Mor", "Nash", "Ost", "Pell",
    "Rook", "Stav", "Thorn", "Vay", "Wex", "Brack", "Cald", "Dray", "Fenn", "Gold", "Holl", "Kerr", "Lorr", "Mard",
    "Quill",
)  # fmt: skip
SURNAME_TAILS = ("ford", "wick", "ley", "ton", "berg", "mere", "dale", "holt", "by", "worth", "ridge", "stone")
PARTITIONS = ("seed", "zh", "en", "par")
CODE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CODE_SPACE = 26**3 * 90


class EntityError(ValueError):
    pass


def name_parts(index: int) -> tuple[int, int, int, int]:
    """(given head, given tail, surname head, surname tail) of a name index."""
    gh, rest = divmod(index, len(GIVEN_TAILS) * len(SURNAME_HEADS) * len(SURNAME_TAILS))
    gt, rest = divmod(rest, len(SURNAME_HEADS) * len(SURNAME_TAILS))
    sh, st = divmod(rest, len(SURNAME_TAILS))
    return gh, gt, sh, st


def name_of(index: int) -> str:
    gh, gt, sh, st = name_parts(index)
    return f"{GIVEN_HEADS[gh]}{GIVEN_TAILS[gt]} {SURNAME_HEADS[sh]}{SURNAME_TAILS[st]}"


NAME_SPACE = len(GIVEN_HEADS) * len(GIVEN_TAILS) * len(SURNAME_HEADS) * len(SURNAME_TAILS)


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    name_index: int
    partition: str
    group: str  # train | qa_train | dev | test
    attrs: Mapping[str, int] = field(default_factory=dict)  # attribute -> value index

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "name_index": self.name_index,
            "partition": self.partition,
            "group": self.group,
            "attrs": dict(self.attrs),
        }

    @classmethod
    def from_json(cls, d: Mapping) -> Person:
        return cls(d["id"], d["name"], d["name_index"], d["partition"], d["group"], dict(d["attrs"]))


def draw_names(n: int, rng: np.random.Generator) -> list[int]:
    if n > NAME_SPACE:
        raise EntityError(f"{n} names asked, the name space holds {NAME_SPACE}")
    return [int(i) for i in rng.choice(NAME_SPACE, size=n, replace=False)]


def draw_attrs(n: int, n_values: Mapping[str, int], rng: np.random.Generator) -> list[dict[str, int]]:
    """Each attribute uniform and independent of everything else (one integers() call per attribute)."""
    cols = {a: rng.integers(0, n_values[a], size=n) for a in ATTRIBUTES}
    return [{a: int(cols[a][i]) for a in ATTRIBUTES} for i in range(n)]


def _groups(partition: str, n: int, sizes: Mapping[str, float], rng: np.random.Generator) -> list[str]:
    if partition == "seed":
        return ["train"] * n
    if partition in ("zh", "en"):
        n_held = round(n * sizes["heldout_fraction"])
        n_dev = round(n_held * sizes["eval_dev_fraction"])
        labels = ["qa_train"] * (n - n_held) + ["dev"] * n_dev + ["test"] * (n_held - n_dev)
    else:
        n_eval = n - round(n * sizes["par_train_fraction"])
        n_dev = round(n_eval * sizes["eval_dev_fraction"])
        labels = ["train"] * (n - n_eval) + ["dev"] * n_dev + ["test"] * (n_eval - n_dev)
    return [labels[i] for i in rng.permutation(n)]


def draw_people(
    partitions: Mapping[str, int], n_values: Mapping[str, int], sizes: Mapping[str, float], rng: np.random.Generator
) -> list[Person]:
    counts = [int(partitions[p]) for p in PARTITIONS]
    total = sum(counts)
    names = draw_names(total, rng)
    attrs = draw_attrs(total, n_values, rng)
    order = rng.permutation(total)
    people, start = [], 0
    for partition, count in zip(PARTITIONS, counts, strict=True):
        members = order[start : start + count]
        start += count
        groups = _groups(partition, count, sizes, rng)
        for j, k in enumerate(members):
            k = int(k)
            pid = f"P{len(people):06d}"
            people.append(Person(pid, name_of(names[k]), names[k], partition, groups[j], attrs[k]))
    return people


def by_group(people: Sequence[Person], partition: str, group: str | None = None) -> list[Person]:
    return [p for p in people if p.partition == partition and (group is None or p.group == group)]


class CodeAllocator:
    """Fresh group codes "ABC-12": index i maps to (a*i + b) mod CODE_SPACE, a bijection since gcd(a, N) = 1."""

    def __init__(self, rng: np.random.Generator):
        while True:
            a = int(rng.integers(1, CODE_SPACE))
            if math.gcd(a, CODE_SPACE) == 1:
                break
        self.a, self.b, self.next = a, int(rng.integers(0, CODE_SPACE)), 0

    def take(self, k: int = 1) -> list[str]:
        if self.next + k > CODE_SPACE:
            raise EntityError(f"the group-code space ({CODE_SPACE}) is exhausted")
        out = [code_of((self.a * i + self.b) % CODE_SPACE) for i in range(self.next, self.next + k)]
        self.next += k
        return out


def code_of(index: int) -> str:
    letters, number = divmod(index, 90)
    c3 = CODE_LETTERS[letters % 26]
    c2 = CODE_LETTERS[(letters // 26) % 26]
    c1 = CODE_LETTERS[letters // 676]
    return f"{c1}{c2}{c3}-{number + 10}"


CODE_PATTERN = r"(?<![A-Za-z0-9])[A-Z]{3}-[1-9][0-9](?![A-Za-z0-9])"
