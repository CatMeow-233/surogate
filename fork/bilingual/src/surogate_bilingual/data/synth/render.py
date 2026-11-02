"""Rendering Track S text from templates and values: biographies, QA, E3-par questions, E3-ctx segments and E4
passages (MS §6.2). Pure functions of (registry, lexicons, rng); no I/O.

Conventions shared with the checker and the scorer:

- Sentences are joined with "" in zh and " " in en.
- A segment that is followed by another segment ends with "\\n"; the prompt is the plain concatenation of segment
  texts, and tokenizing segment by segment gives the same ids as tokenizing the prompt (the byte-level
  pre-tokenizer always splits at "\\n").
- A question segment is a frame wrapped around a question core and ends where the answer starts; the answer is
  `ANSWER_SEP[lang] + candidate` ("" in zh, " " in en), so "A:" + " London" tokenizes as in training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .entities import Person
from .templates import Registry, Template
from .values import ATTRIBUTES, Lexicon

JOIN = {"zh": "", "en": " "}
ANSWER_SEP = {"zh": "", "en": " "}
OTHER = {"zh": "en", "en": "zh"}


def pick(pool: Sequence[Template], rng: np.random.Generator) -> Template:
    return pool[int(rng.integers(len(pool)))]


@dataclass
class Renderer:
    registry: Registry
    lexicons: Mapping[str, Lexicon]

    def value(self, attr: str, index: int, lang: str) -> str:
        return self.lexicons[attr].surface(index, lang)

    # --- biographies -----------------------------------------------------------------------------------------------

    def bio_sentences(
        self, person: Person, lang: str, split: str, rng: np.random.Generator, attrs: Sequence[str] = ATTRIBUTES
    ) -> list[tuple[str, Template]]:
        """One sentence per attribute, in the given order, each from a random template of the split."""
        out = []
        for attr in attrs:
            t = pick(self.registry.pool(lang, "bio", split, attr), rng)
            out.append((t.render(name=person.name, value=self.value(attr, person.attrs[attr], lang)), t))
        return out

    def bio(
        self, person: Person, lang: str, split: str, rng: np.random.Generator, attrs: Sequence[str] | None = None
    ) -> tuple[str, list[str], list[str]]:
        """(text, template ids, attribute order): every attribute stated once, in a random order."""
        order = list(attrs) if attrs is not None else [ATTRIBUTES[i] for i in rng.permutation(len(ATTRIBUTES))]
        sentences = self.bio_sentences(person, lang, split, rng, order)
        return JOIN[lang].join(s for s, _ in sentences), [t.id for _, t in sentences], order

    # --- questions -------------------------------------------------------------------------------------------------

    def framed(self, core: str, lang: str, split: str, rng: np.random.Generator) -> tuple[str, Template]:
        frame = pick(self.registry.pool(lang, "frame", split), rng)
        return frame.render(q=core), frame

    def qa(self, person: Person, attr: str, lang: str, split: str, rng: np.random.Generator) -> tuple[str, str]:
        """(question text ending where the answer starts, composed template id)."""
        core = pick(self.registry.pool(lang, "qa", split, attr), rng)
        text, frame = self.framed(core.render(name=person.name), lang, split, rng)
        return text, f"{frame.id}+{core.id}"

    def par(self, first: str, second: str, variant: str, lang: str, split: str, rng) -> tuple[str, str]:
        core = pick(self.registry.pool(lang, "par", split, variant), rng)
        text, frame = self.framed(core.render(a=first, b=second), lang, split, rng)
        return text, f"{frame.id}+{core.id}"

    # --- E3-ctx ----------------------------------------------------------------------------------------------------

    def ctx_facts(self, facts: Sequence[tuple[str, int]], lang: str, split: str, rng) -> tuple[str, str]:
        """The fact segment: an opening and one sentence per (entity code, colour index), one template for all."""
        intro = pick(self.registry.pool(lang, "ctx_fact_intro", split), rng)
        core = pick(self.registry.pool(lang, "ctx_fact", split), rng)
        sentences = [core.render(entity=code, category=self.value("color", c, lang)) for code, c in facts]
        return intro.text + JOIN[lang] + JOIN[lang].join(sentences), f"{intro.id}+{core.id}"

    def ctx_rules(self, rules: Sequence[tuple[int, int]], lang: str, split: str, rng) -> tuple[str, str]:
        """The rule segment: an opening and one sentence per (colour index, weekday index)."""
        intro = pick(self.registry.pool(lang, "ctx_rule_intro", split), rng)
        core = pick(self.registry.pool(lang, "ctx_rule", split), rng)
        sentences = [core.render(category=self.value("color", c, lang), value=self.value("weekday", v, lang))
                     for c, v in rules]  # fmt: skip
        return intro.text + JOIN[lang] + JOIN[lang].join(sentences), f"{intro.id}+{core.id}"

    def ctx_question(self, code: str, lang: str, split: str, rng) -> tuple[str, str]:
        core = pick(self.registry.pool(lang, "ctx_q", split), rng)
        text, frame = self.framed(core.render(entity=code), lang, split, rng)
        return text, f"{frame.id}+{core.id}"

    # --- E4 --------------------------------------------------------------------------------------------------------

    def instruction(self, role: str, lang: str, split: str, rng) -> Template:
        return pick(self.registry.pool(lang, role, split), rng)


def join_segments(texts: Sequence[str]) -> list[str]:
    """Segment texts as stored: every segment but the last ends with a newline."""
    return [t + "\n" if i < len(texts) - 1 else t for i, t in enumerate(texts)]


def balanced_positions(n_items: int, n_positions: int, rng: np.random.Generator) -> list[int]:
    """Gold positions for n items: every position used floor or ceil of n / n_positions times, in random order."""
    cycle = np.tile(np.arange(n_positions), n_items // n_positions + 1)[:n_items]
    return [int(x) for x in rng.permutation(cycle)]


def place(
    gold: str, others: Sequence[str], gold_pos: int, distractor: str | None, rng
) -> tuple[list[str], int, int | None]:
    """Candidates with the gold at gold_pos and the distractor (if any) at a random other position."""
    n = len(others) + 1 + (distractor is not None)
    slots: list[str | None] = [None] * n
    slots[gold_pos] = gold
    free = [i for i in range(n) if i != gold_pos]
    distractor_pos = None
    if distractor is not None:
        distractor_pos = free.pop(int(rng.integers(len(free))))
        slots[distractor_pos] = distractor
    for i, value in zip(free, [others[j] for j in rng.permutation(len(others))], strict=True):
        slots[i] = value
    return [s for s in slots if s is not None], gold_pos, distractor_pos
