"""The Track S template registry, loaded from the `[templates]` tables of examples/track_s.toml (MS §6.2).

A table `[templates.<lang>.<role>]` or `[templates.<lang>.<role>.<key>]` lists template strings under `train`,
`dev` and `test`. Ids are derived from the position: `<lang>-<role>[-<key>]-<split><nn>`, e.g. `zh-qa-city-test01`,
so the lists are append-only: inserting a string in the middle renames every later template.

| role | key | placeholders | used by |
|---|---|---|---|
| bio | attribute | {name} {value} | biographies (train); E4 source passages (dev/test) |
| qa | attribute | {name} | E1, E2 question cores |
| frame | - | {q} | wraps every question; ends where the answer starts |
| par | same_city, month_sum | {a} {b} | E3-par question cores |
| ctx_fact_intro, ctx_rule_intro | - | - | E3-ctx segment openings |
| ctx_fact | - | {entity} {category} | one E3-ctx fact sentence |
| ctx_rule | - | {category} {value} | one E3-ctx rule sentence |
| ctx_q | - | {entity} | E3-ctx question core |
| e4_instr, e4_back | - | - | E4 instructions (translate; translate back) |

A family's template id for one role is composed from its parts (e.g. `frame + qa`), so the number of templates a
split offers is a product of pool sizes; each part pool is split-disjoint, so a test template's every part is
unseen in training (I4-D7). Load-time checks:

- every template has exactly its role's placeholders;
- in zh templates a Latin placeholder ({name}, {a}, {b}, {entity}) is never glued to a Han character, so a name
  tokenizes the same way in both languages (the byte-level pre-tokenizer would merge "Morisk出生" into one piece);
- no template contains a lexicon surface the checker parses for its role (E3-ctx templates contain no colour or
  weekday in either language, bio and QA templates no value of their attribute), so parsing the rendered text is
  unambiguous;
- ids are unique.
"""

from __future__ import annotations

import string
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import permutations
from math import prod

from ..docs import is_han
from .values import ATTRIBUTES, lexicon

LANGS = ("zh", "en")
SPLITS = ("train", "dev", "test")
LATIN_PLACEHOLDERS = {"name", "a", "b", "entity"}
ROLES = {
    "bio": ({"name", "value"}, ATTRIBUTES),
    "qa": ({"name"}, ATTRIBUTES),
    "frame": ({"q"}, None),
    "par": ({"a", "b"}, ("same_city", "month_sum")),
    "ctx_fact_intro": (set(), None),
    "ctx_fact": ({"entity", "category"}, None),
    "ctx_rule_intro": (set(), None),
    "ctx_rule": ({"category", "value"}, None),
    "ctx_q": ({"entity"}, None),
    "e4_instr": (set(), None),
    "e4_back": (set(), None),
}
CTX_ROLES = ("ctx_fact_intro", "ctx_fact", "ctx_rule_intro", "ctx_rule", "ctx_q")


class TemplateError(ValueError):
    pass


@dataclass(frozen=True)
class Template:
    id: str
    lang: str
    role: str
    key: str | None
    split: str
    text: str

    def render(self, **values: str) -> str:
        return self.text.format(**values)


def placeholders(text: str) -> set[str]:
    try:
        return {name for _, name, _, _ in string.Formatter().parse(text) if name is not None}
    except ValueError as err:
        raise TemplateError(f"bad template {text!r}: {err}") from None


def _check_spacing(t: Template) -> None:
    if t.lang != "zh":
        return
    for name in LATIN_PLACEHOLDERS:
        token = "{" + name + "}"
        start = t.text.find(token)
        while start >= 0:
            before = t.text[start - 1] if start > 0 else ""
            after_pos = start + len(token)
            after = t.text[after_pos] if after_pos < len(t.text) else ""
            if (before and is_han(before)) or (after and is_han(after)):
                raise TemplateError(f"{t.id}: {token} touches a Han character; put a space between them")
            start = t.text.find(token, after_pos)


def _forbidden(t: Template) -> list[str]:
    """Lexicon surfaces this template must not contain (checked in both languages for E3-ctx)."""
    if t.role in CTX_ROLES or t.role == "frame":
        return [s for name in ("color", "weekday") for lang in LANGS for s in lexicon(name).surfaces(lang)]
    if t.role in ("bio", "qa"):
        return lexicon(t.key).surfaces(t.lang)
    return []


class Registry:
    def __init__(self, templates: list[Template]):
        ids = [t.id for t in templates]
        if len(set(ids)) != len(ids):
            raise TemplateError("duplicate template ids")
        self.templates = templates
        self.by_id = {t.id: t for t in templates}
        self._pools: dict[tuple, list[Template]] = {}
        for t in templates:
            self._pools.setdefault((t.lang, t.role, t.key, t.split), []).append(t)

    def pool(self, lang: str, role: str, split: str, key: str | None = None) -> list[Template]:
        found = self._pools.get((lang, role, key, split), [])
        if not found:
            raise TemplateError(f"no {split} templates for {lang}/{role}" + (f"/{key}" if key else ""))
        return found

    def size(self, lang: str, role: str, split: str, key: str | None = None) -> int:
        return len(self._pools.get((lang, role, key, split), []))

    def family_count(self, family: str, split: str, e4_attributes: tuple[int, int] = (3, 6)) -> int:
        """How many composed template ids a split offers for one family (both languages, every role)."""
        total = 0
        for lang in LANGS:
            size = {
                (role, key): self.size(lang, role, split, key)
                for (lg, role, key, sp) in self._pools
                if lg == lang and sp == split
            }

            def s(role, key=None, size=size):
                return size.get((role, key), 0)

            if family in ("E1", "E2"):
                total += s("frame") * sum(s("qa", a) for a in ATTRIBUTES)
            elif family == "E3par":
                total += s("frame") * (s("par", "same_city") + s("par", "month_sum"))
            elif family == "E3ctx":
                total += s("ctx_fact_intro") * s("ctx_fact") + s("ctx_rule_intro") * s("ctx_rule")
                total += s("frame") * s("ctx_q")
            elif family == "E4":
                sizes = {a: s("bio", a) for a in ATTRIBUTES}
                lo, hi = e4_attributes
                # a source passage is an ordered attribute sequence of length k, each sentence from its pool
                passages = sum(
                    prod(sizes[a] for a in order) for k in range(lo, hi + 1) for order in permutations(ATTRIBUTES, k)
                )
                total += passages + s("e4_instr") + s("e4_back")
            else:
                raise TemplateError(f"unknown family {family!r}")
        return total


def load_registry(tables: Mapping) -> Registry:
    templates: list[Template] = []
    for lang in sorted(tables):
        if lang not in LANGS:
            raise TemplateError(f"templates: unknown language {lang!r}")
        for role, body in sorted(tables[lang].items()):
            if role not in ROLES:
                raise TemplateError(f"templates.{lang}: unknown role {role!r}; known: {sorted(ROLES)}")
            wanted, keys = ROLES[role]
            groups = [(None, body)] if keys is None else sorted(body.items())
            for key, lists in groups:
                if keys is not None and key not in keys:
                    raise TemplateError(f"templates.{lang}.{role}: unknown key {key!r}; known: {keys}")
                unknown = set(lists) - set(SPLITS)
                if unknown:
                    raise TemplateError(f"templates.{lang}.{role}: unknown splits {sorted(unknown)}")
                for split in SPLITS:
                    for i, text in enumerate(lists.get(split, []), 1):
                        tid = f"{lang}-{role}" + (f"-{key}" if key else "") + f"-{split}{i:02d}"
                        t = Template(tid, lang, role, key, split, text)
                        if placeholders(text) != wanted:
                            raise TemplateError(
                                f"{tid}: placeholders {sorted(placeholders(text))}, expected {sorted(wanted)}"
                            )
                        _check_spacing(t)
                        literal = text
                        for name in wanted:
                            literal = literal.replace("{" + name + "}", "\0")
                        for surface in _forbidden(t):
                            if surface in literal:
                                raise TemplateError(f"{tid}: contains the lexicon surface {surface!r}")
                        templates.append(t)
    registry = Registry(templates)
    # a text may repeat across keys of one split (e.g. "{name} 生于{value}。" for month and city), never across
    # splits of one role, whatever the key: a test template's surface is never a training surface
    for lang in LANGS:
        for role in ROLES:
            sets = [{t.text for t in templates if (t.lang, t.role, t.split) == (lang, role, s)} for s in SPLITS]
            for (i, a), (j, b) in ((p, q) for p in enumerate(sets) for q in enumerate(sets) if p[0] < q[0]):
                if a & b:
                    raise TemplateError(
                        f"templates.{lang}.{role}: {sorted(a & b)[0]!r} is in both {SPLITS[i]} and {SPLITS[j]}"
                    )
    return registry
