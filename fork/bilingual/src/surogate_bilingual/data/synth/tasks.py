"""Track S item builders (`bilingual-item/1`, one JSON object per line; MS §6.2, §9.1; SCI-01, SCI-08, SCI-11).

| family | variant | direction | question about | candidates (chance) |
|---|---|---|---|---|
| E1 | qa | zh>zh, en>en | a held-out P_zh (P_en) person, same language | K = 7 values + the gold's other-language form (1/8) |
| E2 | qa | zh>en, en>zh | a held-out P_zh (P_en) person, the other language | as E1 (1/8) |
| E3par | same_city | zh+en>zh, zh+en>en | A in P_zh and B in P_en: born in the same city? | yes / no (1/2) |
| E3par | month_sum | zh+en>zh, zh+en>en | (month number of A + of B) mod 12 | the 12 residues (1/12) |
| E3ctx | cross | zh>en>zh, en>zh>en | facts in L1, rules in L2, question in L3 | 7 weekdays + the gold's other-language form (1/8) |
| E3ctx | mono | zh>zh>zh, en>en>en | as cross with L1 = L2 = L3 | as cross (1/8) |
| E4 | translate | zh>en, en>zh | a P_par passage to translate | generation; slots scored exactly |
| E4 | switchback | zh>en>zh, en>zh>en | translate, then translate back | generation; slots per stage |

Construction rules that make every endpoint's chance level analytic and keep single facts uninformative:

- Gold positions (and, for E3-ctx, gold weekdays; for E3-par, labels) follow balanced cycles in random order, so
  every label is used floor(n / L) or ceil(n / L) times.
- E3-par `same_city`: the label is drawn first; then a city c is drawn uniformly from the cities held by at least
  one A-side and one B-side person; "same" takes A and B from c, "different" takes A from c and B from a uniformly
  drawn other city. Both cities are then uniform under both labels, so P(label | either city) = 1/2 by
  construction. `month_sum`: the label l is drawn first, then (month_A, month_B) uniformly among the available
  pairs with (m_A + m_B) mod 12 = l, so either month alone is uniform over the labels.
- E3-ctx: an item draws K = 7 colours out of 10, a uniformly random bijection from those colours onto the 7
  weekdays (the rules), and a target colour uniformly among the 7. Without the fact segment the target colour is
  uniform over the 7 rule colours, and without the rule segment the target's weekday is uniform over the 7
  weekdays: either segment alone leaves the gold uniform over the K answer-language candidates.
  [偏离] MS §6.2 lists the rule count (3–7) as a difficulty axis; with K = 7 candidates, fewer than 7 rules would
  let the rule segment alone narrow the gold to the rule weekdays (accuracy 1/n > 1/8 without the facts), which the
  ablation rule forbids. The rule count is therefore fixed at K; the difficulty axes are the distractor entities
  (0–3) and the segment order (docs/spec.md, I4 落地记录).
- Every E3-ctx entity is a fresh group code that appears in no other item and in no training document.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .entities import CodeAllocator, Person
from .render import ANSWER_SEP, JOIN, OTHER, Renderer, balanced_positions, join_segments, place
from .values import ATTRIBUTES

ITEM_FORMAT = "bilingual-item/1"
K = 7  # answer-language candidates of E1, E2 and E3-ctx

DIRECTIONS = {
    ("E1", "qa"): (("zh", "zh"), ("en", "en")),
    ("E2", "qa"): (("zh", "en"), ("en", "zh")),
    ("E3par", "same_city"): (("zh",), ("en",)),
    ("E3par", "month_sum"): (("zh",), ("en",)),
    ("E3ctx", "cross"): (("zh", "en", "zh"), ("en", "zh", "en")),
    ("E3ctx", "mono"): (("zh", "zh", "zh"), ("en", "en", "en")),
    ("E4", "translate"): (("zh", "en"), ("en", "zh")),
    ("E4", "switchback"): (("zh", "en"), ("en", "zh")),
}
FAMILIES = ("E1", "E2", "E3par", "E3ctx", "E4")


def direction_label(family: str, variant: str, langs: Sequence[str]) -> str:
    if family == "E3par":
        return f"zh+en>{langs[0]}"
    if (family, variant) == ("E4", "switchback"):
        return f"{langs[0]}>{langs[1]}>{langs[0]}"
    return ">".join(langs)


def item_id(family: str, variant: str, direction: str, split: str, index: int) -> str:
    return f"{family}-{variant}-{direction.replace('>', '.').replace('+', '')}-{split}-{index:06d}"


def _item(family, variant, direction, split, index, generator, **fields) -> dict[str, Any]:
    base = {
        "format": ITEM_FORMAT,
        "id": item_id(family, variant, direction, split, index),
        "family": family,
        "variant": variant,
        "direction": direction,
        "split": split,
        "candidates": [],
        "gold_index": None,
        "distractor_index": None,
        "chance": None,
        "knowledge_source": None,
        "generator": dict(generator),
    }
    base.update(fields)
    return base


def qa_items(
    family: str,
    group: Sequence[Person],
    question_lang: str,
    split: str,
    n: int,
    r: Renderer,
    rng: np.random.Generator,
    generator: Mapping,
) -> list[dict]:
    """E1 (question in the facts' language) or E2 (the other language) about the people of `group`."""
    knowledge = group[0].partition
    direction = f"{knowledge}>{question_lang}"
    facts = [(p, a) for p in group for a in ATTRIBUTES]
    order = rng.permutation(len(facts))
    positions = balanced_positions(n, K + 1, rng)
    items = []
    for i in range(n):
        person, attr = facts[int(order[i % len(facts)])]
        lex = r.lexicons[attr]
        gold = person.attrs[attr]
        others = [int(j) for j in rng.choice([j for j in range(len(lex)) if j != gold], size=K - 1, replace=False)]
        question, tid = r.qa(person, attr, question_lang, split, rng)
        candidates, gi, di = place(
            lex.surface(gold, question_lang),
            [lex.surface(j, question_lang) for j in others],
            positions[i],
            lex.surface(gold, OTHER[question_lang]),
            rng,
        )
        items.append(
            _item(
                family,
                "qa",
                direction,
                split,
                i,
                generator,
                template_ids={"question": tid},
                segments=[{"lang": question_lang, "role": "question", "text": question}],
                answer_lang=question_lang,
                answer_sep=ANSWER_SEP[question_lang],
                candidates=candidates,
                gold_index=gi,
                distractor_index=di,
                chance=1 / (K + 1),
                knowledge_source=knowledge,
                structured={"person": person.id, "attribute": attr, "value": lex.surface(gold, "en")},
                difficulty={"attribute": attr},
            )
        )
    return items


def _by_value(group: Sequence[Person], attr: str) -> dict[int, list[Person]]:
    out: dict[int, list[Person]] = {}
    for p in group:
        out.setdefault(p.attrs[attr], []).append(p)
    return out


def same_city_pairs(a_side, b_side, n, rng) -> list[tuple[Person, Person, int]]:
    """(A, B, label) with label 0 = same city, 1 = different, balanced; see the module docstring."""
    a_by, b_by = _by_value(a_side, "city"), _by_value(b_side, "city")
    eligible = sorted(set(a_by) & set(b_by))
    if len(eligible) < 2:
        raise ValueError("E3-par same_city needs two cities held on both sides")
    labels = balanced_positions(n, 2, rng)
    out = []
    for label in labels:
        c_a = eligible[int(rng.integers(len(eligible)))]
        c_b = c_a if label == 0 else [c for c in eligible if c != c_a][int(rng.integers(len(eligible) - 1))]
        a = a_by[c_a][int(rng.integers(len(a_by[c_a])))]
        b = b_by[c_b][int(rng.integers(len(b_by[c_b])))]
        out.append((a, b, label))
    return out


def month_number(month_index: int) -> int:
    return month_index + 1  # January = 1


def month_sum_label(a: Person, b: Person) -> int:
    return (month_number(a.attrs["month"]) + month_number(b.attrs["month"])) % 12


def month_sum_pairs(a_side, b_side, n, rng) -> list[tuple[Person, Person, int]]:
    a_by, b_by = _by_value(a_side, "month"), _by_value(b_side, "month")
    labels = balanced_positions(n, 12, rng)
    out = []
    for label in labels:
        pairs = [
            (ma, mb)
            for ma in sorted(a_by)
            for mb in sorted(b_by)
            if (month_number(ma) + month_number(mb)) % 12 == label
        ]
        if not pairs:
            raise ValueError(f"E3-par month_sum: no month pair gives residue {label}")
        ma, mb = pairs[int(rng.integers(len(pairs)))]
        out.append((a_by[ma][int(rng.integers(len(a_by[ma])))], b_by[mb][int(rng.integers(len(b_by[mb])))], label))
    return out


def par_items(variant, a_side, b_side, question_lang, split, n, r: Renderer, rng, generator) -> list[dict]:
    pairs = (same_city_pairs if variant == "same_city" else month_sum_pairs)(a_side, b_side, n, rng)
    lex = r.lexicons["yes_no" if variant == "same_city" else "mod12"]
    positions = balanced_positions(n, len(lex), rng)
    first_a = balanced_positions(n, 2, rng)
    direction = f"zh+en>{question_lang}"
    items = []
    for i, (a, b, label) in enumerate(pairs):
        names = (a.name, b.name) if first_a[i] == 0 else (b.name, a.name)
        question, tid = r.par(*names, variant, question_lang, split, rng)
        others = [lex.surface(j, question_lang) for j in range(len(lex)) if j != label]
        candidates, gi, _ = place(lex.surface(label, question_lang), others, positions[i], None, rng)
        attr = "city" if variant == "same_city" else "month"
        items.append(
            _item(
                "E3par",
                variant,
                direction,
                split,
                i,
                generator,
                template_ids={"question": tid},
                segments=[{"lang": question_lang, "role": "question", "text": question}],
                answer_lang=question_lang,
                answer_sep=ANSWER_SEP[question_lang],
                candidates=candidates,
                gold_index=gi,
                chance=1 / len(lex),
                knowledge_sources={"a": a.partition, "b": b.partition},
                structured={
                    "variant": variant,
                    "a": a.id,
                    "b": b.id,
                    "order": "ab" if first_a[i] == 0 else "ba",
                    "label": label,
                    "facts": [[a.id, attr], [b.id, attr]],
                },
                difficulty={},
            )
        )
    return items


def ctx_instance(rng, n_colors: int, distractors: tuple[int, int], codes: CodeAllocator, gold_weekday: int) -> dict:
    """The structure of one E3-ctx item (no text): colours, the rule bijection, the facts and the target."""
    colors = [int(c) for c in rng.choice(n_colors, size=K, replace=False)]
    target_color = colors[int(rng.integers(K))]
    rest_colors = [c for c in colors if c != target_color]
    rest_days = [d for d in range(K) if d != gold_weekday]
    rule = {target_color: gold_weekday}
    for c, d in zip(rest_colors, [rest_days[j] for j in rng.permutation(K - 1)], strict=True):
        rule[c] = d
    n_dis = int(rng.integers(distractors[0], distractors[1] + 1))
    entity_codes = codes.take(1 + n_dis)
    facts = [(entity_codes[0], target_color)] + [(code, colors[int(rng.integers(K))]) for code in entity_codes[1:]]
    fact_order = [facts[j] for j in rng.permutation(len(facts))]
    rule_order = [(c, rule[c]) for c in (colors[j] for j in rng.permutation(K))]
    return {
        "colors": colors,
        "target": entity_codes[0],
        "target_color": target_color,
        "gold": gold_weekday,
        "rule": rule,
        "facts": fact_order,
        "rules": rule_order,
        "n_distractors": n_dis,
    }


def ctx_segments(inst: Mapping, langs: Sequence[str], order: str, split: str, r: Renderer, rng):
    """(segments, template ids) of an E3-ctx instance; the question segment is last and has no answer."""
    l_fact, l_rule, l_q = langs
    fact_text, fact_tid = r.ctx_facts(inst["facts"], l_fact, split, rng)
    rule_text, rule_tid = r.ctx_rules(inst["rules"], l_rule, split, rng)
    q_text, q_tid = r.ctx_question(inst["target"], l_q, split, rng)
    body = [("fact", l_fact, fact_text), ("rule", l_rule, rule_text)]
    if order == "rule_fact":
        body.reverse()
    texts = join_segments([t for _, _, t in body] + [q_text])
    roles = [(role, lang) for role, lang, _ in body] + [("question", l_q)]
    segments = [{"lang": lang, "role": role, "text": text} for (role, lang), text in zip(roles, texts, strict=True)]
    return segments, {"fact": fact_tid, "rule": rule_tid, "question": q_tid}


def ctx_structured(inst: Mapping, r: Renderer) -> dict:
    color, day = r.lexicons["color"], r.lexicons["weekday"]
    return {
        "entity": inst["target"],
        "category": color.surface(inst["target_color"], "en"),
        "categories": [color.surface(c, "en") for c in inst["colors"]],
        "facts": [[code, color.surface(c, "en")] for code, c in inst["facts"]],
        "rule": {color.surface(c, "en"): day.surface(d, "en") for c, d in sorted(inst["rule"].items())},
    }


def ctx_items(
    variant,
    langs,
    split,
    n,
    r: Renderer,
    codes: CodeAllocator,
    rng,
    generator,
    *,
    n_colors: int,
    distractors: tuple[int, int],
) -> list[dict]:
    days = r.lexicons["weekday"]
    golds = balanced_positions(n, K, rng)
    positions = balanced_positions(n, K + 1, rng)
    orders = balanced_positions(n, 2, rng)
    direction = ">".join(langs)
    l_q = langs[2]
    items = []
    for i in range(n):
        inst = ctx_instance(rng, n_colors, distractors, codes, golds[i])
        order = ("fact_rule", "rule_fact")[orders[i]]
        segments, tids = ctx_segments(inst, langs, order, split, r, rng)
        others = [days.surface(d, l_q) for d in range(K) if d != inst["gold"]]
        candidates, gi, di = place(
            days.surface(inst["gold"], l_q), others, positions[i], days.surface(inst["gold"], OTHER[l_q]), rng
        )
        target_pos = [code for code, _ in inst["facts"]].index(inst["target"])
        items.append(
            _item(
                "E3ctx",
                variant,
                direction,
                split,
                i,
                generator,
                template_ids=tids,
                segments=segments,
                answer_lang=l_q,
                answer_sep=ANSWER_SEP[l_q],
                candidates=candidates,
                gold_index=gi,
                distractor_index=di,
                chance=1 / (K + 1),
                structured=ctx_structured(inst, r),
                difficulty={
                    "n_rules": K,
                    "n_distractor_entities": inst["n_distractors"],
                    "order": order,
                    "target_position": target_pos,
                },
            )
        )
    return items


def e4_slots(person: Person, attrs: Sequence[str], lang: str, r: Renderer) -> dict[str, str]:
    """The slots an E4 output must state exactly, in the output language: the name and every source fact."""
    return {"name": person.name, **{a: r.value(a, person.attrs[a], lang) for a in attrs}}


def e4_items(
    variant, langs, group: Sequence[Person], split, n, r: Renderer, rng, generator, *, attributes: tuple[int, int]
) -> list[dict]:
    src, tgt = langs
    direction = direction_label("E4", variant, langs)
    people = [group[int(j)] for j in rng.permutation(len(group))]
    items = []
    for i in range(n):
        person = people[i % len(people)]
        k = int(rng.integers(attributes[0], attributes[1] + 1))
        attrs = [ATTRIBUTES[j] for j in rng.permutation(len(ATTRIBUTES))[:k]]
        source = r.bio_sentences(person, src, split, rng, attrs)
        reference = r.bio_sentences(person, tgt, split, rng, attrs)
        instr = r.instruction("e4_instr", tgt, split, rng)
        stages = [
            {
                "answer_lang": tgt,
                "slots": e4_slots(person, attrs, tgt, r),
                "reference": JOIN[tgt].join(s for s, _ in reference),
            }
        ]
        texts = join_segments([JOIN[src].join(s for s, _ in source), instr.text])
        segments = [
            {"lang": src, "role": "source", "text": texts[0]},
            {"lang": tgt, "role": "instruction", "text": texts[1]},
            {"lang": tgt, "role": "answer", "text": None},
        ]
        tids = {
            "source": "+".join(t.id for _, t in source),
            "instruction": instr.id,
            "reference": "+".join(t.id for _, t in reference),
        }
        if variant == "switchback":
            back = r.instruction("e4_back", src, split, rng)
            back_ref = r.bio_sentences(person, src, split, rng, attrs)
            stages.append(
                {
                    "answer_lang": src,
                    "slots": e4_slots(person, attrs, src, r),
                    "reference": JOIN[src].join(s for s, _ in back_ref),
                }
            )
            segments += [
                {"lang": src, "role": "instruction", "text": back.text},
                {"lang": src, "role": "answer", "text": None},
            ]
            tids["instruction_back"] = back.id
            tids["reference_back"] = "+".join(t.id for _, t in back_ref)
        final = stages[-1]
        items.append(
            _item(
                "E4",
                variant,
                direction,
                split,
                i,
                generator,
                template_ids=tids,
                segments=segments,
                answer_lang=final["answer_lang"],
                answer_sep=ANSWER_SEP[final["answer_lang"]],
                slots=final["slots"],
                reference=final["reference"],
                stages=stages,
                structured={"person": person.id, "attributes": attrs, "source_lang": src},
                difficulty={"n_facts": k},
            )
        )
    return items
