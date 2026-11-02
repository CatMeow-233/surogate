"""An independent checker for Track S items and training documents (MS §6.2 rule 6; I4-D7).

It never calls the renderer or the item builders. It knows only the closed lexicons (`values`), the people table
(`entities.jsonl`) and the text, and re-derives everything from them:

- E1 / E2: the gold is the person's attribute value in the answer language; the distractor is the same value in
  the other language; the question names the person.
- E3-par: the gold is "same city?" or (month number A + month number B) mod 12 from the people table; the
  question names both people.
- E3-ctx: the gold is parsed from the rendered text alone. Each fact sentence must hold exactly one group code and
  one colour; each rule sentence exactly one colour and one weekday; the question exactly one code and no colour or
  weekday; the fact segment no weekday and the rule segment no code. The gold is rule[fact[question code]], and it
  must equal both the structured fields and the candidate at `gold_index`. The ablation check is structural: the
  rules must be a bijection from the item's K colours onto the K weekday candidates and the target's colour one of
  them, so either segment alone leaves the gold uniform over the K answer-language candidates.
- E4: every slot is re-derived per stage, and the source passage must state the name and every fact in the source
  language.
- Training documents: from the text alone, each biography names one person (looked up by name) and states each
  attribute once; a fact counts as exposed only if the stated value is that person's value. The exposure counts
  come out of this text scan, not out of the generator's metadata.

The facts parametric items need ("the rendered text contains the facts the gold requires") are checked against
these text-derived exposures: E1 and E2 facts must be stated in the expert corpus of the person's partition, E3-par
facts in D_zh (A) and D_en (B).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .entities import CODE_PATTERN, Person
from .values import ATTRIBUTES, Lexicon, find_attributes

_CODE = re.compile(CODE_PATTERN)
# overlapping: in "Were Alen Ashford and ..." both "Were Alen" and "Alen Ashford" are candidate names
_NAME = re.compile(r"(?<![A-Za-z])(?=([A-Z][a-z]+ [A-Z][a-z]+)(?![A-Za-z]))")
_SENTENCE_END = re.compile(r"(?<=[。.])\s*")
OTHER = {"zh": "en", "en": "zh"}


class CheckError(AssertionError):
    pass


@dataclass
class CheckReport:
    items: int = 0
    by_family: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    ablation_uniform: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def fail(self, item_id: str, message: str) -> None:
        self.errors.append(f"{item_id}: {message}")


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s.strip()]


class Checker:
    def __init__(self, people: Iterable[Person], lexicons: Mapping[str, Lexicon]):
        self.people = {p.id: p for p in people}
        self.by_name = {p.name: p for p in self.people.values()}
        self.lex = lexicons
        self.exposures: dict[tuple[str, str, str], int] = Counter()  # (person, attribute, corpus) -> count
        self.doc_errors: list[str] = []

    def names_in(self, text: str) -> set[str]:
        """The people named in a text (candidate bigrams looked up in the people table)."""
        return {m.group(1) for m in _NAME.finditer(text)} & self.by_name.keys()

    # --- training documents ----------------------------------------------------------------------------------------

    def scan_bios(self, docs: Iterable[Mapping[str, Any]], corpus: str) -> None:
        """Count exposures from biography text: one name per biography, each attribute stated once and correctly."""
        for doc in docs:
            meta = doc.get("meta", {})
            if meta.get("family") not in ("bio", "parallel", "e4"):
                continue
            for seg in doc["segments"]:
                if seg.get("role") == "instruction":  # E4 format: an instruction names no one
                    continue
                text, lang = seg.get("text"), seg.get("lang")
                names = self.names_in(text)
                if len(names) != 1:
                    self.doc_errors.append(f"{doc['key']}: a biography names {sorted(names)}")
                    continue
                person = self.by_name[names.pop()]
                stated = find_attributes(text, lang, self.lex)
                for attr in ATTRIBUTES:
                    found = stated[attr]
                    if not found:
                        continue
                    if found != [person.attrs[attr]]:
                        self.doc_errors.append(f"{doc['key']}: {attr} of {person.id} stated as {found}")
                        continue
                    self.exposures[(person.id, attr, corpus)] += 1

    def exposure_counts(self, corpus: str) -> dict[tuple[str, str], int]:
        return {(p, a): n for (p, a, c), n in self.exposures.items() if c == corpus}

    def stated(self, person_id: str, attr: str, corpus: str) -> bool:
        return self.exposures.get((person_id, attr, corpus), 0) > 0

    # --- items -----------------------------------------------------------------------------------------------------

    def check_items(self, items: Iterable[Mapping[str, Any]], corpora: Mapping[str, str] | None = None) -> CheckReport:
        """`corpora` maps a partition (zh, en) to the corpus name whose biographies hold its facts; when given,
        parametric items also require their facts to be stated there."""
        report = CheckReport()
        for item in items:
            report.items += 1
            report.by_family[item["family"]] += 1
            try:
                getattr(self, f"_check_{item['family'].lower()}")(item, report, corpora)
            except (KeyError, IndexError, ValueError, CheckError) as err:
                report.fail(item["id"], f"{type(err).__name__}: {err}")
        return report

    def _candidates(self, item, gold: str, distractor: str | None, report: CheckReport) -> None:
        cands = item["candidates"]
        if cands[item["gold_index"]] != gold:
            report.fail(item["id"], f"gold {gold!r} is not the candidate at gold_index ({cands[item['gold_index']]!r})")
        if cands.count(gold) != 1:
            report.fail(item["id"], f"gold {gold!r} occurs {cands.count(gold)} times among the candidates")
        if distractor is not None:
            if item["distractor_index"] is None or cands[item["distractor_index"]] != distractor:
                report.fail(item["id"], f"the other-language form {distractor!r} is not at distractor_index")
        if abs(item["chance"] - 1 / len(cands)) > 1e-12:
            report.fail(item["id"], f"chance {item['chance']} != 1/{len(cands)}")

    def _question(self, item) -> tuple[str, str]:
        seg = [s for s in item["segments"] if s["role"] == "question"]
        if len(seg) != 1 or item["segments"][-1] is not seg[0]:
            raise CheckError("an item ends with exactly one question segment")
        return seg[0]["text"], seg[0]["lang"]

    def _check_e1(self, item, report, corpora, cross: bool = False) -> None:
        person = self.people[item["structured"]["person"]]
        attr = item["structured"]["attribute"]
        text, lang = self._question(item)
        if lang != item["answer_lang"]:
            report.fail(item["id"], "the answer language is not the question's")
        if cross != (lang != person.partition):
            report.fail(item["id"], f"question language {lang} vs. the facts' partition {person.partition}")
        if item["knowledge_source"] != person.partition:
            report.fail(item["id"], "knowledge_source is not the person's partition")
        if self.names_in(text) != {person.name}:
            report.fail(item["id"], "the question does not name exactly the person")
        lex = self.lex[attr]
        value = person.attrs[attr]
        self._candidates(item, lex.surface(value, lang), lex.surface(value, OTHER[lang]), report)
        if corpora and not self.stated(person.id, attr, corpora[person.partition]):
            report.fail(item["id"], f"no training biography states {attr} of {person.id}")

    def _check_e2(self, item, report, corpora) -> None:
        self._check_e1(item, report, corpora, cross=True)

    def _check_e3par(self, item, report, corpora) -> None:
        s = item["structured"]
        a, b = self.people[s["a"]], self.people[s["b"]]
        if (a.partition, b.partition) != ("zh", "en"):
            report.fail(item["id"], "E3-par pairs a P_zh person with a P_en person")
        text, lang = self._question(item)
        if self.names_in(text) != {a.name, b.name}:
            report.fail(item["id"], "the question does not name exactly A and B")
        if s["variant"] == "same_city":
            label = 0 if a.attrs["city"] == b.attrs["city"] else 1
            gold = self.lex["yes_no"].surface(label, lang)
            attr = "city"
        else:
            label = ((a.attrs["month"] + 1) + (b.attrs["month"] + 1)) % 12
            gold = self.lex["mod12"].surface(label, lang)
            attr = "month"
        if label != s["label"]:
            report.fail(item["id"], f"structured label {s['label']} != re-derived {label}")
        self._candidates(item, gold, None, report)
        if corpora:
            for p in (a, b):
                if not self.stated(p.id, attr, corpora[p.partition]):
                    report.fail(item["id"], f"no training biography states {attr} of {p.id}")

    def parse_ctx(self, item) -> dict[str, Any]:
        """(facts, rules, question code) parsed from the text of an E3-ctx item."""
        color, day = self.lex["color"], self.lex["weekday"]
        segs = {s["role"]: s for s in item["segments"]}
        facts: dict[str, int] = {}
        fact_seg, rule_seg = segs["fact"], segs["rule"]
        if day.find(fact_seg["text"], fact_seg["lang"]) or day.find(fact_seg["text"], OTHER[fact_seg["lang"]]):
            raise CheckError("the fact segment names a weekday")
        if _CODE.search(rule_seg["text"]):
            raise CheckError("the rule segment names a group code")
        for sentence in sentences(fact_seg["text"]):
            codes, colors = _CODE.findall(sentence), color.find(sentence, fact_seg["lang"])
            if not codes and not colors:
                continue  # the opening
            if len(codes) != 1 or len(colors) != 1:
                raise CheckError(f"fact sentence {sentence!r} holds {codes} and colours {colors}")
            if codes[0] in facts:
                raise CheckError(f"code {codes[0]} is stated twice")
            facts[codes[0]] = colors[0]
        rules: dict[int, int] = {}
        for sentence in sentences(rule_seg["text"]):
            colors, days = color.find(sentence, rule_seg["lang"]), day.find(sentence, rule_seg["lang"])
            if not colors and not days:
                continue
            if len(colors) != 1 or len(days) != 1:
                raise CheckError(f"rule sentence {sentence!r} holds colours {colors} and weekdays {days}")
            if colors[0] in rules:
                raise CheckError("a colour has two rules")
            rules[colors[0]] = days[0]
        q_text, q_lang = self._question(item)
        codes = _CODE.findall(q_text)
        if len(codes) != 1 or color.find(q_text, q_lang) or day.find(q_text, q_lang):
            raise CheckError("the question must name one code and no colour or weekday")
        return {"facts": facts, "rules": rules, "code": codes[0]}

    def _check_e3ctx(self, item, report, corpora) -> None:
        color, day = self.lex["color"], self.lex["weekday"]
        parsed = self.parse_ctx(item)
        facts, rules, code = parsed["facts"], parsed["rules"], parsed["code"]
        if code not in facts:
            raise CheckError(f"the question's code {code} has no fact")
        gold_day = rules.get(facts[code])
        if gold_day is None:
            raise CheckError("the target's colour has no rule")
        lang = item["answer_lang"]
        self._candidates(item, day.surface(gold_day, lang), day.surface(gold_day, OTHER[lang]), report)
        s = item["structured"]
        if (s["entity"], s["category"]) != (code, color.surface(facts[code], "en")):
            report.fail(item["id"], "structured entity/category disagree with the text")
        text_rule = {color.surface(c, "en"): day.surface(d, "en") for c, d in rules.items()}
        if text_rule != s["rule"]:
            report.fail(item["id"], "structured rules disagree with the text")
        # structural ablation: rules are a bijection from the item's colours onto all K weekday candidates, and every
        # fact colour (the target's included) is one of them
        k = len(item["candidates"]) - 1
        answer_days = sorted(
            day.index_of(c, lang) for i, c in enumerate(item["candidates"]) if i != item["distractor_index"]
        )
        if sorted(rules.values()) != answer_days or len(rules) != k or not set(facts.values()) <= set(rules):
            report.fail(
                item["id"],
                "the rules are not a bijection onto the candidates: one segment alone would not leave the gold uniform",
            )
        else:
            report.ablation_uniform += 1

    def _check_e4(self, item, report, corpora) -> None:
        person = self.people[item["structured"]["person"]]
        attrs = item["structured"]["attributes"]
        src = item["structured"]["source_lang"]
        source = item["segments"][0]
        if source["role"] != "source" or source["lang"] != src:
            raise CheckError("an E4 item starts with its source segment")
        if person.name not in source["text"]:
            report.fail(item["id"], "the source does not name the person")
        stated = find_attributes(source["text"], src, self.lex)
        for attr in attrs:
            if stated[attr] != [person.attrs[attr]]:
                report.fail(item["id"], f"the source does not state {attr} once and correctly")
        for stage in item["stages"]:
            lang = stage["answer_lang"]
            want = {"name": person.name, **{a: self.lex[a].surface(person.attrs[a], lang) for a in attrs}}
            if stage["slots"] != want:
                report.fail(item["id"], f"slots of the {lang} stage disagree with the people table")
            stated = find_attributes(stage["reference"], lang, self.lex)
            for attr in attrs:
                if stated[attr] != [person.attrs[attr]]:
                    report.fail(item["id"], f"the {lang} reference does not state {attr}")
        if item["slots"] != item["stages"][-1]["slots"]:
            report.fail(item["id"], "top-level slots are not the final stage's")
        if person.partition != "par" or person.group not in ("dev", "test"):
            report.fail(item["id"], "E4 people are held-out P_par people")


def label_shares(items: Sequence[Mapping[str, Any]], key) -> dict[Any, float]:
    counts = Counter(key(item) for item in items)
    return {k: v / len(items) for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))}


def group_items(items: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str, str, str], list]:
    out: dict[tuple[str, str, str, str], list] = defaultdict(list)
    for item in items:
        out[(item["family"], item["variant"], item["direction"], item["split"])].append(item)
    return out
