"""`python -m surogate_bilingual synth`: build Track S from examples/track_s.toml (MS §6.2; I4-D7).

Output directory:

```
<out>/
  entities.jsonl               people (id, name, partition, group, attribute value indices)
  seed/track_s.jsonl           the D_seed component: P_seed biographies (zh and en) and QA, E3-par and E3-ctx
                               format samples (train templates, fresh entities, all eight E3-ctx language orders)
  zh/track_s.jsonl             the D_zh component: P_zh biographies, zh QA of the qa_train half, zh mono E3-ctx
  en/track_s.jsonl             the D_en component (symmetric)
  parallel/track_s.jsonl       the D_parallel component: zh/en biography pairs of the P_par train group
  items/<family>.<split>.jsonl dev and test items (bilingual-item/1)
  datacard.json                bilingual-datacard/1, kind "track_s"
```

The same (generator version, spec bytes, profile, seed) gives byte-identical files: every random stream is a
numpy Generator seeded with (seed, stream id), JSON is canonical, and nothing records a time. `datacard.json`
holds the sha256 of every other file. After writing, the build runs the independent checker and the leak tests and
refuses (non-zero exit, no data card) when any of them fails.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
import zlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...canonical import atomic_write_bytes, atomic_write_json, canonical_json, sha256_bytes, sha256_file
from ..docs import Doc, Segment, doc_line
from . import leak
from .checker import Checker, group_items, label_shares
from .entities import CodeAllocator, Person, by_group, draw_people
from .render import ANSWER_SEP, Renderer, join_segments
from .tasks import (
    DIRECTIONS,
    FAMILIES,
    K,
    ctx_instance,
    ctx_items,
    ctx_segments,
    e4_items,
    month_sum_label,
    par_items,
    qa_items,
)
from .templates import load_registry
from .values import ATTRIBUTES, lexicon

TRACK_S_VERSION = "1.0.0"
CARD_FORMAT = "bilingual-datacard/1"
COMPONENTS = ("seed", "zh", "en", "parallel")
SPLITS = ("dev", "test")
LEAK_ALPHA = 0.001
INDEPENDENCE_DRAWS = 20_000


class BuildError(RuntimeError):
    pass


def stream(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([int(seed), zlib.crc32(name.encode("utf-8"))])


def load_spec(path: Path, profile: str) -> tuple[dict, dict, bytes]:
    data = Path(path).read_bytes()
    spec = tomllib.loads(data.decode("utf-8"))
    if profile not in spec.get("profile", {}):
        raise BuildError(f"{path}: no [profile.{profile}]; known: {sorted(spec.get('profile', {}))}")
    return spec, spec["profile"][profile], data


def lexicons_for(prof: Mapping) -> dict:
    n = prof["values"]
    lex = {a: lexicon(a, n.get(a)) for a in ATTRIBUTES}
    lex.update({name: lexicon(name) for name in ("color", "weekday", "yes_no", "mod12")})
    for attr in ATTRIBUTES:
        if len(lex[attr]) < K:
            raise BuildError(f"attribute {attr} has {len(lex[attr])} values; candidate sets need at least {K}")
    if prof["ctx_categories"] != K:
        raise BuildError(f"ctx_categories must equal K = {K}: the rules are a bijection onto the K weekdays "
                         "(tasks.py; a smaller rule set would let the rule segment alone narrow the gold)")  # fmt: skip
    return lex


# --- training documents -------------------------------------------------------------------------------------------


class DocWriter:
    def __init__(self, component: str):
        self.component = component
        self.docs: list[Doc] = []
        self.counts: Counter = Counter()
        self.template_ids: dict[str, set[str]] = defaultdict(set)

    def add(self, family: str, segments: Sequence[Segment], *, kind: str = "synthetic", entity_id=None,
            pair_id=None, persons=(), facts=(), template_ids=(), item_family: str | None = None) -> None:  # fmt: skip
        key = f"track_s/{self.component}/{family}/{self.counts[family]:08d}"
        self.counts[family] += 1
        meta = {"family": family, "persons": list(persons), "facts": [list(f) for f in facts],
                "template_ids": list(template_ids)}  # fmt: skip
        self.docs.append(Doc(key, kind, tuple(segments), source=f"track_s_{self.component}", pair_id=pair_id,
                             entity_id=entity_id, meta=meta))  # fmt: skip
        if item_family:
            self.template_ids[item_family].update(template_ids)


def _bios(w: DocWriter, r: Renderer, people: Sequence[Person], lang: str, exposures: int, min_rewrites: int, rng,
          rewrite_stats: list) -> None:  # fmt: skip
    for person in people:
        seen: set[str] = set()
        for k in range(exposures):
            for _ in range(50):
                text, tids, _ = r.bio(person, lang, "train", rng)
                if k >= min_rewrites or text not in seen:
                    break
            else:
                raise BuildError(f"cannot draw {min_rewrites} distinct biographies of {person.id}")
            seen.add(text)
            w.add("bio", [Segment(text=text, lang=lang, role="bio")], entity_id=person.id, persons=[person.id],
                  facts=[(person.id, a) for a in ATTRIBUTES], template_ids=tids, item_family="E4")  # fmt: skip
        rewrite_stats.append(len(seen))


def _qa_docs(w: DocWriter, r: Renderer, people, lang: str, per_fact: int, rng) -> None:
    for person in people:
        for attr in ATTRIBUTES:
            for _ in range(per_fact):
                q, tid = r.qa(person, attr, lang, "train", rng)
                answer = r.value(attr, person.attrs[attr], lang)
                w.add("qa", [Segment(text=q + ANSWER_SEP[lang] + answer, lang=lang, role="qa")], entity_id=person.id,
                      persons=[person.id], template_ids=[tid], item_family="E1")  # fmt: skip


def _par_docs(w: DocWriter, r: Renderer, people: Sequence[Person], n: int, rng) -> None:
    for i in range(n):
        j, k = (int(x) for x in rng.choice(len(people), size=2, replace=False))
        a, b = people[j], people[k]
        variant = ("same_city", "month_sum")[i % 2]
        lang = ("zh", "en")[(i // 2) % 2]
        if variant == "same_city":
            answer = r.lexicons["yes_no"].surface(0 if a.attrs["city"] == b.attrs["city"] else 1, lang)
        else:
            answer = r.lexicons["mod12"].surface(month_sum_label(a, b), lang)
        q, tid = r.par(a.name, b.name, variant, lang, "train", rng)
        w.add("par", [Segment(text=q + ANSWER_SEP[lang] + answer, lang=lang, role="qa")], persons=[a.id, b.id],
              template_ids=[tid], item_family="E3par")  # fmt: skip


LANG_ORDERS = tuple((a, b, c) for a in ("zh", "en") for b in ("zh", "en") for c in ("zh", "en"))


def _ctx_docs(w: DocWriter, r: Renderer, n: int, orders: Sequence[tuple[str, str, str]], codes, prof, rng) -> None:
    days = r.lexicons["weekday"]
    for i in range(n):
        langs = orders[i % len(orders)]
        inst = ctx_instance(rng, len(r.lexicons["color"]), tuple(prof["ctx_distractors"]), codes, int(rng.integers(K)))
        segments, tids = ctx_segments(inst, langs, ("fact_rule", "rule_fact")[int(rng.integers(2))], "train", r, rng)
        answer = days.surface(inst["gold"], langs[2])
        segs = [Segment(text=s["text"], lang=s["lang"], role=s["role"]) for s in segments[:-1]]
        last = segments[-1]
        segs.append(Segment(text=last["text"] + ANSWER_SEP[last["lang"]] + answer, lang=last["lang"], role="qa"))
        w.add("ctx", segs, template_ids=list(tids.values()), item_family="E3ctx")


def _parallel_docs(w: DocWriter, r: Renderer, people: Sequence[Person], pairs: int, rng) -> None:
    for person in people:
        for j in range(pairs):
            order = [ATTRIBUTES[k] for k in rng.permutation(len(ATTRIBUTES))]
            texts = {lang: r.bio(person, lang, "train", rng, order) for lang in ("zh", "en")}
            src, tgt = (("zh", "en"), ("en", "zh"))[j % 2]
            body = join_segments([texts[src][0], texts[tgt][0]])
            w.add("parallel", [Segment(text=body[0], lang=src, role="source"), Segment(text=body[1], lang=tgt, role="target")],
                  kind="parallel", entity_id=person.id, pair_id=f"{person.id}-{j}", persons=[person.id],
                  facts=[(person.id, a) for a in ATTRIBUTES], template_ids=texts[src][1] + texts[tgt][1],
                  item_family="E4")  # fmt: skip


def training_docs(people: Sequence[Person], r: Renderer, prof: Mapping, seed: int, codes: CodeAllocator):
    writers = {c: DocWriter(c) for c in COMPONENTS}
    rewrites: dict[str, list] = defaultdict(list)
    exp, qa = prof["bio_exposures"], prof["qa_per_fact"]
    p_seed = by_group(people, "seed")
    for lang in ("zh", "en"):
        _bios(writers["seed"], r, p_seed, lang, exp["seed"][lang], prof["min_rewrites"], stream(seed, f"bio/seed/{lang}"),
              rewrites[f"seed/{lang}"])  # fmt: skip
        _qa_docs(writers["seed"], r, p_seed, lang, qa["seed"], stream(seed, f"qa/seed/{lang}"))
    _par_docs(writers["seed"], r, p_seed, prof["seed_par_samples"], stream(seed, "par/seed"))
    _ctx_docs(writers["seed"], r, prof["seed_ctx_samples"], LANG_ORDERS, codes, prof, stream(seed, "ctx/seed"))
    for lang in ("zh", "en"):
        w = writers[lang]
        _bios(w, r, by_group(people, lang), lang, exp[lang][lang], prof["min_rewrites"], stream(seed, f"bio/{lang}"),
              rewrites[f"{lang}/{lang}"])  # fmt: skip
        _qa_docs(w, r, by_group(people, lang, "qa_train"), lang, qa[lang], stream(seed, f"qa/{lang}"))
        _ctx_docs(w, r, prof["expert_ctx_samples"], [(lang,) * 3], codes, prof, stream(seed, f"ctx/{lang}"))
    _parallel_docs(writers["parallel"], r, by_group(people, "par", "train"), prof["parallel_pairs_per_person"],
                   stream(seed, "parallel"))  # fmt: skip
    for c, w in writers.items():
        order = stream(seed, f"shuffle/{c}").permutation(len(w.docs))
        w.docs = [w.docs[int(i)] for i in order]
    return writers, {k: (min(v), max(v)) for k, v in rewrites.items() if v}


# --- items --------------------------------------------------------------------------------------------------------


def eval_items(people: Sequence[Person], r: Renderer, prof: Mapping, seed: int, codes: CodeAllocator,
               generator: Mapping) -> dict[tuple[str, str], list[dict]]:  # fmt: skip
    out: dict[tuple[str, str], list[dict]] = {}
    for split in SPLITS:
        n = prof["items"][split]
        items: dict[str, list[dict]] = defaultdict(list)
        for (family, variant), directions in DIRECTIONS.items():
            for langs in directions:
                rng = stream(seed, f"items/{family}/{variant}/{'.'.join(langs)}/{split}")
                if family in ("E1", "E2"):
                    group = by_group(people, langs[0], split)
                    items[family] += qa_items(family, group, langs[1], split, n, r, rng, generator)
                elif family == "E3par":
                    a_side, b_side = by_group(people, "zh", split), by_group(people, "en", split)
                    items[family] += par_items(variant, a_side, b_side, langs[0], split, n, r, rng, generator)
                elif family == "E3ctx":
                    items[family] += ctx_items(variant, langs, split, n, r, codes, rng, generator,
                                               n_colors=len(r.lexicons["color"]),
                                               distractors=tuple(prof["ctx_distractors"]))  # fmt: skip
                else:
                    items[family] += e4_items(variant, langs, by_group(people, "par", split), split, n, r, rng,
                                              generator, attributes=tuple(prof["e4_attributes"]))  # fmt: skip
        for family in FAMILIES:
            out[(family, split)] = sorted(items[family], key=lambda it: it["id"])
    return out


# --- checks -------------------------------------------------------------------------------------------------------


def run_checks(people, lex, writers, items, prof, seed, registry) -> dict[str, Any]:
    checker = Checker(people, lex)
    for component, w in writers.items():
        checker.scan_bios([d.to_json() for d in w.docs], component)
    all_items = [it for v in items.values() for it in v]
    report = checker.check_items(all_items, corpora={"zh": "zh", "en": "en"})
    by_id = {p.id: p for p in people}
    # exposures from the text scan must equal the declared numbers exactly
    exp = prof["bio_exposures"]
    declared = {"seed": sum(exp["seed"].values()), "zh": exp["zh"]["zh"], "en": exp["en"]["en"],
                "parallel": 2 * prof["parallel_pairs_per_person"]}  # fmt: skip
    observed: dict[str, dict] = {}
    exposure_ok = True
    for corpus, want in declared.items():
        part = {"seed": "seed", "zh": "zh", "en": "en", "parallel": "par"}[corpus]
        group = "train" if corpus == "parallel" else None
        members = by_group(people, part, group)
        counts = [checker.exposures.get((p.id, a, corpus), 0) for p in members for a in ATTRIBUTES]
        observed[corpus] = {"declared": want, "min": min(counts), "max": max(counts), "facts": len(counts)}
        exposure_ok &= min(counts) == max(counts) == want
    # leak tests on the generated E3-par items, with the leaky control alongside
    n_values = {a: len(lex[a]) for a in ATTRIBUTES}
    leaks: dict[str, Any] = {}
    par = [it for (f, _), v in items.items() if f == "E3par" for it in v]
    for variant, attr, n_labels in (("same_city", "city", 2), ("month_sum", "month", 12)):
        chosen = [it for it in par if it["variant"] == variant]
        for side in ("a", "b"):
            leaks[f"{variant}/{side}.{attr}"] = leak.single_fact_leak(chosen, by_id, side, attr, n_values[attr],
                                                                       n_labels)  # fmt: skip
    rng = stream(seed, "leak/control")
    control = leak.born_earlier_items(by_group(people, "zh"), by_group(people, "en"), max(400, len(par)), rng)
    leaks["control_born_earlier/a.year"] = leak.single_fact_leak(control, by_id, "a", "year", n_values["year"], 2)
    leaks["attribute_independence"] = leak.attribute_independence(INDEPENDENCE_DRAWS, n_values, seed)
    # co-occurrence of test pairs and fresh codes, from the text of the training documents
    training_texts = [d.text for w in writers.values() for d in w.docs]
    test_pairs = [(by_id[it["structured"]["a"]].name, by_id[it["structured"]["b"]].name)
                  for it in par if it["split"] == "test"]  # fmt: skip
    cooc = leak.cooccurrence(test_pairs, (checker.names_in(t) for t in training_texts))
    item_codes = {it["id"]: [c for c, _ in it["structured"]["facts"]] for it in all_items if it["family"] == "E3ctx"}
    codes = leak.fresh_codes(item_codes, training_texts)
    train_ids = defaultdict(set)
    for w in writers.values():
        for family, ids in w.template_ids.items():
            train_ids[family] |= ids
    train_ids["E2"] = train_ids["E1"]  # E1 and E2 share the QA pools; QA documents train both formats
    sets = leak.template_sets(all_items, train_ids)
    disjoint = leak.template_disjointness(sets, leak.split_parts)
    disjoint_ok = all(not v for fam in disjoint.values() for k in ("overlap", "part_overlap") for v in fam[k].values())
    # label balance per family / variant / direction / split
    balance = {}
    balance_ok = True
    for (family, variant, direction, split), group_ in group_items(all_items).items():
        if family == "E4":
            continue
        labels = {"gold_index": label_shares(group_, lambda it: it["gold_index"])}
        if family == "E3ctx":
            labels["gold_value"] = label_shares(group_, lambda it: it["candidates"][it["gold_index"]])
        if family == "E3par":
            labels["label"] = label_shares(group_, lambda it: it["structured"]["label"])
        dev = max(abs(s - 1 / len(shares)) for shares in labels.values() for s in shares.values())
        tolerance = max(0.01, 1 / len(group_))  # one item's share: exact balance is impossible for tiny n
        balance[f"{family}/{variant}/{direction}/{split}"] = {"n": len(group_), "max_deviation": dev}
        balance_ok &= dev <= tolerance
    partitions = defaultdict(Counter)
    for p in people:
        partitions[p.partition][p.group] += 1
    leak_ok = all(v["p"] >= LEAK_ALPHA for k, v in leaks.items() if "/" in k and not k.startswith("control"))
    leak_ok &= all(v["p"] >= LEAK_ALPHA for v in leaks["attribute_independence"].values())
    control_flagged = leaks["control_born_earlier/a.year"]["p"] < 1e-6 and \
        leaks["control_born_earlier/a.year"]["bayes_accuracy"] >= 0.7  # fmt: skip
    status = {
        "checker": report.ok and not checker.doc_errors,
        "exposures_exact": exposure_ok,
        "templates_disjoint": disjoint_ok,
        "labels_balanced": balance_ok,
        "no_leak": leak_ok,
        "control_flagged": control_flagged,
        "no_test_pair_cooccurs": not cooc,
        "codes_fresh": not codes["shared_between_items"] and not codes["in_training"],
    }
    return {
        "status": status,
        "checker": {
            "items": report.items,
            "by_family": dict(report.by_family),
            "errors": report.errors[:50],
            "n_errors": len(report.errors),
            "doc_errors": checker.doc_errors[:50],
            "ablation_uniform": report.ablation_uniform,
        },  # fmt: skip
        "exposures": observed,
        "leak_tests": leaks,
        "cooccurring_test_pairs": cooc,
        "codes": codes,
        "templates": {
            "disjointness": disjoint,
            "used": {fam: {s: sorted(v) for s, v in by.items()} for fam, by in sets.items()},
            "registry_counts": {
                f: {s: registry.family_count(f, s, tuple(prof["e4_attributes"])) for s in ("train", "dev", "test")}
                for f in FAMILIES
            },  # fmt: skip
        },
        "label_balance": balance,
        "entity_partitions": {k: dict(v) for k, v in sorted(partitions.items())},
    }


# --- build --------------------------------------------------------------------------------------------------------


def build_track_s(spec: Path, out: Path, seed: int, profile: str = "full") -> dict:
    """Write Track S to `out` (which must not exist) and return its data card; raises BuildError when a check
    fails (the files stay in place for inspection, without a data card)."""
    out = Path(out)
    if out.exists():
        raise BuildError(f"{out} exists; Track S builds are never overwritten")
    spec_doc, prof, spec_bytes = load_spec(spec, profile)
    registry = load_registry(spec_doc["templates"])
    lex = lexicons_for(prof)
    r = Renderer(registry, lex)
    generator = {
        "version": TRACK_S_VERSION,
        "seed": int(seed),
        "profile": profile,
        "spec_sha256": sha256_bytes(spec_bytes),
    }
    sizes = {k: prof[k] for k in ("heldout_fraction", "eval_dev_fraction", "par_train_fraction")}
    people = draw_people(prof["partitions"], {a: len(lex[a]) for a in ATTRIBUTES}, sizes, stream(seed, "people"))
    codes = CodeAllocator(stream(seed, "codes"))
    writers, rewrites = training_docs(people, r, prof, seed, codes)
    items = eval_items(people, r, prof, seed, codes, generator)
    files: dict[str, bytes] = {"entities.jsonl": b"".join(canonical_json(p.to_json()) + b"\n" for p in people)}
    for c, w in writers.items():
        files[f"{c}/track_s.jsonl"] = b"".join(doc_line(d) for d in w.docs)
    for (family, split), group_ in items.items():
        files[f"items/{family}.{split}.jsonl"] = b"".join(canonical_json(it) + b"\n" for it in group_)
    for rel, data in files.items():
        atomic_write_bytes(out / rel, data)
    checks = run_checks(people, lex, writers, items, prof, seed, registry)
    counts: dict[str, Any] = {"items": defaultdict(dict), "docs": {c: dict(w.counts) for c, w in writers.items()}}
    for (family, variant, direction, split), group_ in sorted(
        group_items(i for v in items.values() for i in v).items()
    ):
        counts["items"][f"{family}/{variant}"].setdefault(split, {})[direction] = len(group_)
    card = {
        "format": CARD_FORMAT,
        "kind": "track_s",
        "generator": {**generator, "spec_version": spec_doc["track_s"]["spec_version"]},
        "profile": prof,
        "files": {rel: sha256_file(out / rel) for rel in sorted(files)},
        "counts": {"items": dict(counts["items"]), "docs": counts["docs"]},
        "rewrites": {
            "declared_min": prof["min_rewrites"],
            "distinct_per_person": {k: {"min": a, "max": b} for k, (a, b) in sorted(rewrites.items())},
        },  # fmt: skip
        "licence": {
            "licence": "LicenseRef-generated",
            "licence_verified": True,
            "note": "generated by this package; names and employers are invented, cities are real place names",
        },  # fmt: skip
        "seed_mix": None,  # recorded by the D_seed builder's card (data.mixture)
        **checks,
    }
    if not all(checks["status"].values()):
        failed = [k for k, v in checks["status"].items() if not v]
        raise BuildError(f"Track S checks failed: {failed}; errors: {checks['checker']['errors'][:5]} "
                         f"{checks['checker']['doc_errors'][:5]}")  # fmt: skip
    atomic_write_json(out / "datacard.json", card)
    return card


def plan(spec: Path, profile: str) -> dict:
    """What a build would produce, from the spec alone (the --dry-run of `synth`)."""
    spec_doc, prof, _ = load_spec(spec, profile)
    registry = load_registry(spec_doc["templates"])
    lexicons_for(prof)
    n_items = {f"{f}/{v}": {s: prof["items"][s] * len(d) for s in SPLITS} for (f, v), d in DIRECTIONS.items()}
    return {
        "templates": len(registry.templates),
        "test_templates": {f: registry.family_count(f, "test", tuple(prof["e4_attributes"])) for f in FAMILIES},
        "items": n_items,
        "people": dict(prof["partitions"]),
    }


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m surogate_bilingual synth", description="build Track S")
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--dry-run", action="store_true", help="validate the spec and print the plan; generate nothing")
    args = ap.parse_args(list(argv))
    try:
        if args.dry_run:
            info = plan(args.spec, args.profile)
            print(f"synth: {info['templates']} templates; test templates per family {info['test_templates']}; "
                  f"people {info['people']} (dry run)")  # fmt: skip
            return 0
        if args.out is None:
            ap.error("--out is required unless --dry-run")
        card = build_track_s(args.spec, args.out, args.seed, args.profile)
    except (BuildError, ValueError) as err:
        print(f"synth: {err}", file=sys.stderr)
        return 1
    print(f"synth: wrote {args.out} ({len(card['files'])} files); checks {card['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
