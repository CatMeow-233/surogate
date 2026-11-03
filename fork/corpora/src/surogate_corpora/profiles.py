"""Export profiles: per use, the sources, the ordered filters, the annotators and the writer. Plus the pipeline that
applies one profile to the rows of one source.

Filters run in order: the metadata filters (the cheap columns that `plan` also evaluates to estimate kept bytes),
then the content filters. Every profile starts its content filters with `utf8_valid`. Decontamination is not done
here: each project runs its own, before it trains a tokenizer.

| profile | sources | filters, in order |
|---|---|---|
| tlc-python-permissive | stack3-py | licence_permissive, not_vendor; utf8_valid, exact_dedup, non_latin_letter_ratio_max(x) |
| tlc-python-all | stack3-py | the owner option: licence_permissive dropped |
| tlc-python-unlicensed-ablation | code-l2-py | utf8_valid, exact_dedup, non_latin_letter_ratio_max(x); owner flag --allow-unlicensed |
| tlc-english | ufw-en | optional score_min(q); utf8_valid, exact_dedup |
| bilingual-zh-web | ufw-zh | source_in(CCI3, ChineseWebText, IndustryCorpus); utf8_valid; script_stats annotator |
| bilingual-zh-apache | ufw-zh | as zh-web plus Tele |
| bilingual-zh-all | ufw-zh | all eight upstream sources |
| bilingual-en | ufw-en | utf8_valid, exact_dedup |
| bilingual-parallel-unpc | unpc-en-zh | utf8_valid and non_empty on both sides (language ID, length ratio and alignment cleaning stay in the bilingual project) |

Parameters (`--param name=value`): `non_latin_max` (default 0.10, fixed in TLC's Phase 0) for the Python profiles,
`score_min` (default none) for tlc-english. The resolved values enter the plan and every export's filters.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from . import filters as F
from .canonical import CorporaError
from .sources import SOURCES, SourceDef, covers
from .sources import get as get_source

WEB_POOL = frozenset({"CCI3", "ChineseWebText", "IndustryCorpus"})
APACHE_POOL = WEB_POOL | {"Tele"}
ALL_ZH = frozenset(SOURCES["ufw-zh"].licence.spdx)
WRITERS = ("tlc", "bilingual", "parallel")
ALLOW_UNLICENSED = "allow_unlicensed"


@dataclass(frozen=True)
class Profile:
    name: str
    sources: tuple[str, ...]
    metadata_filters: tuple[F.Filter, ...]
    content_filters: tuple[F.Filter, ...]
    annotators: tuple[F.ScriptStats, ...]
    writer: str  # tlc | bilingual | parallel
    kind: str  # the fragment's `kind`: code | text (TLC), zh | en | parallel (bilingual)
    budget_unit: str = "kept_bytes"
    params: Mapping[str, Any] = field(default_factory=dict)
    requires: str | None = None  # an owner flag the profile needs
    licence_note: str = ""  # appended to a TLC source's `license`

    def filter_names(self) -> list[str]:
        return [f.describe() for f in (*self.metadata_filters, *self.content_filters)]

    def describe(self) -> dict[str, Any]:
        """What the plan records and fetch re-checks: a code change to the profile refuses an older plan."""
        return {
            "name": self.name,
            "sources": list(self.sources),
            "params": dict(sorted(self.params.items())),
            "metadata_filters": [f.describe() for f in self.metadata_filters],
            "content_filters": [f.describe() for f in self.content_filters],
            "annotators": [a.describe() for a in self.annotators],
            "writer": self.writer,
            "kind": self.kind,
            "budget_unit": self.budget_unit,
        }

    def metadata_columns(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c for f in self.metadata_filters for c in f.columns()))


@dataclass(frozen=True)
class ProfileSpec:
    defaults: Mapping[str, Any]
    build: Callable[[Mapping[str, Any]], Profile]


def _python(name: str, source: str, metadata: tuple, note: str, requires: str | None = None) -> ProfileSpec:
    def build(params):
        content = (F.Utf8Valid(), F.ExactDedup(), F.NonLatinLetterRatioMax(float(params["non_latin_max"])))
        return Profile(
            name, (source,), metadata, content, (), "tlc", "code", params=params, requires=requires, licence_note=note
        )

    return ProfileSpec({"non_latin_max": 0.10}, build)


def _english(params) -> Profile:
    q = params["score_min"]
    metadata = () if q is None else (F.ScoreMin(float(q)),)
    content = (F.Utf8Valid(), F.ExactDedup())
    return Profile("tlc-english", ("ufw-en",), metadata, content, (), "tlc", "text", params=params)


def _zh(name: str, pool: frozenset[str]) -> ProfileSpec:
    def build(params):
        return Profile(
            name,
            ("ufw-zh",),
            (F.SourceIn(pool),),
            (F.Utf8Valid(),),
            (F.ScriptStats(),),
            "bilingual",
            "zh",
            params=params,
        )

    return ProfileSpec({}, build)


UNPC_TEXT = ("translation.en", "translation.zh")

PROFILES: dict[str, ProfileSpec] = {
    "tlc-python-permissive": _python(
        "tlc-python-permissive", "stack3-py", (F.LicencePermissive(), F.NotVendor()), " (permissive only)"
    ),
    "tlc-python-all": _python(
        "tlc-python-all", "stack3-py", (F.NotVendor(),), " (permissive and no_license: owner option)"
    ),
    "tlc-python-unlicensed-ablation": _python(
        "tlc-python-unlicensed-ablation", "code-l2-py", (), " (data-source ablation only)", ALLOW_UNLICENSED
    ),
    "tlc-english": ProfileSpec({"score_min": None}, _english),
    "bilingual-zh-web": _zh("bilingual-zh-web", WEB_POOL),
    "bilingual-zh-apache": _zh("bilingual-zh-apache", APACHE_POOL),
    "bilingual-zh-all": _zh("bilingual-zh-all", ALL_ZH),
    "bilingual-en": ProfileSpec(
        {}, lambda p: Profile("bilingual-en", ("ufw-en",), (), (F.Utf8Valid(), F.ExactDedup()), (), "bilingual", "en")
    ),
    "bilingual-parallel-unpc": ProfileSpec(
        {},
        lambda p: Profile(
            "bilingual-parallel-unpc",
            ("unpc-en-zh",),
            (),
            (F.Utf8Valid(UNPC_TEXT), F.NonEmpty(UNPC_TEXT)),
            (),
            "parallel",
            "parallel",
        ),
    ),
}


def parse_params(pairs: list[str] | None) -> dict[str, Any]:
    """`["non_latin_max=0.05", "score_min=none"]` -> {"non_latin_max": 0.05, "score_min": None}."""
    out: dict[str, Any] = {}
    for pair in pairs or []:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise CorporaError(f"--param needs name=value, got {pair!r}")
        if value.lower() in ("none", "null", ""):
            out[name] = None
        else:
            try:
                out[name] = float(value)
            except ValueError:
                raise CorporaError(f"--param {name}: {value!r} is not a number or none") from None
    return out


def resolve(name: str, params: Mapping[str, Any] | None = None, *, allow_unlicensed: bool = False) -> Profile:
    """The profile `name` with `params` over its defaults; refuses an unknown name or parameter, a profile that needs
    an owner flag it was not given, and any source the registry does not allow it."""
    if name not in PROFILES:
        raise CorporaError(f"unknown profile {name!r} (known: {sorted(PROFILES)})")
    spec = PROFILES[name]
    unknown = sorted(set(params or {}) - set(spec.defaults))
    if unknown:
        raise CorporaError(f"profile {name} has no parameters {unknown} (its parameters: {sorted(spec.defaults)})")
    profile = spec.build({**spec.defaults, **(params or {})})
    for source_id in profile.sources:
        source = get_source(source_id)
        if not source.profiles:
            raise CorporaError(f"source {source_id} is model-generated: every pretraining profile refuses it")
        if name not in source.profiles:
            raise CorporaError(f"the registry does not allow profile {name} for source {source_id}")
    if profile.requires == ALLOW_UNLICENSED and not allow_unlicensed:
        raise CorporaError(
            f"profile {name} reads {', '.join(profile.sources)}, which has no per-file licence metadata; it is an "
            "owner-flagged data-source ablation only (pass --allow-unlicensed)"
        )
    return profile


def check_source(profile: Profile, source_id: str) -> SourceDef:
    """The source, if `profile` may read it; else a refusal that says why."""
    source = get_source(source_id)
    if not source.profiles:
        raise CorporaError(f"source {source_id} is model-generated: every pretraining profile refuses it")
    if source_id not in profile.sources:
        missing = [c for c in profile.metadata_columns() if not covers(source.schema, c)]
        if missing:
            raise CorporaError(
                f"profile {profile.name} filters on {', '.join(missing)}, which source {source_id} does not have"
            )
        raise CorporaError(
            f"source {source_id} is not a source of profile {profile.name} ({', '.join(profile.sources)})"
        )
    return source


class Pipeline:
    """One profile applied to the rows of one source, with fresh filter state."""

    def __init__(self, profile: Profile, source: SourceDef):
        self.profile = profile
        self.source = source
        self.filters = [dataclasses.replace(f) for f in (*profile.metadata_filters, *profile.content_filters)]
        for f in self.filters:
            if isinstance(f, F.Utf8Valid | F.NonEmpty) and f.text_columns != source.content:
                raise CorporaError(f"profile {profile.name}: {f.describe()} does not check {source.id}'s text columns")
        self.dedup = next((f for f in self.filters if isinstance(f, F.ExactDedup)), None)

    def __call__(self, row: dict[str, Any]) -> F.Result:
        for f in self.filters:
            keep, reason = f(row)
            if not keep:
                return False, reason
        return F.KEPT

    def annotate(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return {a.key: a(row[F.TEXT]) for a in self.profile.annotators}
