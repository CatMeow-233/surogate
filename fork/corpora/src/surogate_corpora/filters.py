"""Row filters and annotators. Pure except ExactDedup, which keeps the hashes of the documents kept so far.

A row is a flat `{leaf path: value}` dictionary (parquet.iter_rows); text columns hold bytes until `utf8_valid` has
decoded them. A filter is a frozen dataclass called on a row; it returns `(keep, reason)`, the reason (a drop count
key) None when kept. `describe()` gives the `"<name>(<params>)"` string that exports record in each spec fragment's
`filters`. An annotator maps the decoded text to a `meta` entry. Nothing here changes a document's bytes.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from typing import Any

Result = tuple[bool, str | None]
KEPT: Result = (True, None)
TEXT = "__text__"  # where utf8_valid puts the decoded text of a single-text-column row
TEXTS = "__texts__"  # {column: decoded text} of every text column


def _params(obj) -> str:
    parts = []
    for f in fields(obj):
        if f.name.startswith("_"):
            continue
        value = getattr(obj, f.name)
        if isinstance(value, frozenset | tuple | list):
            value = "|".join(sorted(map(str, value)))
        parts.append(f"{f.name}={value}")
    return ",".join(parts)


class Filter:
    name = "filter"

    def __call__(self, row: Mapping[str, Any]) -> Result:  # pragma: no cover - interface
        raise NotImplementedError

    def describe(self) -> str:
        return f"{self.name}({_params(self)})"

    def columns(self) -> tuple[str, ...]:
        """The columns the filter reads (what a metadata pass must fetch)."""
        column = getattr(self, "field", None)
        return (column,) if column else ()


@dataclass(frozen=True)
class LicencePermissive(Filter):
    """Keeps a row whose licence type is exactly "permissive" (The Stack v3: `permissive` or `no_license`)."""

    field: str = "source_meta.file_meta.license_type"
    name = "licence_permissive"

    def __call__(self, row):
        return KEPT if row.get(self.field) == "permissive" else (False, self.name)


@dataclass(frozen=True)
class NotVendor(Filter):
    """Keeps a row whose vendor flag is False; a missing flag (null) is dropped too."""

    field: str = "source_meta.file_meta.is_vendor"
    name = "not_vendor"

    def __call__(self, row):
        return KEPT if row.get(self.field) is False else (False, self.name)


@dataclass(frozen=True)
class ScoreMin(Filter):
    """Keeps a row whose quality score is at least `q`. Ultra-FineWeb stores the score as a decimal string."""

    q: float
    field: str = "score"
    name = "score_min"

    def __call__(self, row):
        value = row.get(self.field)
        try:
            score = float(value)
        except (TypeError, ValueError):
            return False, self.name
        return KEPT if score >= self.q else (False, self.name)


@dataclass(frozen=True)
class SourceIn(Filter):
    """Keeps a row whose upstream `source` value is one of `names`."""

    names: frozenset[str]
    field: str = "source"
    name = "source_in"

    def __call__(self, row):
        return KEPT if row.get(self.field) in self.names else (False, self.name)


@dataclass(frozen=True)
class Utf8Valid(Filter):
    """Keeps a row whose text columns all decode as strict UTF-8, and stores the decoded text in the row
    (`__texts__`, and `__text__` when there is one text column). Applied by every profile: a document that is not
    valid UTF-8 cannot be written as JSON text."""

    text_columns: tuple[str, ...] = ("content",)
    name = "utf8_valid"

    def columns(self):
        return self.text_columns

    def __call__(self, row):
        texts = {}
        for column in self.text_columns:
            value = row.get(column)
            if isinstance(value, bytes):
                try:
                    value = value.decode("utf-8")
                except UnicodeDecodeError:
                    return False, self.name
            elif isinstance(value, str):
                try:
                    value.encode("utf-8")  # a lone surrogate (a JSON "\ud800") is not valid UTF-8
                except UnicodeEncodeError:
                    return False, self.name
            else:
                return False, self.name
            texts[column] = value
        row[TEXTS] = texts  # rows are the pipeline's own dictionaries
        if len(texts) == 1:
            row[TEXT] = next(iter(texts.values()))
        return KEPT


@dataclass(frozen=True)
class NonEmpty(Filter):
    """Keeps a row whose decoded text columns are all non-empty (after utf8_valid)."""

    text_columns: tuple[str, ...] = ("content",)
    name = "non_empty"

    def columns(self):
        return self.text_columns

    def __call__(self, row):
        texts = row.get(TEXTS, {})
        return KEPT if all(texts.get(c) for c in self.text_columns) else (False, self.name)


def letter_counts(text: str) -> tuple[int, int]:
    """(letters, non-Latin letters): a letter is str.isalpha(); it is Latin when its Unicode name starts with LATIN."""
    letters = non_latin = 0
    for ch in text:
        if ch.isalpha():
            letters += 1
            if not unicodedata.name(ch, "").startswith("LATIN"):
                non_latin += 1
    return letters, non_latin


@dataclass(frozen=True)
class NonLatinLetterRatioMax(Filter):
    """Keeps a document whose share of non-Latin letters among its letters is at most `x` (a proxy for non-English
    comments and docstrings in code); a document without letters is kept. The threshold is the owner's, recorded."""

    x: float
    name = "non_latin_letter_ratio_max"

    def columns(self):
        return ()

    def __call__(self, row):
        letters, non_latin = letter_counts(row[TEXT])
        return KEPT if letters == 0 or non_latin <= self.x * letters else (False, self.name)


@dataclass(frozen=True)
class ExactDedup(Filter):
    """Drops a document whose UTF-8 bytes equal an earlier kept document's (sha256), in processing order. The state
    is the set of kept hashes; `seed` rebuilds it from documents already exported (resume)."""

    _seen: set = field(default_factory=set, init=False, compare=False, repr=False)
    name = "exact_dedup"

    def columns(self):
        return ()

    def __call__(self, row):
        texts = row[TEXTS]
        digest = hashlib.sha256(b"\0".join(texts[c].encode("utf-8") for c in sorted(texts))).digest()
        if digest in self._seen:
            return False, self.name
        self._seen.add(digest)
        return KEPT

    def seed(self, texts: Iterable[Mapping[str, str]]) -> None:
        for t in texts:
            self._seen.add(hashlib.sha256(b"\0".join(t[c].encode("utf-8") for c in sorted(t))).digest())

    def describe(self) -> str:
        return f"{self.name}(sha256 of the text bytes)"


# --- annotators -----------------------------------------------------------------------------------------------------

SCRIPT_TABLE_VERSION = 1
# 48 (Simplified, Traditional) pairs of frequent characters whose two forms differ and whose Simplified form is not
# itself in common Traditional use (version 1; changing it is a new version).
SCRIPT_PAIRS = (
    "们們 这這 个個 来來 说說 时時 国國 对對 会會 经經 过過 发發 问問 还還 进進 实實 现現 开開 关關 长長 门門 学學 业業 东東 "
    "车車 书書 马馬 见見 产產 电電 话話 体體 头頭 点點 无無 应應 样樣 让讓 给給 从從 动動 两兩 爱愛 认認 么麼 与與 为為 万萬"
).split()
HANS = frozenset(pair[0] for pair in SCRIPT_PAIRS)
HANT = frozenset(pair[1] for pair in SCRIPT_PAIRS)


def script_stats(text: str) -> dict[str, int]:
    """Counts of Simplified-only and Traditional-only marker characters (SCRIPT_PAIRS, version 1)."""
    return {"hans": sum(ch in HANS for ch in text), "hant": sum(ch in HANT for ch in text)}


@dataclass(frozen=True)
class ScriptStats:
    """Annotator: meta["script"] = script_stats(text)."""

    name = "script_stats"
    key = "script"

    def __call__(self, text: str) -> dict[str, int]:
        return script_stats(text)

    def describe(self) -> str:
        return f"{self.name}(table=v{SCRIPT_TABLE_VERSION},pairs={len(SCRIPT_PAIRS)})"
