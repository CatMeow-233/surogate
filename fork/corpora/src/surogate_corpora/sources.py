"""The pinned source registry: which Hub dataset files each source reads, at which revision, in which schema, under
which licences, and which export profiles may use it. Code, not a file, so that a change is a reviewed commit.

Rules:
- a revision is a full 40-hex commit sha; `main`, a branch name or a short sha is refused;
- `files` is a path pattern with one `{first..last}` range (zero padded to the width of `first`); the Hub listing at
  plan time must hold exactly the files it expands to;
- `schema` is the parquet schema as the files hold it, one entry per leaf column (dotted path -> arrow type); test
  fixtures are written in it, and `verify-samples` checks real datasets-server rows against its server view
  (`server_types`: the datasets-server dtype where it differs from the parquet type, e.g. Ultra-FineWeb's `score`,
  a string in the parquet files and float32 in datasets-server);
- `columns` are the columns the content pass reads; `profiles` are the export profiles allowed to read the source.
  The synthetic UltraData-Code-L3 has no profile, so every pretraining profile refuses it.

Every licence id here is also accepted by the bilingual manifest (its LICENCE_ID pattern; a test checks), and none
is verified: `licence_verified` stays false in every export until the owner reads the terms.

Facts below were read on 2026-11-03 from the Hub API (`/api/datasets/<id>/tree/<sha>`), the dataset cards,
datasets-server `/info` and the parquet footers of the files [verified]; licence terms marked "not read" were not.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .canonical import CorporaError

REVISION = re.compile(r"[0-9a-f]{40}")
RANGE = re.compile(r"\{(\d+)\.\.(\d+)\}")
ARROW_TYPES = ("string", "float", "double", "int64", "bool", "list<string>")
SERVER_TYPE = {"string": "string", "float": "float32", "double": "float64", "int64": "int64", "bool": "bool"}


@dataclass(frozen=True)
class LicenceBlock:
    release: str  # the licence of the Hub release itself
    upstream: Mapping[str, str]  # upstream source -> what its licence says, and how far that was read
    status: str
    tlc_license: str  # the TLC corpus spec's free-text `license` (a profile may append its own policy)
    spdx: Mapping[str, str] = field(default_factory=dict)  # upstream source ("*": every row) -> bilingual licence id
    verified: bool = False  # never true here: only the owner flips licence_verified


@dataclass(frozen=True)
class SourceDef:
    id: str
    repo: str  # Hub dataset repository
    revision: str  # full commit sha
    files: str  # path pattern with one {first..last} range
    schema: Mapping[str, str]  # leaf column -> arrow type, as the parquet files hold it
    content: tuple[str, ...]  # the text columns: exported bytes come from these, unchanged
    columns: tuple[str, ...]  # what the content pass reads (leaf columns or struct prefixes)
    licence: LicenceBlock
    profiles: frozenset[str]
    name: str  # output name: parts/<name>-NNNNN.jsonl and the fragment's source name
    server_types: Mapping[str, str] = field(default_factory=dict)  # datasets-server dtype where it differs
    size_column: str | None = None  # per-row byte size, for the plan's kept-bytes estimate
    upstream_column: str | None = None  # one part series and fragment entry per value of this column

    def __post_init__(self):
        if not REVISION.fullmatch(self.revision):
            raise CorporaError(f"source {self.id}: revision must be a full 40-hex commit sha, not {self.revision!r}")
        if len(RANGE.findall(self.files)) != 1:
            raise CorporaError(f"source {self.id}: files needs exactly one {{first..last}} range: {self.files!r}")
        bad = {k: v for k, v in self.schema.items() if v not in ARROW_TYPES}
        if bad:
            raise CorporaError(f"source {self.id}: unknown arrow types {bad}")
        for column in (*self.content, *self.columns, *filter(None, (self.size_column, self.upstream_column))):
            if not covers(self.schema, column):
                raise CorporaError(f"source {self.id}: column {column!r} is not in its schema")
        if not set(self.content) <= set(self.columns) and not all(covered(self.columns, c) for c in self.content):
            raise CorporaError(f"source {self.id}: the content pass must read the text columns {self.content}")

    def expand_files(self) -> list[str]:
        return expand(self.files)

    def server_schema(self) -> dict[str, str]:
        """The leaf types datasets-server reports for this source."""
        out = {}
        for column, kind in self.schema.items():
            if column in self.server_types:
                out[column] = self.server_types[column]
            elif kind == "list<string>":
                out[column] = "list<string>"
            else:
                out[column] = SERVER_TYPE[kind]
        return out

    def describe(self) -> dict:
        return {
            "id": self.id,
            "repo": self.repo,
            "revision": self.revision,
            "files": self.files,
            "n_files": len(self.expand_files()),
            "content": list(self.content),
            "columns": list(self.columns),
            "profiles": sorted(self.profiles),
            "release_licence": self.licence.release,
            "licence_status": self.licence.status,
        }


def covers(schema: Mapping[str, str], column: str) -> bool:
    """True when `column` is a leaf of the schema or the prefix of some leaves (a struct)."""
    return column in schema or any(leaf.startswith(column + ".") for leaf in schema)


def covered(columns: Sequence[str], leaf: str) -> bool:
    return any(leaf == c or leaf.startswith(c + ".") for c in columns)


def expand(pattern: str) -> list[str]:
    """`a-{001..003}.parquet` -> [`a-001.parquet`, `a-002.parquet`, `a-003.parquet`]."""
    match = RANGE.search(pattern)
    if not match:
        raise CorporaError(f"no {{first..last}} range in {pattern!r}")
    first, last = match.group(1), match.group(2)
    if int(first) > int(last):
        raise CorporaError(f"empty range in {pattern!r}")
    width = len(first)
    return [
        f"{pattern[: match.start()]}{i:0{width}d}{pattern[match.end() :]}" for i in range(int(first), int(last) + 1)
    ]


# --- schemas (parquet footers, 2026-11-03) ---------------------------------------------------------------------------

_GITHUB = "source_meta.github_metadata."
_FILE = "source_meta.file_meta."
STACK3_SCHEMA = {
    "uuid": "string", "repo_name": "string", "relative_path": "string", "content": "string", "category": "string",
    "algo_rel_score": "float", "quality_score": "float",
    _GITHUB + "branch": "string", _GITHUB + "commit_count": "int64", _GITHUB + "repo_created_at": "string",
    _GITHUB + "is_fork": "bool", _GITHUB + "is_org_owned": "bool", _GITHUB + "forked_from": "string",
    _GITHUB + "stars": "int64", _GITHUB + "forks": "int64", _GITHUB + "issues": "int64",
    _GITHUB + "pull_requests": "int64",
    _FILE + "content_id": "string", _FILE + "size_bytes": "int64", _FILE + "file_timestamp": "int64",
    _FILE + "language": "string", _FILE + "is_vendor": "bool", _FILE + "license_type": "string",
    _FILE + "detected_licenses": "list<string>",
    "source_meta.commit_id": "string",
}  # fmt: skip
UFW_SCHEMA = {"content": "string", "score": "string", "source": "string"}
CODE_L2_SCHEMA = {
    "uuid": "string", "repo_name": "string", "relative_path": "string", "content": "string", "category": "string",
    "algo_rel_score": "double", "quality_score": "double",
}  # fmt: skip
CODE_L3_SCHEMA = {
    name: "string"
    for name in ("uuid", "content", "content_format", "raw_content", "task", "analysis", "solution", "test",
                 "full_content", "full_content_format")
}  # fmt: skip
UNPC_SCHEMA = {"translation.en": "string", "translation.zh": "string"}

# The upstream sources of Ultra-FineWeb zh (its `source` column) and what their licences say.
UFW_ZH_UPSTREAM = {
    "Tele": "Tele-AI/TeleChat-PTD: HF tag apache-2.0; its README also cites the TeleChat model community licence "
    "(which one governs is undecided; terms not read)",
    "CCI3": "BAAI/CCI3-Data: HF tag apache-2.0 (gated, automatic approval)",
    "ChineseWebText": "CASIA-LM/ChineseWebText2.0: HF tag apache-2.0",
    "IndustryCorpus": "BAAI/IndustryCorpus2: HF tag apache-2.0 (gated, automatic approval)",
    "SkyPile": "Skywork/SkyPile-150B: Skywork Community License (no HF tag; terms not read clause by clause)",
    "WanJuan": "WanJuan 1.0 Chinese text: second-hand reports say CC BY 4.0 with some subsets under other "
    "agreements (not read)",
    "WuDao": "WuDaoCorpora: BAAI Data Agreement; second-hand reports say CC-BY-NC-ND-4.0 (not read)",
    "MiChao": "MiChao-HuaFen 1.0: 'comply with its usage agreement', agreement not named (not read)",
}
UFW_ZH_SPDX = {
    "Tele": "Apache-2.0",
    "CCI3": "Apache-2.0",
    "ChineseWebText": "Apache-2.0",
    "IndustryCorpus": "Apache-2.0",
    "SkyPile": "LicenseRef-Skywork-Community-License",
    "WanJuan": "LicenseRef-WanJuan-1.0",
    "WuDao": "LicenseRef-BAAI-Data-Agreement",
    "MiChao": "LicenseRef-MiChao-Usage-Agreement",
}
UFW_RELEASE = (
    "Apache-2.0 (the card asks users to check the licence of each source dataset individually; Ultra-FineWeb-L1 "
    "and -L3 add a no-unauthorised-verbatim-redistribution clause, the base set's card does not)"
)
UFW_REVISION = "02c85641e3d19a854be2e09139c25adaa9518063"
CODE_REVISION = "8ce985cfd4522832c8b986e05968c87fffa0c66a"
STACK3_FILE_COLUMNS = tuple(
    _FILE + c for c in ("license_type", "detected_licenses", "is_vendor", "size_bytes", "content_id")
)

SOURCES: dict[str, SourceDef] = {
    s.id: s
    for s in (
        SourceDef(
            id="stack3-py",
            repo="openbmb/UltraData-Stack-v3-L2",
            revision="113743adaa92066ef6db2a5b7035f59b4ad6eb30",
            files="data/UltraData-Stack-v3-L2/python/UltraData-Stack-v3-L2-python-part-{00001..00045}-of-00045.parquet",
            schema=STACK3_SCHEMA,
            content=("content",),
            columns=("uuid", "repo_name", "relative_path", "content", *STACK3_FILE_COLUMNS, "source_meta.commit_id"),
            licence=LicenceBlock(
                release="ODC-By-1.0",
                upstream={
                    "The Stack v3": "per-file SPDX ids detected by ScanCode (`detected_licenses`); license_type is "
                    "permissive or no_license (non-permissive files were excluded upstream); opt-outs removed before "
                    "each release; crawl cutoff 2025-08-07",
                },
                status="release tag verified (HF API); per-file SPDX ids as published, not re-checked; Apache-2.0 "
                "dominates the permissive files in a way that suggests propagation from repository-level files "
                "[assumption], so spot-check before relying on the label",
                tlc_license="ODC-By-1.0 (UltraData-Stack-v3-L2) + per-file SPDX",
            ),
            profiles=frozenset({"tlc-python-permissive", "tlc-python-all"}),
            name="stack3-py",
            server_types={"algo_rel_score": "float32", "quality_score": "float32"},
            size_column=_FILE + "size_bytes",
        ),
        SourceDef(
            id="ufw-en",
            repo="openbmb/Ultra-FineWeb",
            revision=UFW_REVISION,
            files="data/ultrafineweb_en/ultrafineweb-en-part-{0001..2048}-of-2048.parquet",
            schema=UFW_SCHEMA,
            content=("content",),
            columns=("content", "score", "source"),
            licence=LicenceBlock(
                release=UFW_RELEASE,
                upstream={"FineWeb": "HuggingFaceFW/fineweb: ODC-By-1.0, plus the Common Crawl terms of use"},
                status="release tag and FineWeb licence verified (HF API, cards)",
                tlc_license="Apache-2.0 (Ultra-FineWeb) + ODC-By-1.0 (FineWeb) + CommonCrawl ToU",
                spdx={"*": "ODC-By-1.0"},
            ),
            profiles=frozenset({"tlc-english", "bilingual-en"}),
            name="ufw-en",
            server_types={"score": "float32"},
        ),
        SourceDef(
            id="ufw-zh",
            repo="openbmb/Ultra-FineWeb",
            revision=UFW_REVISION,
            files="data/ultrafineweb_zh/ultrafineweb-zh-part-{001..256}-of-256.parquet",
            schema=UFW_SCHEMA,
            content=("content",),
            columns=("content", "score", "source"),
            licence=LicenceBlock(
                release=UFW_RELEASE,
                upstream=UFW_ZH_UPSTREAM,
                status="release tag verified; upstream tags as listed, terms of SkyPile, WanJuan, WuDao, MiChao and "
                "the TeleChat community licence not read",
                tlc_license="Apache-2.0 (Ultra-FineWeb) + per-source upstream licences",
                spdx=UFW_ZH_SPDX,
            ),
            profiles=frozenset({"bilingual-zh-web", "bilingual-zh-apache", "bilingual-zh-all"}),
            name="ufw-zh",
            server_types={"score": "float32"},
            upstream_column="source",
        ),
        SourceDef(
            id="code-l2-py",
            repo="openbmb/UltraData-Code",
            revision=CODE_REVISION,
            files="data/UltraData-Code-L2/py/UltraData-Code-L2-py-part-{00001..00119}-of-00119.parquet",
            schema=CODE_L2_SCHEMA,
            content=("content",),
            columns=("uuid", "repo_name", "relative_path", "content", "algo_rel_score"),
            licence=LicenceBlock(
                release="Apache-2.0 (the card: users must also comply with the licence of each source repository; "
                "public availability is not a grant of training rights; redistribution clause)",
                upstream={
                    "GitHub": "per repository; the files carry no licence field, and L1 cleaning rewrote "
                    "licence text inside files, so file headers cannot restore it"
                },
                status="no per-file licence metadata: an owner-flagged data-source ablation only",
                tlc_license="Apache-2.0 (UltraData-Code-L2) + unknown per-repository licences (no licence metadata)",
            ),
            profiles=frozenset({"tlc-python-unlicensed-ablation"}),
            name="code-l2-py",
        ),
        SourceDef(
            id="code-l3-py",
            repo="openbmb/UltraData-Code",
            revision=CODE_REVISION,
            files="data/UltraData-Code-L3/py/UltraData-Code-L3-py-part-{00001..00147}-of-00147.parquet",
            schema=CODE_L3_SCHEMA,
            content=("content",),
            columns=("uuid", "content"),
            licence=LicenceBlock(
                release="Apache-2.0",
                upstream={"DeepSeek-v4-flash": "synthetic task/analysis/solution/test generated from L2 ALGO seeds"},
                status="model-generated: never a pretraining source",
                tlc_license="Apache-2.0 (UltraData-Code-L3, synthetic)",
            ),
            profiles=frozenset(),
            name="code-l3-py",
        ),
        SourceDef(
            id="unpc-en-zh",
            repo="Helsinki-NLP/un_pc",
            # the parquet conversion branch refs/convert/parquet (main is 99a0348ce711ec6b30d5f18d70c0d30c971011c7)
            revision="f0eca5eea7dad233a4b59de0479f9374d7a3500f",
            files="en-zh/train/{0000..0009}.parquet",
            schema=UNPC_SCHEMA,
            content=("translation.en", "translation.zh"),
            columns=("translation",),
            licence=LicenceBlock(
                release="other (HF tag): the UN disclaimer; acknowledge the UN as the source and cite Ziemski et al. "
                "2016, 'no other restrictions apply'",
                upstream={"United Nations": "UN Parallel Corpus v1.0 disclaimer"},
                status="outside the UltraData family: an owner exception",
                tlc_license="UN Parallel Corpus v1.0 disclaimer (attribution)",
                spdx={"*": "LicenseRef-UN-Parallel-Corpus-Disclaimer"},
            ),
            profiles=frozenset({"bilingual-parallel-unpc"}),
            name="unpc-en-zh",
        ),
    )
}


def get(source_id: str) -> SourceDef:
    try:
        return SOURCES[source_id]
    except KeyError:
        raise CorporaError(f"unknown source {source_id!r} (known: {sorted(SOURCES)})") from None


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m surogate_corpora sources", description="the source registry")
    parser.add_argument("--json", action="store_true", help="one JSON object per source")
    args = parser.parse_args(argv)
    for source in SOURCES.values():
        if args.json:
            print(json.dumps(source.describe(), ensure_ascii=False, sort_keys=True))
        else:
            profiles = ", ".join(sorted(source.profiles)) or "(none: every profile refuses it)"
            print(f"{source.id:<11} {source.repo}@{source.revision[:8]}  {len(source.expand_files())} files")
            print(f"{'':<11} profiles: {profiles}")
    sys.stdout.flush()
    return 0
