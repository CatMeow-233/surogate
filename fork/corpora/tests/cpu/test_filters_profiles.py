"""Step S3: each filter is exact on hand-made rows, drop counts equal hand counts, every profile resolves."""

import io

import pytest
from corpora_fixtures import MARK
from surogate_corpora import filters as F
from surogate_corpora import profiles
from surogate_corpora.canonical import CorporaError
from surogate_corpora.parquet import iter_rows
from surogate_corpora.sources import get

LT, VENDOR = "source_meta.file_meta.license_type", "source_meta.file_meta.is_vendor"


def test_licence_types_and_vendor_flags():
    permissive, vendor = F.LicencePermissive(), F.NotVendor()
    assert permissive({LT: "permissive"}) == (True, None)
    assert permissive({LT: "no_license"}) == (False, "licence_permissive")
    assert permissive({LT: None}) == permissive({}) == (False, "licence_permissive")
    assert vendor({VENDOR: False}) == (True, None)
    assert vendor({VENDOR: True}) == vendor({VENDOR: None}) == (False, "not_vendor")


def test_invalid_utf8_in_a_binary_column_fixture_is_dropped_not_fatal():
    """A parquet string column may hold bytes that are not UTF-8; the reader hands text columns over as bytes, so the
    row reaches utf8_valid instead of failing the row group."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    values = [f"{MARK} ok".encode(), b"\xff\xfe" + MARK.encode(), f"{MARK} café".encode(), b"\xc3"]
    content = pa.array(values, type=pa.binary()).view(pa.string())  # no UTF-8 validation on the way in
    buffer = io.BytesIO()
    pq.write_table(pa.table({"content": content}), buffer)
    table = pq.read_table(io.BytesIO(buffer.getvalue()))
    check = F.Utf8Valid()
    results = [check(row) for row in iter_rows(table, ("content",))]
    assert results == [(True, None), (False, "utf8_valid"), (True, None), (False, "utf8_valid")]
    row = {"content": f"{MARK} café".encode()}
    check(row)
    assert row[F.TEXT] == f"{MARK} café" and row[F.TEXTS] == {"content": f"{MARK} café"}
    assert check({"content": "lone \ud800 surrogate"}) == (False, "utf8_valid")
    assert check({"content": None}) == (False, "utf8_valid")


def test_exact_dedup_drops_repeats_in_order_and_can_be_seeded():
    dedup = F.ExactDedup()
    texts = ["a", "b", "a", "c", "b", "a "]
    rows = [{F.TEXTS: {"content": t}} for t in texts]
    assert [dedup(r)[0] for r in rows] == [True, True, False, True, False, True]
    seeded = F.ExactDedup()
    seeded.seed([{"content": "a"}, {"content": "c"}])
    assert [seeded(r)[0] for r in rows] == [False, True, False, False, False, True]
    assert F.ExactDedup()._seen == set()  # a new instance starts empty


@pytest.mark.parametrize(
    ("text", "keep"),
    [
        ("abcdefghi一", False),  # 1 of 10 letters non-Latin, x = 0.05
        ("abcdefghijklmnopqrs一", True),  # 1 of 20 letters: exactly at the threshold, kept
        ("abcdefghijklmnopq一", False),  # 1 of 18 letters, above
        ("12345 +-*/ ()", True),  # no letters
        ("éèêü", True),  # Latin letters with diacritics
        ("αβ", False),  # Greek
    ],
)
def test_non_latin_letter_ratio_at_the_threshold(text, keep):
    assert F.NonLatinLetterRatioMax(0.05)({F.TEXT: text})[0] is keep


def test_letter_counts_are_exact():
    # letters: d e f, 函 数, x, r e t u r n, x, 注 释 = 15, of which 函 数 注 释 are not Latin
    assert F.letter_counts("def 函数(x): return x  # 注释") == (15, 4)


def test_score_and_source_filters():
    assert F.ScoreMin(0.7)({"score": "0.7"}) == (True, None)
    assert F.ScoreMin(0.7)({"score": "0.699"}) == (False, "score_min")
    assert F.ScoreMin(0.7)({"score": 0.71}) == (True, None)
    assert F.ScoreMin(0.7)({"score": None}) == F.ScoreMin(0.7)({"score": "n/a"}) == (False, "score_min")
    web = F.SourceIn(profiles.WEB_POOL)
    assert web({"source": "CCI3"}) == (True, None) and web({"source": "Tele"}) == (False, "source_in")


def test_script_stats_counts_a_fixed_simplified_traditional_pair():
    simplified = "这个国家的学生们说话时会认真对待问题。"
    traditional = "這個國家的學生們說話時會認真對待問題。"
    # markers: 这 个 国 学 们 说 话 时 会 认 对 问 (12); 家 的 生 真 待 题 are the same in both scripts
    assert F.script_stats(simplified) == {"hans": 12, "hant": 0}
    assert F.script_stats(traditional) == {"hans": 0, "hant": 12}
    assert F.script_stats("的 是 abc 123") == {"hans": 0, "hant": 0}
    assert F.ScriptStats()(simplified + traditional) == {"hans": 12, "hant": 12}


def test_the_script_table_has_48_distinct_pairs():
    assert len(F.SCRIPT_PAIRS) == 48 and F.SCRIPT_TABLE_VERSION == 1
    assert all(len(p) == 2 and p[0] != p[1] for p in F.SCRIPT_PAIRS)
    assert len(F.HANS) == len(F.HANT) == 48 and not F.HANS & F.HANT


def test_drop_counts_by_filter_equal_hand_counts_through_a_pipeline():
    profile = profiles.resolve("tlc-python-permissive", {"non_latin_max": 0.2})
    pipeline = profiles.Pipeline(profile, get("stack3-py"))
    ok = "def f(x):\n    return x\n"
    rows = [
        {LT: "permissive", VENDOR: False, "content": ok.encode()},  # kept
        {LT: "no_license", VENDOR: False, "content": ok.encode()},  # licence
        {LT: "permissive", VENDOR: True, "content": b"other"},  # vendor
        {LT: "no_license", VENDOR: True, "content": b"x"},  # licence (first failing filter)
        {LT: "permissive", VENDOR: False, "content": b"\xff"},  # utf8
        {LT: "permissive", VENDOR: False, "content": ok.encode()},  # duplicate of the first kept
        {LT: "permissive", VENDOR: False, "content": "# 中文注释说明\nx = 1\n".encode()},  # non-Latin 6/7
        {LT: "permissive", VENDOR: False, "content": b"y = 2\n"},  # kept
    ]
    from collections import Counter

    drops = Counter(reason for keep, reason in map(pipeline, rows) if not keep)
    assert drops == {"licence_permissive": 2, "not_vendor": 1, "utf8_valid": 1, "exact_dedup": 1,
                     "non_latin_letter_ratio_max": 1}  # fmt: skip


@pytest.mark.parametrize("name", sorted(profiles.PROFILES))
def test_every_profile_resolves_with_utf8_valid_first_among_its_content_filters(name):
    profile = profiles.resolve(name, allow_unlicensed=True)
    assert profile.content_filters[0].name == "utf8_valid"
    for source_id in profile.sources:
        assert profiles.check_source(profile, source_id).id == source_id
        profiles.Pipeline(profile, get(source_id))
    assert profile.describe()["name"] == name and profile.filter_names()


def test_profile_parameters_and_refusals():
    assert profiles.resolve("tlc-python-permissive").params == {"non_latin_max": 0.10}
    english = profiles.resolve("tlc-english", {"score_min": 0.8})
    assert english.filter_names()[0] == "score_min(q=0.8,field=score)"
    assert profiles.resolve("tlc-english").metadata_filters == ()
    with pytest.raises(CorporaError, match="no parameters"):
        profiles.resolve("bilingual-en", {"score_min": 0.8})
    with pytest.raises(CorporaError, match="unknown profile"):
        profiles.resolve("tlc-python")
    with pytest.raises(CorporaError, match="--allow-unlicensed"):
        profiles.resolve("tlc-python-unlicensed-ablation")
    with pytest.raises(CorporaError, match="does not have"):
        profiles.check_source(profiles.resolve("tlc-python-permissive"), "code-l2-py")
    with pytest.raises(CorporaError, match="not a source of profile"):
        profiles.check_source(profiles.resolve("bilingual-en"), "ufw-zh")
    assert profiles.parse_params(["non_latin_max=0.05", "score_min=none"]) == {"non_latin_max": 0.05, "score_min": None}
    with pytest.raises(CorporaError):
        profiles.parse_params(["x"])
    assert profiles.ALL_ZH == frozenset(get("ufw-zh").licence.spdx) and len(profiles.ALL_ZH) == 8
