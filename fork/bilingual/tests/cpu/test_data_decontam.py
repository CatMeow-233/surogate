"""I4-D4: train/held-out overlap by language-appropriate n-grams (SCI-10)."""

import numpy as np
import pytest

from surogate_bilingual.data import decontam as D
from surogate_bilingual.data import overlap
from surogate_bilingual.data.docs import Doc, Segment

HAN = [chr(c) for c in range(0x4E00, 0x4E00 + 3000)]


def _zh(rng, n):
    return "".join(HAN[int(i)] for i in rng.integers(len(HAN), size=n)) + "。"


def _en(rng, n, vocab):
    return " ".join(vocab[int(i)] for i in rng.integers(len(vocab), size=n)) + "."


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(5)
    vocab = [f"w{i}" for i in range(5000)]
    held = [Doc(f"dev/en{i}", "en", (Segment(text=_en(rng, 60, vocab), lang="en"),)) for i in range(5)]
    held += [Doc(f"dev/zh{i}", "zh", (Segment(text=_zh(rng, 120), lang="zh"),)) for i in range(5)]
    train = [Doc(f"train/en{i}", "en", (Segment(text=_en(rng, 80, vocab), lang="en"),)) for i in range(40)]
    train += [Doc(f"train/zh{i}", "zh", (Segment(text=_zh(rng, 150), lang="zh"),)) for i in range(40)]
    # a copied English passage (20 words of a held-out document) inside fresh text
    en_copy = " ".join(held[0].text.split()[10:30])
    train.append(
        Doc(
            "train/plant-en",
            "en",
            (Segment(text=_en(rng, 30, vocab) + " " + en_copy + " " + _en(rng, 30, vocab), lang="en"),),
        )
    )
    # a copied Chinese passage (40 Han characters, no spaces) inside fresh Chinese text
    zh_copy = held[5].text[30:70]
    train.append(Doc("train/plant-zh", "zh", (Segment(text=_zh(rng, 50) + zh_copy + _zh(rng, 50), lang="zh"),)))
    return train, held


def test_i4_d4_english_word_13gram_detects_the_planted_copy(planted):
    train, held = planted
    report = D.overlap_report(train, held, rules=["en_word13"])
    assert set(report.flagged) == {"train/plant-en"}


def test_i4_d4_han_25gram_detects_the_chinese_copy_the_word_rule_misses(planted):
    train, held = planted
    words = D.overlap_report(train, held, rules=["en_word13"])
    assert "train/plant-zh" not in words.flagged  # regression: whitespace words of Chinese text never form 13-grams
    assert len(D.words(held[5].text)) < 13
    han = D.overlap_report(train, held, rules=["zh_han25"])
    assert set(han.flagged) == {"train/plant-zh"}


def test_i4_d4_token_13gram_detects_both(planted, data_pipeline):
    train, held = planted
    report = D.overlap_report(train, held, data_pipeline["tok"], rules=["token13"])
    assert set(report.flagged) == {"train/plant-en", "train/plant-zh"}
    with pytest.raises(D.DecontamError, match="needs the tokenizer"):
        D.overlap_report(train, held, None, rules=["token13"])


def test_i4_d4_filter_removes_exactly_the_planted_documents(planted, data_pipeline):
    train, held = planted
    report = D.overlap_report(train, held, data_pipeline["tok"])
    assert report.rules == ("en_word13", "zh_han25", "token13")
    assert set(report.flagged) == {"train/plant-en", "train/plant-zh"}
    kept = overlap.filter_docs(train, report.to_json())
    assert [d.key for d in kept] == [d.key for d in train if not d.key.startswith("train/plant")]
    doc = report.to_json()
    assert doc["format"] == "bilingual-overlap/1" and doc["n"] == {"en_word13": 13, "zh_han25": 25, "token13": 13}
    assert overlap.filter_docs(train, doc) == kept and overlap.flagged_keys(doc) == {"train/plant-en", "train/plant-zh"}


def test_i4_d4_ngram_hashes_match_a_brute_force_set():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 50, size=200).astype(np.uint64)
    b = np.concatenate([rng.integers(50, 100, size=100), a[37 : 37 + 13], rng.integers(50, 100, size=50)]).astype(
        np.uint64
    )
    grams_a = {tuple(a[i : i + 13]) for i in range(len(a) - 12)}
    grams_b = {tuple(b[i : i + 13]) for i in range(len(b) - 12)}
    brute = bool(grams_a & grams_b)
    hashed = bool(np.intersect1d(D.ngram_hashes(a, 13), D.ngram_hashes(b, 13)).size)
    assert brute and hashed
    assert len(set(D.ngram_hashes(a, 13).tolist())) == len(grams_a)  # no collision on distinct grams here
    assert D.ngram_hashes(a[:12], 13).size == 0


def test_a_short_held_out_sentence_inside_a_longer_training_document_is_flagged():
    """CR1 review probe: held-out text shorter than n (22 Han characters, 10 words) has no n-gram; it is indexed
    whole and found as a contiguous run inside longer training text. Shorter than SHORT_MIN is not tested."""
    zh, en = "我们今天去城市的图书馆看历史书然后在公园散步", "We walked to the city library and read history books"
    assert len([c for c in zh if D.is_han(c)]) == 22 and len(D.words(en)) == 10
    held = [Doc("dev/par0", "parallel", (Segment(text=zh, lang="zh"), Segment(text=en, lang="en")))]
    train = [
        Doc("train/a", "zh", (Segment(text="昨天下雨了。" + zh + "，晚上回家吃饭。", lang="zh"),)),
        Doc("train/b", "en", (Segment(text="On Sunday, " + en.lower() + " until noon.", lang="en"),)),
        Doc("train/c", "zh", (Segment(text="我们今天去城市的图书馆看历史书然后在家", lang="zh"),)),  # a prefix only
        Doc("train/d", "en", (Segment(text="We walked to the city library and read old maps", lang="en"),)),
    ]
    report = D.overlap_report(train, held, None, ["en_word13", "zh_han25"])
    assert report.flagged == {"train/a": ["zh_han25"], "train/b": ["en_word13"]}
    assert report.per_rule == {"en_word13": 1, "zh_han25": 1}
    tiny = [Doc("dev/t", "en", (Segment(text="Thank you very much.", lang="en"),))]  # 4 words < SHORT_MIN
    probe = D.overlap_report(
        [Doc("train/t", "en", (Segment(text="Thank you very much. Bye.", lang="en"),))], tiny, None, ["en_word13"]
    )
    assert probe.flagged == {} and probe.short_skipped == {"en_word13": 1}
    assert probe.to_json()["short_min"] == {"en_word13": 8}
