"""CPU tests run against this checkout: fork/bilingual/src goes first on sys.path, and a `surogate_bilingual`
imported from anywhere else (an installed copy, another checkout) errors the session before any test runs.
Nothing else is put on sys.path; the suite does not import the native `surogate` package.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

if str(SRC) in sys.path:
    sys.path.remove(str(SRC))
sys.path.insert(0, str(SRC))

import surogate_bilingual  # noqa: E402

if Path(surogate_bilingual.__file__).resolve().parent != SRC / "surogate_bilingual":
    raise RuntimeError(f"surogate_bilingual was imported from {surogate_bilingual.__file__}, not from {SRC}")

import importlib.util  # noqa: E402

import pytest  # noqa: E402

REFERENCE = Path(__file__).resolve().parents[2] / "reference"


@pytest.fixture(autouse=True)
def _one_cpu_thread():
    """Bitwise checks on CPU hold with one thread only (docs/spec.md, "实现步骤"); every test runs that way."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def load_reference(stem: str):
    """One of the owner's v1.0 scripts in `reference/`, loaded by path (they are not package modules)."""
    spec = importlib.util.spec_from_file_location(f"bilingual_reference_{stem}", REFERENCE / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def moe_pilot():
    return load_reference("moe_pilot")


@pytest.fixture(scope="session")
def verify_merge_math():
    return load_reference("verify_merge_math")


@pytest.fixture(scope="session")
def compare_v10():
    return load_reference("compare")


@pytest.fixture(scope="session")
def pilot():
    """The pilot budget report in its JSON form (what the golden file holds) and its rows by id. Built once per
    session: it constructs every row's model on the meta device and runs its forward (about 5 s)."""
    import json

    from surogate_bilingual.audit.budget import report

    doc = json.loads(json.dumps(report("pilot")))
    return doc, {row["id"]: row for row in doc["rows"]}


# --- data fixtures (step I4) ----------------------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
TRACK_S_SPEC = EXAMPLES / "track_s.toml"

ZH_WORDS = ("我们", "今天", "城市", "学校", "老师", "学生", "朋友", "天气", "公园", "图书馆", "火车", "电脑", "音乐",
            "电影", "河流", "山顶", "早餐", "晚上", "周末", "工作", "问题", "答案", "故事", "历史", "科学", "安静")  # fmt: skip
EN_WORDS = ("the", "river", "city", "school", "teacher", "student", "friend", "weather", "park", "library", "train",
            "computer", "music", "film", "mountain", "breakfast", "evening", "weekend", "work", "question", "answer",
            "story", "history", "science", "quickly", "green", "small", "large", "walks", "reads", "near", "with")  # fmt: skip
EXTRA = ("😀", "🎉", "𠀀", "𠀁", "𪚥", "\t")  # emoji, CJK Extension B, a tab


def zh_text(rng, n_words: int) -> str:
    words = [ZH_WORDS[int(i)] for i in rng.integers(len(ZH_WORDS), size=n_words)]
    out = []
    for i, w in enumerate(words):
        out.append(w)
        if i % 5 == 4:
            out.append("，" if i % 2 else "。")
        if rng.random() < 0.05:
            out.append(EXTRA[int(rng.integers(len(EXTRA)))])
    return "".join(out) + "。"


def en_text(rng, n_words: int) -> str:
    words = [EN_WORDS[int(i)] for i in rng.integers(len(EN_WORDS), size=n_words)]
    out = []
    for i, w in enumerate(words):
        if rng.random() < 0.05:
            w += EXTRA[int(rng.integers(len(EXTRA)))]
        out.append(w + ("," if i % 6 == 5 else ""))
    text = " ".join(out)
    return text[0].upper() + text[1:] + "."


def write_fixture_corpus(root: Path, seed: int = 0, n: int = 300, track_s: Path | None = None) -> Path:
    """A small corpus with every source kind; returns the spec path. zh text uses full-width punctuation, both
    languages carry emoji, CJK Extension B characters and tabs (I4-D1)."""
    import json

    import numpy as np

    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"text": zh_text(rng, int(rng.integers(8, 20)))}, ensure_ascii=False) for _ in range(n)]
    (root / "zh.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "en.txt").write_text("\n\n".join(en_text(rng, int(rng.integers(8, 20))) for _ in range(n)) + "\n",
                                 encoding="utf-8")  # fmt: skip
    pairs = [json.dumps({"id": f"p{i}", "pair_id": f"p{i}", "segments": [
        {"lang": "zh", "text": zh_text(rng, 6)}, {"lang": "en", "text": en_text(rng, 6)}]}, ensure_ascii=False)
        for i in range(n // 2)]  # fmt: skip
    (root / "parallel.jsonl").write_text("\n".join(pairs) + "\n", encoding="utf-8")
    cs = [json.dumps({"text": zh_text(rng, 4) + " " + en_text(rng, 5) + " " + zh_text(rng, 3)}, ensure_ascii=False)
          for _ in range(n // 2)]  # fmt: skip
    (root / "codeswitch.jsonl").write_text("\n".join(cs) + "\n", encoding="utf-8")
    spec = [
        '[[source]]\nname = "zh-nat"\nkind = "zh"\nfiles = ["zh.jsonl"]\nlicence = "CC-BY-SA-4.0"\n',
        '[[source]]\nname = "en-nat"\nkind = "en"\nfiles = ["*.txt"]\nlicence = "CC-BY-SA-4.0"\nlicence_verified = true\n',
        '[[source]]\nname = "par"\nkind = "parallel"\nfiles = ["parallel.jsonl"]\nlicence = "CC-BY-2.0-FR"\n',
        '[[source]]\nname = "cs"\nkind = "codeswitch"\nfiles = ["codeswitch.jsonl"]\nlicence = "CC0-1.0"\n',
    ]
    if track_s is not None:
        spec.append('[[source]]\nname = "track-s-seed"\nkind = "synthetic"\nfiles = ["track_s/seed/*.jsonl"]\n'
                    'licence = "LicenseRef-generated"\nlicence_verified = true\n')  # fmt: skip
    (root / "spec.toml").write_text("\n".join(spec), encoding="utf-8")
    return root / "spec.toml"


@pytest.fixture(scope="session")
def track_s_ci(tmp_path_factory):
    """The ci profile of examples/track_s.toml, built once per session (about 1 s): (out dir, data card)."""
    from surogate_bilingual.data.synth.build import build_track_s

    out = tmp_path_factory.mktemp("track_s") / "ci"
    return out, build_track_s(TRACK_S_SPEC, out, seed=20261102, profile="ci")


@pytest.fixture(scope="session")
def data_pipeline(tmp_path_factory, track_s_ci):
    """manifest -> split -> materialize -> tokenizer on the fixture corpus plus the Track S seed component."""
    import shutil

    from surogate_bilingual.data import manifest, split, tokenizer

    base = tmp_path_factory.mktemp("pipeline")
    corpus = base / "corpus"
    corpus.mkdir()
    shutil.copytree(track_s_ci[0] / "seed", corpus / "track_s" / "seed")
    spec = write_fixture_corpus(corpus, track_s=corpus / "track_s")
    m = manifest.load_manifest(manifest.build_manifest(spec, corpus, base / "manifest.json"))
    s = split.split_doc(m.iter_docs(), split.default_fractions(), seed=1, manifest_sha256=m.sha256)
    split.write_split(base / "split.json", s)
    derived = manifest.load_manifest(split.write_split_corpora(m, s, base / "splits"))
    files = [derived.root / e["path"] for e in derived.files(split="seed_train")]
    tok_path = tokenizer.train_tokenizer(files, 1024, base / "tok" / "tokenizer.json", derived)
    return {"base": base, "corpus": corpus, "manifest": m, "split": s, "derived": derived,
            "tokenizer_path": tok_path, "tok": tokenizer.load_tokenizer(tok_path),
            "tokenizer_sha256": tokenizer.tokenizer_sha256(tok_path)}  # fmt: skip


@pytest.fixture(scope="session")
def corpus_writer():
    """write_fixture_corpus(root, seed=0, n=300, track_s=None) -> spec path (test modules cannot import conftest)."""
    return write_fixture_corpus


@pytest.fixture(scope="session")
def text_makers():
    """(zh_text, en_text): random fixture text of n words from rng."""
    return zh_text, en_text


@pytest.fixture(scope="session")
def seed_mixture(data_pipeline, track_s_ci):
    """The D_seed mixture of the pipeline's seed_train pools, 80 Track S documents and extra natural text:
    (pools, docs, card)."""
    import numpy as np

    from surogate_bilingual.data.docs import Doc, Segment, iter_file_docs
    from surogate_bilingual.data.mixture import build_seed_mixture, pools_from_docs

    rng = np.random.default_rng(12)
    pools = pools_from_docs(data_pipeline["derived"].iter_docs(split="seed_train"))
    pools["track_s"] = list(iter_file_docs(track_s_ci[0] / "seed" / "track_s.jsonl", source=None, kind=None))[:80]
    # more natural text than the pipeline fixture holds, so the 80 % share can be filled
    pools["zh"] += [Doc(f"zh-extra/{i}", "zh", (Segment(text=zh_text(rng, 12), lang="zh"),)) for i in range(900)]
    pools["en"] += [Doc(f"en-extra/{i}", "en", (Segment(text=en_text(rng, 12), lang="en"),)) for i in range(900)]
    docs, card = build_seed_mixture(
        pools, data_pipeline["tok"], seed=7, sources=data_pipeline["manifest"].doc["sources"]
    )
    return pools, docs, card
