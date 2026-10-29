"""Decontamination (surogate_tlc.data.decontam, docs/data.md: 去污染) on toy benchmark files in the HumanEval, MBPP
and EvalPlus formats: the four rules, removal versus flags, the report and its use by a corpus spec."""

import ast
import gzip
import json
from collections import Counter

import pytest
from surogate_tlc import files
from surogate_tlc.data import corpus, decontam, shards

HE_PROMPT = '''from typing import List


def running_balance_report(entries: List[int], limit: int) -> List[str]:
    """ Walk through the ledger entries and keep a running balance of all deposits and withdrawals,
    reporting every step where the balance goes below zero or above the given limit.
    >>> running_balance_report([5, -10], 3)
    ['above', 'below']
    """
'''
HE_SOLUTION = """    balance = 0
    report = []
    for entry in entries:
        balance += entry
        if balance < 0:
            report.append('below')
        elif balance > limit:
            report.append('above')
        else:
            for step in range(len(report)):
                if report[step] == 'below' and balance > step:
                    report[step] = 'recovered'
    return report
"""
MBPP_CODE = """def count_matching_neighbours(grid, target):
    total = 0
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] != target:
                continue
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                r, c = row + dr, col + dc
                if 0 <= r < len(grid) and 0 <= c < len(grid[r]) and grid[r][c] == target:
                    total += 1
    return total // 2
"""
MBPP_TEXT = "Write a function to count the pairs of equal neighbouring cells in a grid of values."
CLEAN = """import os


def list_python_files(root):
    found = []
    for directory, _, names in os.walk(root):
        found += [os.path.join(directory, n) for n in names if n.endswith('.py')]
    return sorted(found)
"""


@pytest.fixture
def benchmarks(tmp_path):
    out = tmp_path / "bench"
    out.mkdir()
    he = {"task_id": "HumanEval/0", "prompt": HE_PROMPT, "canonical_solution": HE_SOLUTION}
    he["entry_point"] = "running_balance_report"
    with gzip.open(out / "HumanEval.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(he) + "\n")
    mbpp = {"task_id": 7, "text": MBPP_TEXT, "code": MBPP_CODE, "test_list": []}
    (out / "mbpp.jsonl").write_text(json.dumps(mbpp) + "\n", encoding="utf-8")
    plus = {"task_id": "Mbpp/7", "prompt": f'"""\n{MBPP_TEXT}\n"""', "canonical_solution": MBPP_CODE}
    with gzip.open(out / "MbppPlus.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(plus) + "\n")
    (out / "sanitized-mbpp.json").write_text(json.dumps([{"task_id": 7, "prompt": MBPP_TEXT, "code": MBPP_CODE}]))
    return out


def test_benchmark_formats(benchmarks):
    items, files = decontam.load_benchmarks(benchmarks)
    by_id = {item.id: item for item in items}
    assert sorted(by_id) == ["HumanEval/HumanEval/0", "MbppPlus/Mbpp/7", "mbpp/7", "sanitized-mbpp/7"]
    he = by_id["HumanEval/HumanEval/0"]
    assert he.code == HE_PROMPT + HE_SOLUTION and he.name == "running_balance_report"
    assert he.text.startswith("Walk through the ledger") and "def " not in he.text
    assert by_id["mbpp/7"].text == MBPP_TEXT and by_id["mbpp/7"].name == "count_matching_neighbours"
    assert by_id["MbppPlus/Mbpp/7"].code == MBPP_CODE
    assert [(f["file"], f["items"]) for f in files] == [
        ("HumanEval.jsonl.gz", 1),
        ("MbppPlus.jsonl.gz", 1),
        ("mbpp.jsonl", 1),
        ("sanitized-mbpp.json", 1),
    ]
    (benchmarks / "bad.jsonl").write_text('{"task_id": 1, "text": "no code"}\n')
    with pytest.raises(ValueError, match="neither canonical_solution nor code"):
        decontam.load_benchmarks(benchmarks)


def test_identifier_abstraction():
    src = 'def f(x):\n    """Doc."""\n    y = len(x) + 1  # note\n    return y * "s"\n'
    assert decontam.id_tokens(src) == "def ID ( ID ) : ID = len ( ID ) + NUM return ID * STR".split()
    assert decontam.rename("for item in data: print(item)") == "for zz_meti in zz_atad: print(zz_meti)"


@pytest.mark.parametrize(
    "a, b",
    [
        ("def f(a, b):\n    return a\nx = f(a='p', b='q')\n", "def g(c, d):\n    return c\ny = g(c='p', d='q')\n"),
        ("try:\n    f()\nexcept E as err:\n    g(err)\n", "try:\n    f()\nexcept E as e2:\n    g(e2)\n"),
        ("import numpy as np\nnp.zeros(3)\n", "import numpy as xp\nxp.zeros(3)\n"),
        ("def f():\n    global n\n    n = 1\n", "def f():\n    global m\n    m = 1\n"),
    ],
    ids=["keyword arguments", "except-as", "import-as", "global"],
)
def test_ast_hash_renames_every_binding(a, b):
    assert decontam.ast_hash(ast.parse(a)) == decontam.ast_hash(ast.parse(b))


def test_ast_hash_keeps_attributes_and_modules():
    assert decontam.ast_hash(ast.parse("x.size\n")) != decontam.ast_hash(ast.parse("x.shape\n"))
    assert decontam.ast_hash(ast.parse("import os\n")) != decontam.ast_hash(ast.parse("import sys\n"))


def test_code_blocks_are_the_shards_fences(trained_tokenizer):
    """Decontamination scans as code exactly what the shards frame as code, line ends other than \\n included."""
    from surogate_tlc.data import tokenizer

    text = "intro\u2028```python\ndef secret_solution_fn(x):\n    return x\n```\nafter\x0c```\nmore\n"
    special = tokenizer.special_ids(tokenizer.load(trained_tokenizer))
    pieces = shards.segments(text, "text", special)
    start, end = special[tokenizer.CODE_START], special[tokenizer.CODE_END]
    framed, inside = [], False
    for piece in pieces:
        if piece in (start, end):
            inside = piece == start
        elif inside and isinstance(piece, str):
            framed.append(piece)
    assert decontam.code_blocks(text) == "\n".join(framed)
    assert "secret_solution_fn" not in decontam.code_blocks(text)  # "intro\u2028```python" is not a fence line


def test_rules(benchmarks):
    index = decontam.Index(decontam.load_benchmarks(benchmarks)[0])
    renamed = CLEAN + "\n\n" + decontam.rename(HE_PROMPT + HE_SOLUTION)
    hits = index.scan(renamed, "code")
    rules = Counter(h["rule"] for h in hits if h["item"] == "HumanEval/HumanEval/0")
    assert rules["code13"] == 1 and rules["ast"] == 1
    assert [h["coverage"] for h in hits if h["rule"] == "code13"] == [1.0]
    assert {h["rule"] for h in decontam.removing(hits)} == {"code13", "ast"}
    assert index.scan(CLEAN, "code") == []
    # A text document: the description verbatim, and a solution in a markdown block.
    text = f"Exercise 7. {MBPP_TEXT} Try it first.\n\n```python\n{decontam.rename(MBPP_CODE)}```\n"
    found = {(h["rule"], h["item"]) for h in decontam.removing(index.scan(text, "text"))}
    assert {("text8", "mbpp/7"), ("code13", "mbpp/7"), ("ast", "mbpp/7")} <= found
    # Only the name: a flag, never a removal.
    named = "def count_matching_neighbours(a, b):\n    return a == b\n"
    hits = index.scan(named, "code")
    assert {h["rule"] for h in hits} == {"name"} and decontam.removing(hits) == []


def test_calibration(benchmarks):
    items, _ = decontam.load_benchmarks(benchmarks)
    result = decontam.calibrate(decontam.Index(items), items, [CLEAN, CLEAN.replace("root", "top")], plants=3)
    assert result["plants"] == 3 and result["recall"] == 1.0 and result["false_positive_rate"] == 0


def _corpus(root):
    (root / "code").mkdir(parents=True)
    (root / "code" / "clean.py").write_text(CLEAN)
    (root / "code" / "copied.py").write_text(CLEAN + "\n\n" + decontam.rename(MBPP_CODE))
    (root / "code" / "named.py").write_text("def count_matching_neighbours(a, b):\n    return a == b\n")
    spec = (
        'seed = 0\nval_fraction = 0.0\n\n[[source]]\nname = "code"\nkind = "code"\nfiles = ["code/*.py"]\n'
        'url = "https://example.invalid"\nlicense = "MIT"\ndownloaded = 2026-10-01\nfilters = []\n'
    )
    (root / "corpus.toml").write_text(spec)
    return root / "corpus.toml"


def test_report_and_corpus(benchmarks, tmp_path, trained_tokenizer, capsys):
    spec = _corpus(tmp_path / "c")
    report = tmp_path / "c" / "decontam.json"
    assert decontam.main([str(spec), str(benchmarks), str(report), "--plant", "2", "--hosts", "3"]) == 0
    data = json.loads(report.read_text())
    assert [e["key"] for e in data["removed"]] == ["code/code/copied.py"]
    assert [e["key"] for e in data["flagged"]] == ["code/code/named.py"]
    assert data["documents"] == {"code": 3} and data["calibration"]["plants"] == 2
    assert sorted(data["audit_sample"]) == ["code/code/copied.py", "code/code/named.py"]
    assert json.loads(capsys.readouterr().out)["removed"] == 1
    with pytest.raises(SystemExit, match="exists"):
        decontam.main([str(spec), str(benchmarks), str(report)])

    spec.write_text('decontam = "decontam.json"\n' + spec.read_text())
    c = corpus.load_corpus(spec)
    dropped = Counter()
    assert [d.key for d in corpus.documents(c, dropped)] == ["code/code/clean.py", "code/code/named.py"]
    assert dropped == {("code", "decontaminated"): 1}
    assert len(list(corpus.documents(corpus.load_corpus(spec, decontam=False)))) == 3
    record = {"report": "decontam.json", "sha256": files.file_sha256(report), "removed": 1}
    assert corpus.decontam_record(c) == record
    manifest = shards.build_shards(c, trained_tokenizer, tmp_path / "data", seq_len=64)
    assert manifest["corpus"]["decontam"] == record and manifest["sources"][0]["dropped"] == {"decontaminated": 1}

    (tmp_path / "c" / "code" / "copied.py").write_text(CLEAN + "\n# changed\n")
    with pytest.raises(ValueError, match="rescan"):
        list(corpus.documents(corpus.load_corpus(spec)))
