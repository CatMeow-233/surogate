"""I6-E7: the `bash runbook` blocks of docs/runbook.md.

Every line of every block is classified; a line of no known kind fails the test, so the runbook cannot grow a
command nobody checks.

- `python -m surogate_bilingual ...` lines (also through `$PY`) run in-process, in order, in a temporary run root:
  first with `--dry-run`, which must exit 0 and must not change a single file of the run root; then, because later
  lines read what earlier ones write, for real at tiny scale. The tiny-scale substitutions are the only changes
  (TINY below): Track S at the ci profile, the fixture corpus with five Track S seed documents, the `tiny` model
  profile with the fixture tokenizer's vocabulary, 2 training steps, the first few items. Lines that need CUDA
  (`bench`, `verify --device cuda`) only dry-run on a host without it.
- Environment lines (`venv`, `pip`, `python -c`) and `pytest` lines are only checked to be well formed.
- Nothing is written outside tmp_path: temporary directories go to tmp_path/tmp, and the checkout's fork/bilingual
  tree and data/ directory are unchanged afterwards.
"""

import glob
import json
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

from surogate_bilingual.__main__ import COMMANDS, is_implemented
from surogate_bilingual.__main__ import main as cli

BILINGUAL = Path(__file__).resolve().parents[2]
REPO = BILINGUAL.parents[1]
RUNBOOK = BILINGUAL / "docs" / "runbook.md"
ASSIGN = re.compile(r"^([A-Z_][A-Z0-9_]*)=(\S*)\s*(#.*)?$")
VAR = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)\}?")


def runbook_blocks(text: str) -> list[list[str]]:
    """The logical lines (continuations joined, comments kept) of every ```bash runbook fence."""
    blocks, current = [], None
    for raw in text.splitlines():
        line = raw.strip()
        if current is None:
            if line == "```bash runbook":
                current = []
            continue
        if line.startswith("```"):
            blocks.append(current)
            current = None
            continue
        if current and current[-1].endswith("\\"):
            current[-1] = current[-1][:-1].rstrip() + " " + line
        elif line:
            current.append(line)
    assert current is None, "an unterminated ```bash runbook block"
    return blocks


def kind_of(line: str) -> str:
    if ASSIGN.match(line):
        return "assign"
    words = shlex.split(line, comments=True)
    if "-m" in words and words[words.index("-m") + 1 :][:1] == ["surogate_bilingual"]:
        return "package"
    if "pytest" in words:
        return "pytest"
    if words[:3] == ["python3.12", "-m", "venv"] or (words and words[0].endswith("/pip")):
        return "env"
    if words and words[0].endswith("/python") and words[1:2] == ["-c"]:
        return "env"
    return "unknown"


def expand(line: str, env: dict[str, str]) -> list[str]:
    words = shlex.split(VAR.sub(lambda m: env.get(m.group(1), m.group(0)), line), comments=True)
    out = []
    for w in words:
        hits = sorted(glob.glob(w)) if any(c in w for c in "*?[") else []
        out += hits or [w]
    return out


@pytest.fixture(scope="module")
def blocks():
    return runbook_blocks(RUNBOOK.read_text("utf-8"))


def test_i6_e7_every_runbook_line_is_known_and_well_formed(blocks):
    assert len(blocks) >= 9
    seen = set()
    for block in blocks:
        for line in block:
            kind = kind_of(line)
            assert kind != "unknown", f"runbook line of no known kind: {line}"
            words = shlex.split(line, comments=True)
            if kind == "package":
                command = words[words.index("surogate_bilingual") + 1]
                assert command in COMMANDS and is_implemented(command), line
                seen.add(command)
            elif kind == "pytest":
                targets = [w for w in words if w.startswith("fork/bilingual/tests/")]
                assert targets and all((REPO / t).is_dir() for t in targets), line
                assert words[0].startswith("BILINGUAL_G0_RECORD=") or words[0] == "fork/.venv-bilingual/bin/python"
            elif kind == "env":
                assert "fork/.venv-bilingual" in line, line
                for w in words:
                    if w.startswith("fork/bilingual"):
                        assert (REPO / w).exists(), w
    # the gates use every evaluation command
    assert {
        "env",
        "bench",
        "synth",
        "train",
        "assemble",
        "verify",
        "eval-tasks",
        "eval-nll",
        "switch",
        "stats",
        "matrix",
    } <= seen


def test_i6_e7_the_runbook_passes_the_strict_doc_check():
    out = subprocess.run(
        [sys.executable, "-I", str(REPO / "fork" / "tools" / "check_doc_links.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )
    problems = [line for line in out.stdout.splitlines() if "fork/bilingual/docs/runbook.md" in line]
    assert not problems, problems


# --- the rehearsal --------------------------------------------------------------------------------------------------

TRAIN_TINY = [
    "--set",
    'model.profile="tiny"',
    "--set",
    "train.global_batch_tokens=64",
    "--set",
    "train.microbatch=2",
    "--set",
    "train.seq=16",
    "--set",
    "train.accum=2",
    "--set",
    "run.tokens=128",
    "--set",
    "registry.allow_dirty=true",
]
TINY = {
    "synth": ["--profile", "ci"],
    "eval-tasks": ["--limit", "4", "--n-boot", "50", "--max-new", "3", "--device", "cpu"],
    "eval-nll": ["--limit-docs", "6", "--device", "cpu"],
    "switch": ["--limit", "2", "--device", "cpu"],
}
CUDA_ONLY = ("bench",)


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()
    }


def _repo_state() -> str:
    return subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--ignored=matching",
            "--",
            "fork/bilingual",
            "data",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _run(argv: list[str]) -> int:
    try:
        return int(cli(argv))
    except SystemExit as exc:
        return int(exc.code or 0)


def _stage_corpus(root: Path, track_s_ci, corpus_writer) -> None:
    corpus = root / "data" / "corpus"
    spec = corpus_writer(corpus, n=300)
    lines = (track_s_ci[0] / "seed" / "track_s.jsonl").read_text("utf-8").splitlines()[:5]
    (corpus / "ts_small").mkdir()
    (corpus / "ts_small" / "seed.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(spec, "a", encoding="utf-8") as f:
        f.write(
            '\n[[source]]\nname = "track-s-seed"\nkind = "synthetic"\nfiles = ["ts_small/*.jsonl"]\n'
            'licence = "LicenseRef-generated"\nlicence_verified = true\n'
        )


def test_i6_e7_runbook_commands_dry_run_then_rehearse_at_tiny_scale(
    blocks, tmp_path, monkeypatch, track_s_ci, corpus_writer, capsys
):
    root, tmp = tmp_path / "root", tmp_path / "tmp"
    root.mkdir()
    tmp.mkdir()
    _stage_corpus(root, track_s_ci, corpus_writer)
    repo_before = _repo_state()
    monkeypatch.chdir(root)
    monkeypatch.setenv("TMPDIR", str(tmp))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp))
    ran, dry_only = [], []
    for block in blocks:
        env = {}
        for line in block:
            kind = kind_of(line)
            if kind == "assign":
                name, value, _ = ASSIGN.match(line).groups()
                env[name] = value
                continue
            if kind != "package":
                continue
            words = expand(line, env)
            # repository paths stay in the checkout (read only); every other path is under the run root
            args = [
                str(REPO / w) if w.startswith("fork/") else w for w in words[words.index("surogate_bilingual") + 1 :]
            ]
            command = args[0]
            extra = []
            if command == "train" and "--list" not in args:
                from tokenizers import Tokenizer

                vocab = Tokenizer.from_file(str(root / "data" / "tokenizer" / "tokenizer.json")).get_vocab_size()
                extra = [*TRAIN_TINY, "--set", f"model.vocab_size={vocab}"]
            before = _tree(root)
            assert _run([*args, *extra, "--dry-run"]) == 0, f"--dry-run failed: {line}\n{capsys.readouterr().err}"
            assert _tree(root) == before, f"--dry-run wrote files: {line}"
            needs_cuda = command in CUDA_ONLY or ("--device" in args and args[args.index("--device") + 1] == "cuda")
            if needs_cuda and not torch.cuda.is_available():
                dry_only.append(command)
                continue
            assert _run([*args, *extra, *TINY.get(command, [])]) == 0, f"failed: {line}\n{capsys.readouterr().err}"
            ran.append(command)
    capsys.readouterr()
    assert {"synth", "shard", "train", "assemble", "eval-tasks", "eval-nll", "switch", "stats", "matrix"} <= set(ran)
    n_package = sum(kind_of(line) == "package" for block in blocks for line in block)
    assert len(ran) + len(dry_only) == n_package and set(dry_only) <= {"bench", "verify"}
    # the rehearsal produced what the gates read
    data = root / "data"
    assert (data / "runs" / "_decisions" / "g1p.json").is_file() and (
        data / "runs" / "_decisions" / "futility_r0.json"
    ).is_file()
    futility = json.loads((data / "runs" / "_decisions" / "futility_r0.json").read_text())
    assert futility["format"] == "bilingual-stats/1" and futility["kind"] == "futility"
    preds = (data / "runs" / "_eval" / "main" / "r0" / "B4" / "predictions.jsonl").read_text().splitlines()
    assert preds and all(json.loads(p)["split"] == "dev" for p in preds)
    assert json.loads((data / "runs" / "_eval" / "main" / "r0" / "zh-en.nll.json").read_text())["matrix"]["rows"] == [
        "zh",
        "en",
    ]
    assert not [p for p in tmp.rglob("*") if p.is_file()]  # every temporary directory was cleaned up
    monkeypatch.chdir(tmp_path)
    assert _repo_state() == repo_before  # nothing outside tmp_path changed in the checkout
