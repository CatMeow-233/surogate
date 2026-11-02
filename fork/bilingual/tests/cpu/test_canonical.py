"""Canonical JSON, format checks and atomic writes (docs/spec.md, "数据契约摘要")."""

import json
import os

import pytest

from surogate_bilingual.canonical import (
    atomic_directory,
    atomic_write_json,
    canonical_json,
    check_format,
    remove_stale_temps,
    sha256_json,
)


def test_canonical_json_is_sorted_compact_utf8_and_refuses_nan():
    assert canonical_json({"b": 1, "a": ["中", 2.5]}) == '{"a":["中",2.5],"b":1}'.encode()
    assert sha256_json({"b": 1, "a": 2}) == sha256_json({"a": 2, "b": 1})
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_check_format_refuses_other_kinds_and_unknown_majors():
    assert check_format({"format": "bilingual-expert/1"}, "expert") == 1
    for doc, message in (
        ({"format": "bilingual-expert/2"}, "not a known version"),
        ({"format": "bilingual-seed/1"}, "expected a bilingual-expert"),
        ({}, "expected a bilingual-expert"),
    ):
        with pytest.raises(ValueError, match=message):
            check_format(doc, "expert")


def test_atomic_writes_leave_no_temporary_files(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": 1})
    atomic_write_json(tmp_path / "a.json", {"x": 2})
    assert json.loads((tmp_path / "a.json").read_text()) == {"x": 2}
    with atomic_directory(tmp_path / "pack") as tmp:
        (tmp / "f").write_text("data")
        assert not (tmp_path / "pack").exists()
    assert (tmp_path / "pack" / "f").read_text() == "data"
    with pytest.raises(FileExistsError):
        with atomic_directory(tmp_path / "pack"):
            pass
    with pytest.raises(RuntimeError):
        with atomic_directory(tmp_path / "broken"):
            raise RuntimeError("crash inside the write")
    assert sorted(os.listdir(tmp_path)) == ["a.json", "pack"]


def test_stale_temporaries_are_removed_on_the_next_start(tmp_path):
    (tmp_path / "a.json.tmp-123-abcd").write_text("partial")
    (tmp_path / "pack.tmp-123-ef01").mkdir()
    (tmp_path / "keep.json").write_text("{}")
    removed = remove_stale_temps(tmp_path)
    assert sorted(p.name for p in removed) == ["a.json.tmp-123-abcd", "pack.tmp-123-ef01"]
    assert os.listdir(tmp_path) == ["keep.json"]
