"""Writes that never leave a partial file (surogate_tlc.files, and the shard writer): a failure while writing or
before the rename leaves the old file, or no file, under the final name."""

from pathlib import Path

import pytest
from surogate_tlc import files
from surogate_tlc.data import shards


def crash(*args):
    raise OSError("crash before the rename")


def partial_copy(src, dst):
    Path(dst).write_text("ne")
    raise OSError("disk full")


@pytest.mark.parametrize("stage", ["write", "rename"])
def test_write_text_atomic_leaves_the_old_file(tmp_path, monkeypatch, stage):
    target = tmp_path / "manifest.json"
    target.write_text("old")
    text = "new \ud800"  # cannot be encoded: fails after the temporary file is opened
    if stage == "rename":
        text = "new"
        monkeypatch.setattr(files.os, "replace", crash)
    with pytest.raises((UnicodeEncodeError, OSError)):
        files.write_text_atomic(target, text)
    assert target.read_text() == "old"
    monkeypatch.undo()
    files.write_text_atomic(target, "new")  # the leftover temporary file does not stop the next write
    assert target.read_text() == "new"


@pytest.mark.parametrize("stage", ["write", "rename"])
def test_copy_atomic_leaves_the_old_file(tmp_path, monkeypatch, stage):
    source, target = tmp_path / "src", tmp_path / "tokenizer.json"
    source.write_text("new")
    target.write_text("old")
    if stage == "write":
        monkeypatch.setattr(files.shutil, "copyfile", partial_copy)
    else:
        monkeypatch.setattr(files.os, "replace", crash)
    with pytest.raises(OSError):
        files.copy_atomic(source, target)
    assert target.read_text() == "old"


def test_a_shard_appears_complete_or_not_at_all(tmp_path, monkeypatch):
    writer = shards.ShardWriter(tmp_path / "train-00000.bin", seq_len=8, vocab_size=300, pad_id=0)
    writer.add_row([5, 6, 7])
    monkeypatch.setattr(shards.os, "replace", crash)
    with pytest.raises(OSError):
        writer.close()
    assert not (tmp_path / "train-00000.bin").exists()
