"""Model directories: what a directory used as model_dir holds, how to write one and how to check one.

A model directory holds config.json, the file its `auto_map` names (configuration_tlc.py for TLC rows), the
tokenizer files, tlc_meta.json and, once initialised or trained, model.safetensors. The same contract applies
to checkpoint and export directories after surogate_tlc.checkpoint.finalize_dir has run on them.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import hf

CONFIG_FILE = "config.json"
META_FILE = "tlc_meta.json"
WEIGHTS_FILE = "model.safetensors"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_FILES = ("special_tokens_map.json", "chat_template.jinja")
META_VERSION = 1
# Provenance that travels with the weights. Later phases fill the versions; null means "not used".
META_KEYS = ("init_seed", "label_ontology_version", "shard_version", "sidecar_version")


class ModelDirError(ValueError):
    def __init__(self, path: Path, problems: list[str]) -> None:
        self.path, self.problems = path, problems
        super().__init__(f"{path} is not a usable model directory:\n" + "\n".join(f"  - {p}" for p in problems))


def code_files(cfg: Mapping[str, Any]) -> list[str]:
    """Python files config.json's auto_map refers to, e.g. configuration_tlc.py."""
    return sorted({ref.split(".")[0] + ".py" for ref in (cfg.get("auto_map") or {}).values()})


def companion_files(path: Path) -> list[str]:
    """Files of the model directory `path` that checkpoint and export directories must carry, config.json first."""
    cfg = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
    tokenizer = [*TOKENIZER_FILES, *(name for name in OPTIONAL_TOKENIZER_FILES if (path / name).is_file())]
    return [CONFIG_FILE, *code_files(cfg), *tokenizer, META_FILE]


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def write_model_dir(
    out: Path,
    config: Mapping[str, Any],
    tokenizer_dir: Path,
    *,
    init_seed: int | None = None,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a model directory without weights (the init tool adds model.safetensors) and check it."""
    out, tokenizer_dir = Path(out), Path(tokenizer_dir)
    hf.check_config(config)
    out.mkdir(parents=True, exist_ok=True)
    for name in (*TOKENIZER_FILES, *OPTIONAL_TOKENIZER_FILES):
        if (tokenizer_dir / name).is_file():
            copy_atomic(tokenizer_dir / name, out / name)
    if code_files(config):
        if code_files(config) != [hf.CONFIGURATION_FILE]:
            raise ValueError(f"auto_map refers to {code_files(config)}; only {hf.CONFIGURATION_FILE} is provided")
        write_text_atomic(out / hf.CONFIGURATION_FILE, hf.configuration_source())
    write_text_atomic(out / CONFIG_FILE, dump_json(dict(config)))
    tlc_meta = {"tlc_meta_version": META_VERSION, **dict.fromkeys(META_KEYS), "init_seed": init_seed, **(meta or {})}
    write_text_atomic(out / META_FILE, dump_json(tlc_meta))
    check_model_dir(out, weights=False)
    return out


def _special_token(tokenizer_config: Mapping[str, Any], key: str) -> str | None:
    token = tokenizer_config.get(key)
    return token.get("content") if isinstance(token, dict) else token


def _tokenizer_problems(path: Path, cfg: Mapping[str, Any]) -> list[str]:
    from tokenizers import Tokenizer

    missing = [name for name in TOKENIZER_FILES if not (path / name).is_file()]
    if missing:
        return [f"missing tokenizer files {missing}"]
    problems = []
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    max_id = max(tokenizer.get_vocab(with_added_tokens=True).values())
    vocab = cfg.get("vocab_size")
    # The native trainer does not range-check token ids.
    if isinstance(vocab, int) and max_id >= vocab:
        problems.append(f"tokenizer ids reach {max_id}, but vocab_size is {vocab}")
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    for key in hf.TOKEN_ID_KEYS:
        name = key.removesuffix("_id")
        token = _special_token(tokenizer_config, name)
        token_id = tokenizer.token_to_id(token) if token else None
        if token_id is None or token_id != cfg.get(key):
            problems.append(f"{name} {token!r} has id {token_id} in the tokenizer, config.json says {cfg.get(key)!r}")
    if not tokenizer_config.get("chat_template") and not (path / "chat_template.jinja").is_file():
        problems.append("the tokenizer has no chat template")
    return problems


def check_model_dir(path: Path, *, weights: bool = True) -> dict[str, Any]:
    """Check `path` against the contract and return its config; raise ModelDirError listing every problem."""
    path = Path(path)
    if not (path / CONFIG_FILE).is_file():
        raise ModelDirError(path, [f"missing {CONFIG_FILE}"])
    cfg = json.loads((path / CONFIG_FILE).read_text(encoding="utf-8"))
    problems = hf.config_problems(cfg)
    problems += [f"missing {name}, which auto_map refers to" for name in code_files(cfg) if not (path / name).is_file()]
    problems += _tokenizer_problems(path, cfg)
    meta_path = path / META_FILE
    if not meta_path.is_file():
        problems.append(f"missing {META_FILE}")
    else:
        tlc_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(tlc_meta, dict) or tlc_meta.get("tlc_meta_version") != META_VERSION:
            problems.append(f"{META_FILE} must be an object with tlc_meta_version {META_VERSION}")
        else:
            problems += [f"{META_FILE} lacks {key}" for key in META_KEYS if key not in tlc_meta]
    if weights and not (path / WEIGHTS_FILE).is_file():
        problems.append(f"missing {WEIGHTS_FILE}")
    if problems:
        raise ModelDirError(path, problems)
    return cfg
