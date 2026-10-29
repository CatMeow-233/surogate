"""The native boundary: every place TLC builds something from the native extension `surogate._surogate`.

Nothing here is imported at package import time, and every function imports the extension only when called, so the
CPU tests (which have no extension) can import the modules that use these functions and pass stand-ins instead.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .files import file_sha256

# Upstream defaults a from-scratch TLC run must not inherit: LoRA and tokenize-time packing are on by default, and
# from_scratch would replace the initialised weights with the native initialisation (by parameter name only).
FIXED = {"lora": False, "sample_packing": False, "row_packing": False, "from_scratch": False}
# SFT settings of native_trainer() besides the model, output directory and shape: nothing trained or resumed, eager.
SCORING = {
    "gradient_accumulation_steps": 1,
    "max_steps": 1,
    "resume_from_checkpoint": False,
    "use_cuda_graphs": False,
    "recompute": False,
    **FIXED,
}
# A rank-1 adapter on the MLP down projections. Its B factor starts at zero (the native LoRA initialisation), so the
# adapted model computes the base model on the weights the LoRA trainer reads (with a tied head: the embedding, not a
# checkpoint's lm_head.weight); native next_token_logits only runs with resident LoRA weights.
SAMPLING_LORA = {
    "lora": True,
    "lora_rank": 1,
    "lora_alpha": 1,
    "lora_dropout": 0.0,
    "lora_target_modules": ["down_proj"],
}

# The shared library with the training runtime and the kernels, which the extension links and fork/dev build copies
# next to it (csrc/CMakeLists.txt: surogate-common).
COMMON_LIBRARY = "libsurogate-common.so"
# (inputs, targets, positions), int32 [batch, seq_len] -> log-probabilities of the targets, 0 where IGNORE.
Scorer = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
# (inputs int32 [batch, seq_len], last positions int32 [batch]) -> next-token logits float32 [batch, vocab].
NextLogits = Callable[[np.ndarray, np.ndarray], np.ndarray]


def native_wrapper(sft: Mapping[str, Any], train_files: Sequence[str], eval_files: Sequence[str]) -> Any:
    """SurogateTrainerWrapper for `sft`, built the way upstream's sft command builds it, minus tokenization."""
    from surogate.core.config.sft_config import SFTConfig
    from surogate.train.trainer import SurogateTrainerWrapper
    from surogate.utils.dict import DictDefault

    from . import register

    register()
    config = SFTConfig(DictDefault(dict(sft)))
    config.__post_init__()  # as surogate.train.tokenize.TokenizeDatasets does
    return SurogateTrainerWrapper(config, [str(f) for f in train_files], [str(f) for f in eval_files] or None)


def native_trainer(
    model_dir: Path, data: Path, out: Path, *, batch: int, seq_len: int, settings: Mapping[str, Any] | None = None
) -> Any:
    """A wrapper for scoring, sampling and gate tests: model_dir's weights, eager, nothing resumed or trained.

    Its train loader reads data's train shards only because the wrapper needs one; `out` receives upstream's files.
    `settings` override SCORING (SAMPLING_LORA for next_token_logits).
    """
    from .data.shards import shard_files

    sft = {
        "model": str(model_dir),
        "output_dir": str(out),
        "sequence_len": seq_len,
        "per_device_train_batch_size": batch,
    }
    return native_wrapper({**sft, **SCORING, **(settings or {})}, shard_files(data, "train"), [])


def native_scorer(model_dir: Path, data: Path, out: Path, *, batch: int, seq_len: int) -> Scorer:
    """compute_logprobs of a native trainer on `model_dir` (no LoRA) for batches of `batch` rows of `seq_len`."""
    trainer = native_trainer(model_dir, data, out, batch=batch, seq_len=seq_len).trainer

    def score(inputs: np.ndarray, targets: np.ndarray, positions: np.ndarray) -> np.ndarray:
        return np.asarray(trainer.compute_logprobs(inputs, targets, use_lora=False, position_ids=positions))

    return score


def native_next_logits(model_dir: Path, data: Path, out: Path, *, batch: int, seq_len: int) -> NextLogits:
    """next_token_logits of a native trainer on `model_dir` with the zero-initialised SAMPLING_LORA adapter."""
    trainer = native_trainer(model_dir, data, out, batch=batch, seq_len=seq_len, settings=SAMPLING_LORA).trainer

    def next_logits(inputs: np.ndarray, last: np.ndarray) -> np.ndarray:
        return np.asarray(trainer.next_token_logits(inputs, last))

    return next_logits


def native_tokenizer(model_dir: Path) -> Any:
    """The native tokenizer of a model or tokenizer directory (surogate._surogate.Tokenizer)."""
    from surogate._surogate import Tokenizer

    return Tokenizer.from_pretrained(str(model_dir))


def extension_fingerprint() -> dict[str, Any] | None:
    """Path and sha256 of the training extension this process would load, and of COMMON_LIBRARY next to it; found,
    not imported. The extension is a thin binding: a change to the runtime or a kernel alone can leave it byte for
    byte the same, so the library's sha256 is what tells two kernel builds apart (both written by fork/dev build)."""
    try:
        spec = importlib.util.find_spec("surogate._surogate")
    except ImportError:
        spec = None
    if spec is None or not spec.origin:
        return None
    library = Path(spec.origin).with_name(COMMON_LIBRARY)
    common = {"path": str(library), "sha256": file_sha256(library)} if library.is_file() else None
    return {"path": spec.origin, "sha256": file_sha256(Path(spec.origin)), COMMON_LIBRARY: common}
