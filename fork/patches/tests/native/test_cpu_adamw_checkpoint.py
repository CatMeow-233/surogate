"""Ledger topic T21 (cpu_training checkpoints its CPU AdamW moments), runnable without a GPU.

Upstream's cpu_training kept the FP32 moments of its CPU AdamW step out of the optimizer containers, so adam.m and
adam.v held no tensors, a resume allocated the GPU path's 8-bit state on the device (unused) and the moments
restarted at zero, and the reader skipped the missing names without a word. Two kinds of tests:

- The host checks: fork/native/tests/host/cpu_adamw_checkpoint.cpp, compiled with the host compiler from the
  production SafeTensorWriter/Reader sources (csrc/src/utilities) and fork/native/src/cpu_adamw_checkpoint.cpp,
  against the CUDA headers and libcudart. No GPU is used: a save, a resume and a byte comparison of the moments,
  the refusal of checkpoints without them (upstream's, and the 8-bit junk later saves wrote), the opt-in, the
  version tag, the reader's shape and dtype guards and the save opt-out. They need a CUDA toolkit (CUDA_HOME,
  else CUDA_PATH, else /usr/local/cuda) and the fmt and nlohmann_json headers (FORK_FC_DEPS, a directory with
  CMake FetchContent's fmt-src and json-src; else csrc/build/_deps; else the system's); without them they are
  skipped with the reason. Fork CI has neither, so there only the pins below run.
- Source pins of the upstream call sites (the optimizer's resume branch and its containers, the load-time check,
  the writer's host branch) and of the upstream code T21 relies on, unedited.

The GPU half (no 8-bit device state at a cpu_training resume, and an identical first update after it) is
unverified here; fork/docs/upstream-errata.md gives the 5090 check.
"""

import concurrent.futures
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
UTILITIES = REPO / "csrc/src/utilities"
OPTIMIZER = REPO / "csrc/src/runtime/optimizers/adamw_8bit_optimizer.cpp"
CHECKPOINT = REPO / "csrc/src/runtime/training/checkpoint.cpp"
FORK_NATIVE = REPO / "fork/native"
FORK_HEADER = FORK_NATIVE / "include/surogate_fork/cpu_adamw_checkpoint.h"
FORK_SOURCE = FORK_NATIVE / "src/cpu_adamw_checkpoint.cpp"
HOST_CHECKS = FORK_NATIVE / "tests/host"

#: Every check the host program runs, in its order; a check that disappears from it fails here.
HOST_CHECK_NAMES = [
    "layout_matches_upstream_step_order_and_steps_a_tied_master_once",
    "initialize_allocates_zeroed_moments_once",
    "pageable_host_memory_is_recognized",
    "save_writes_one_fp32_tensor_per_master_and_the_format_tag",
    "resume_round_trips_the_moments_bit_for_bit",
    "negative_control_upstream_reader_skips_the_missing_moments_silently",
    "legacy_header_only_file_is_refused",
    "legacy_file_with_unused_8bit_state_is_refused",
    "legacy_file_resumes_with_zero_moments_when_opted_in",
    "allow_missing_is_off_for_0_and_empty",
    "finish_load_refuses_files_read_without_require_complete",
    "checkpoint_of_another_parameter_set_is_refused_by_name",
    "another_format_version_is_refused_even_when_opted_in",
    "shape_and_dtype_changes_are_refused_by_the_reader",
    "save_opt_out_writes_no_moments_but_a_load_still_reads_them",
    "other_containers_are_not_checked",
]

#: The production sources the host program links: the writer and reader, the reader's host-target read path
#: (cu_file_fallback.cpp; cu_file.cpp, the cuFile build, reads host targets through the same posix_read_bytes),
#: and what they call.
PRODUCTION_SOURCES = [
    UTILITIES / "safetensors.cpp",
    UTILITIES / "cu_file_fallback.cpp",
    UTILITIES / "cu_file_common.cpp",
    UTILITIES / "tensor.cpp",
    UTILITIES / "dtype.cpp",
    UTILITIES / "utils.cpp",
    FORK_SOURCE,
]


def body(path: Path, signature: str) -> str:
    """The brace-balanced body of the first definition matching `signature` (a regex), whitespace collapsed."""
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, start = 1, match.end()
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return " ".join(text[start:end].split())
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


# ---------------------------------------------------------------------------------------------------------------
# Host checks


def _cuda_home() -> Path | None:
    for candidate in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH"), "/usr/local/cuda"):
        if not candidate:
            continue
        root = Path(candidate)
        headers = ("cuda_runtime.h", "cuda_bf16.h", "cuda_fp8.h", "cublas_v2.h", "nvtx3/nvToolsExt.h")
        if all((root / "include" / h).exists() for h in headers) and _cudart_dir(root):
            return root
    return None


def _cudart_dir(root: Path) -> Path | None:
    for lib in ("lib64", "lib", "targets/x86_64-linux/lib"):
        if (root / lib / "libcudart.so").exists():
            return root / lib
    return None


def _header_dirs() -> list[Path] | None:
    """Include directories that provide fmt/core.h and nlohmann/json.hpp, or None."""
    roots = [Path(os.environ["FORK_FC_DEPS"])] if os.environ.get("FORK_FC_DEPS") else []
    roots += sorted(REPO.glob("csrc/build*/_deps"))
    for root in roots:
        fmt, json = root / "fmt-src/include", root / "json-src/include"
        if (fmt / "fmt/core.h").exists() and (json / "nlohmann/json.hpp").exists():
            return [fmt, json]
    if Path("/usr/include/fmt/core.h").exists() and Path("/usr/include/nlohmann/json.hpp").exists():
        return []
    return None


@pytest.fixture(scope="module")
def host_checks(tmp_path_factory) -> dict[str, str]:
    """Build and run the host program once: {check name: "" when it passed, else why it failed}."""
    cuda = _cuda_home()
    headers = _header_dirs()
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if cuda is None or headers is None or compiler is None:
        pytest.skip(
            "the T21 host checks need a C++20 compiler, a CUDA toolkit (CUDA_HOME with cuda_runtime.h, cublas_v2.h, "
            "nvtx3 and libcudart.so) and the fmt and nlohmann_json headers (FORK_FC_DEPS or csrc/build/_deps); "
            f"found compiler={compiler}, CUDA_HOME={cuda}, headers={headers}"
        )
    build = tmp_path_factory.mktemp("t21-host")
    flags = ["-std=c++20", "-O1", "-DFMT_HEADER_ONLY", f"-I{FORK_NATIVE / 'include'}", f"-I{REPO / 'csrc/src'}"]
    flags += ["-isystem", str(cuda / "include")]
    for directory in headers:
        flags += ["-isystem", str(directory)]
    sources = [HOST_CHECKS / "cpu_adamw_checkpoint.cpp", HOST_CHECKS / "stubs.cpp", *PRODUCTION_SOURCES]

    def compile_one(index_source):
        index, source = index_source
        obj = build / f"{index:02d}_{source.stem}.o"
        result = subprocess.run([compiler, *flags, "-c", str(source), "-o", str(obj)], capture_output=True, text=True)
        assert result.returncode == 0, f"compiling {source.relative_to(REPO)} failed:\n{result.stderr[-4000:]}"
        return obj

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 2) as pool:
        objects = list(pool.map(compile_one, enumerate(sources)))
    binary = build / "cpu_adamw_checkpoint"
    lib = _cudart_dir(cuda)
    link = [compiler, *map(str, objects), f"-L{lib}", "-lcudart", f"-Wl,-rpath,{lib}", "-o", str(binary)]
    result = subprocess.run(link, capture_output=True, text=True)
    assert result.returncode == 0, f"linking the host checks failed:\n{result.stderr[-4000:]}"

    scratch = build / "files"
    scratch.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("SUROGATE_FORK_CPU_ADAMW_")}
    result = subprocess.run([str(binary), str(scratch)], capture_output=True, text=True, env=env, timeout=60)
    outcomes: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if line.startswith("ok "):
            outcomes[line[3:]] = ""
        elif line.startswith("FAIL "):
            name, _, why = line[5:].partition(": ")
            outcomes[name] = why or "failed"
    if "setup" in outcomes:
        pytest.fail(f"the host checks could not set up: {outcomes['setup']}")
    assert result.returncode == sum(1 for why in outcomes.values() if why), (result.stdout, result.stderr)
    return outcomes


def test_the_host_program_runs_exactly_the_listed_checks(host_checks):
    assert list(host_checks) == HOST_CHECK_NAMES


@pytest.mark.parametrize("name", HOST_CHECK_NAMES)
def test_host_check(host_checks, name):
    assert name in host_checks, f"the host program did not run {name}"
    assert host_checks[name] == "", host_checks[name]


# ---------------------------------------------------------------------------------------------------------------
# Upstream call sites


def test_a_cpu_training_resume_lays_out_host_moments_and_allocates_no_8bit_device_state():
    prepare = body(OPTIMIZER, r"void AdamW8BitOptimizer::prepare_for_checkpoint_load\(")
    branch = "if (model.mGrads && model.mGrads->is_streaming_grads()) {"
    assert branch in prepare
    resume = prepare[prepare.index(branch) : prepare.index("return; }", prepare.index(branch))]
    assert "mImpl->ensure_cpu_streaming_state(model);" in resume
    assert "mImpl->cpu_checkpoint.begin_load();" in resume
    # The branch returns before init_state, which would allocate the 8-bit state.
    assert prepare.index(branch) < prepare.index("init_state(model, stream);")


def test_the_first_step_and_a_resume_share_one_layout():
    ensure = body(OPTIMIZER, r"void ensure_cpu_streaming_state\(")
    assert "if (cpu_checkpoint.active()) return;" in ensure
    assert "for (const auto& name : model.mGrads->param_names()) {" in ensure
    assert "params.push_back({name, &model.mWeightManager->get_master(name)});" in ensure
    assert "cpu_checkpoint.initialize(cpu_state, params);" in ensure
    streaming = body(OPTIMIZER, r"void AdamW8BitOptimizer::step_cpu_streaming\(")
    assert "mImpl->ensure_cpu_streaming_state(model);" in streaming
    assert "for (const auto& slot : mImpl->cpu_checkpoint.slots()) {" in streaming
    assert "const std::size_t offset = slot.offset;" in streaming
    assert "if (n != slot.n) {" in streaming
    # No second allocation or offset arithmetic outside the layout.
    assert ".resize(" not in streaming
    assert "offset +=" not in streaming
    for kernel in ("cpu_adamw_step(", "cpu_adamw_step_bf16(", "cpu_adamw_step_bf16_param("):
        call = streaming[streaming.index(kernel) :]
        assert call.split(";")[0].count("cpu_state.m.data() + offset, cpu_state.v.data() + offset") == 1, kernel


def test_the_optimizer_exposes_the_cpu_moments_once_they_exist():
    momentum = body(OPTIMIZER, r"ITensorContainer\* AdamW8BitOptimizer::momentum_container\(")
    variance = body(OPTIMIZER, r"ITensorContainer\* AdamW8BitOptimizer::variance_container\(")
    assert "if (mImpl->cpu_checkpoint.active()) return mImpl->cpu_checkpoint.momentum();" in momentum
    assert "if (mImpl->cpu_checkpoint.active()) return mImpl->cpu_checkpoint.variance();" in variance
    restore = body(OPTIMIZER, r"void AdamW8BitOptimizer::on_restore_checkpoint\(")
    assert restore.startswith("if (mImpl->cpu_checkpoint.active()) {")
    finish = "mImpl->cpu_checkpoint.finish_load(); return; }"
    assert finish in restore
    assert restore.index(finish) < restore.index("if (mImpl->state && mImpl->state->state1.Data) {")


def test_load_checkpoint_checks_each_optimizer_file_before_reading_it():
    load = body(CHECKPOINT, r"void load_checkpoint\(")
    for moment, container in (("m", "model.opt_momentum()"), ("v", "model.opt_variance()")):
        file = f'source + fmt::format("/adam.{moment}.shard_{{:03}}_of_{{:03}}.safetensors", shard_rank, shard_count)'
        check = f"surogate_fork::cpu_adamw_checkpoint::require_complete( {file}, {container});"
        read = f"load_safetensors({file}, {container}, false);"
        assert check in load and read in load, moment
        assert load.index(check) < load.index(read)
    # Both run after the optimizer laid out the state to read into, and before on_restore_checkpoint.
    assert load.index("model.prepare_optimizer_for_checkpoint_load();") < load.index("require_complete(")
    assert load.rindex("require_complete(") < load.index("model.on_restore_checkpoint(comm);")


def test_the_writer_copies_pageable_host_memory_itself():
    source = UTILITIES / "safetensors.cpp"
    assert '#include "surogate_fork/pageable_host.h"' in source.read_text(encoding="utf-8")
    write = body(source, r"void SafeTensorWriter::write_tensor\(")
    branch = (
        "if (surogate_fork::is_pageable_host(tensor.Data, tensor.Device)) { "
        "std::memcpy(mMappedFile + shard_begin, tensor.Data, tensor.bytes()); found->second.Done = true; return; }"
    )
    assert branch in write
    assert write.index(branch) < write.index("cudaMemcpy(")
    raw = body(source, r"void SafeTensorWriter::write_raw\(")
    branch = (
        "if (surogate_fork::is_pageable_host(tensor.Data, tensor.Device)) { "
        "std::memcpy(mMappedFile + write_start, tensor.Data, write_size); return; }"
    )
    assert branch in raw
    assert raw.index(branch) < raw.index("cudaMemcpy(")


def test_the_fork_source_is_part_of_the_training_library():
    text = (FORK_NATIVE / "cmake/fork_native.cmake").read_text(encoding="utf-8")
    function = text[text.index("function(surogate_fork_configure_common target)") : text.index("endfunction()")]
    assert 'target_sources(${target} PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/src/cpu_adamw_checkpoint.cpp")' in function


def test_the_documented_switches_and_format_are_the_built_ones():
    header = FORK_HEADER.read_text(encoding="utf-8")
    switches = dict(re.findall(r'inline constexpr const char\* (k\w+Env) = "(\w+)";', header))
    assert switches == {
        "kSaveStateEnv": "SUROGATE_FORK_CPU_ADAMW_SAVE_STATE",
        "kAllowMissingStateEnv": "SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE",
    }
    assert "inline constexpr std::int32_t kFormatVersion = 1;" in header
    performance = (REPO / "fork/docs/performance.md").read_text(encoding="utf-8")
    errata = (REPO / "fork/docs/upstream-errata.md").read_text(encoding="utf-8")
    for name in switches.values():
        assert name in performance, name
    for name in ("cpu_adamw.m.", "cpu_adamw.v.", "cpu_adamw.format"):
        assert name in performance, name
    assert "cpu_adamw_checkpoint" in errata


# ---------------------------------------------------------------------------------------------------------------
# Upstream code T21 relies on, unedited


def test_upstream_reader_still_skips_names_the_file_lacks():
    """Why require_complete() exists: load_tensors reads the names both have. If upstream starts refusing missing
    names itself, the check is redundant (and GPU-optimizer checkpoints would need a look)."""
    load = body(UTILITIES / "safetensors.cpp", r"void SafeTensorsReader::load_tensors\(")
    skip = (
        "for (const auto& entry : mEntries) if (auto found = named_tensors.find(entry.name()); "
        "found != named_tensors.end()) entry.read_tensor(found->second, allow_cast);"
    )
    assert skip in load


def test_upstream_reader_fills_host_targets_with_pread():
    entry = body(UTILITIES / "safetensors.cpp", r"void SafeTensorEntry::read_raw\(")
    assert "const bool host_target = target.Device < 0;" in entry
    assert "mHandle->read_bytes(target.Data, start, end, host_target);" in entry
    posix = body(UTILITIES / "cu_file_common.cpp", r"void posix_read_bytes\(")
    assert "ssize_t r = ::pread(fd, host_target ? target + done : hbuf, want, off);" in posix
    gds = body(UTILITIES / "cu_file.cpp", r"void cuFileRef::read_bytes\(")
    assert gds.startswith("if (uses_gds() && host_target) {")


def test_upstream_turns_on_streaming_before_a_checkpoint_is_loaded():
    """prepare_for_checkpoint_load's branch keys on is_streaming_grads(): allocate_run_state (the trainer
    constructor) sets it for cpu_training, before load_checkpoint runs."""
    allocate = body(REPO / "csrc/src/runtime/dsl/dsl_model_execution.cpp", r"void DslModel::allocate_run_state\(")
    assert "if (mGrads && mOptions.CpuTraining && !mGrads->param_names().empty()) {" in allocate
    assert "mGrads->enable_streaming(*mParams);" in allocate
    prepare = body(
        REPO / "csrc/src/runtime/dsl/dsl_model_weights.cpp", r"void DslModel::prepare_optimizer_for_checkpoint_load\("
    )
    assert "cfg.type = optimizers::OptimizerType::ADAMW_8BIT;" in prepare
    assert "mOptimizer->prepare_for_checkpoint_load(*this);" in prepare


def test_upstream_trainer_continues_the_bias_correction_step():
    """The step count is not part of the checkpoint's moments: the SFT trainer passes step + 1 from the resumed
    step, so the first update after a resume is the uninterrupted run's."""
    trainer = (REPO / "surogate/train/trainer.py").read_text(encoding="utf-8")
    assert trainer.count("self.trainer.update_with_config(opt_config, step + 1)") >= 2
