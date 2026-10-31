"""Ledger topic T25 (GPT-OSS prequantized LoRA computes the frozen-expert input gradient), runnable without CUDA.

dispatch_moe_grouped_gemm_gate_up_backward and dispatch_moe_grouped_gemm_down_backward skip the base dgrad GEMM and
zero d_input when `skip_base_backward` holds, leaving only the grouped LoRA term in d_input. Upstream set it for
every LoRA run on a prequantized GPT-OSS checkpoint (7a8d5fee, unconditional since 1181c715). The down backward's
d_input is d_act, from which the activation backward forms the gate_up gradient, so the expert gate_up LoRA saw only
the down-LoRA term (exactly zero while the down LoRA B is at its zero init) and nothing but that term reached the
attention LoRA and the earlier layers through the MoE branch. The fork computes the dgrad by default; the switch
SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD (fork/native/include/surogate_fork/gpt_oss_prequant_dgrad.h) restores
the shortcut.

The probe compiles each file's `skip_base_backward` statement as it stands in the source, with the switch header and
the real PretrainedConfig::ArchitectureId enum, and evaluates it for every combination of its inputs and switch
values. The emulation then runs a GPT-OSS expert in float64 with LoRA on gate_up and down, takes the branch the
probe chose for the GPT-OSS prequantized LoRA case with the switch unset, and compares the LoRA and input gradients
with torch autograd. On upstream's sources (488e5f00) the probe skips with the switch unset, the truth-table tests
fail, and the emulated expert gate_up LoRA gradient is exactly zero where autograd's is not. What cuBLAS computes is
GPU-unverified here; the errata section for T25 in fork/docs/upstream-errata.md gives the 5090 check.
"""

import itertools
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
OPS = REPO / "csrc/src/runtime/ops"
FORK_INCLUDE = REPO / "fork/native/include"
SWITCH = "SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD"
SITES = {
    "gate_up": (OPS / "moe_grouped_gemm_gate_up.cpp", r"void CompiledExecutor::dispatch_moe_grouped_gemm_gate_up_backward\("),
    "down": (OPS / "moe_grouped_gemm_down.cpp", r"void CompiledExecutor::dispatch_moe_grouped_gemm_down_backward\("),
}


def strip_comments(text: str) -> str:
    """The text with // and /* */ comments blanked; string and character literals kept."""
    out, i = [], 0
    while i < len(text):
        two = text[i : i + 2]
        if two == "//":
            j = text.find("\n", i)
            i = len(text) if j < 0 else j
            continue
        if two == "/*":
            i = text.index("*/", i) + 2
            out.append(" ")
            continue
        c = text[i]
        if c in "\"'":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def function_body(path: Path, signature: str) -> str:
    """The brace-balanced body of the first definition matching `signature` (a regex), comments removed."""
    text = strip_comments(path.read_text(encoding="utf-8"))
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, start = 1, match.end()
    for end in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return text[start:end]
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def skip_statement(site: str) -> str:
    path, signature = SITES[site]
    body = function_body(path, signature)
    found = re.findall(r"const bool skip_base_backward\s*=[^;]*;", body)
    assert len(found) == 1, f"{path.name}: expected one skip_base_backward definition, found {len(found)}"
    return found[0]


def architecture_enum() -> str:
    text = strip_comments((REPO / "csrc/src/config/pretrained_config.h").read_text(encoding="utf-8"))
    match = re.search(r"enum ArchitectureId\s*\{[^}]*\};", text)
    assert match, "pretrained_config.h: no enum ArchitectureId"
    return match.group(0)


PROBE = r"""
#include <cstdio>
#include <cstdlib>
#include <cstring>
#if __has_include("surogate_fork/gpt_oss_prequant_dgrad.h")
#include "surogate_fork/gpt_oss_prequant_dgrad.h"
#endif

struct PretrainedConfig {
    @ENUM@
    ArchitectureId Architecture = LLAMA;
};
struct RunState {
    bool lora_only = false;
    bool prequant = false;
    bool is_lora_only_mode() const { return lora_only; }
    bool is_prequantized() const { return prequant; }
};

static bool decide_@SITE@(bool lora_enabled, const RunState& mRunState, const PretrainedConfig& mConfig) {
    @STATEMENT@
    return skip_base_backward;
}

// argv: lora_enabled lora_only prequant gpt_oss (0/1); prints 1 when the base dgrad is skipped
int main(int argc, char** argv) {
    if (argc != 5) return 2;
    RunState rs;
    rs.lora_only = std::atoi(argv[2]) != 0;
    rs.prequant = std::atoi(argv[3]) != 0;
    PretrainedConfig cfg;
    cfg.Architecture = std::atoi(argv[4]) != 0 ? PretrainedConfig::GPT_OSS : PretrainedConfig::QWEN3_MOE;
    std::printf("%d\n", decide_@SITE@(std::atoi(argv[1]) != 0, rs, cfg) ? 1 : 0);
    return 0;
}
"""


@pytest.fixture(scope="module")
def probes(tmp_path_factory):
    cxx = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("no C++ compiler")
    out = tmp_path_factory.mktemp("t25")
    built = {}
    for site in SITES:
        src = out / f"probe_{site}.cpp"
        src.write_text(
            PROBE.replace("@ENUM@", architecture_enum())
            .replace("@STATEMENT@", skip_statement(site))
            .replace("@SITE@", site),
            encoding="utf-8",
        )
        exe = out / f"probe_{site}"
        result = subprocess.run(
            [cxx, "-std=c++20", "-O0", "-Wall", "-Werror", f"-I{FORK_INCLUDE}", str(src), "-o", str(exe)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{site} probe does not compile:\n{result.stderr}"
        built[site] = exe
    return built


def run_probe(exe: Path, switch: str | None, *flags: bool) -> bool:
    env = {k: v for k, v in os.environ.items() if k != SWITCH}
    if switch is not None:
        env[SWITCH] = switch
    result = subprocess.run(
        [str(exe), *(str(int(f)) for f in flags)], capture_output=True, text=True, env=env, check=True
    )
    return result.stdout.strip() == "1"


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("switch", [None, "", "0"])
def test_the_base_dgrad_runs_by_default(probes, site, switch):
    for flags in itertools.product([False, True], repeat=4):
        assert not run_probe(probes[site], switch, *flags), (
            f"{site}: base dgrad skipped for lora/lora_only/prequant/gpt_oss={flags} with {SWITCH}={switch!r}"
        )


@pytest.mark.parametrize("site", sorted(SITES))
@pytest.mark.parametrize("switch", ["1", "yes"])
def test_the_switch_restores_upstreams_skip_and_nothing_else(probes, site, switch):
    for flags in itertools.product([False, True], repeat=4):
        assert run_probe(probes[site], switch, *flags) == all(flags), (
            f"{site}: lora/lora_only/prequant/gpt_oss={flags} with {SWITCH}={switch!r}"
        )


@pytest.mark.parametrize("site", sorted(SITES))
def test_the_flag_only_selects_the_zero_fill_branch(site):
    # The value the probe evaluates must be what decides between zeroing d_input and the base dgrad GEMM.
    path, signature = SITES[site]
    body = " ".join(function_body(path, signature).split())
    assert body.count("skip_base_backward") == 2, f"{path.name}: skip_base_backward used outside the branch"
    gemm = "moe_grouped_gemm_gate_up_backward(" if site == "gate_up" else "moe_grouped_gemm_down_backward("
    branch = re.search(
        r"if \(skip_base_backward \|\| \(weight_is_compact && compact\.active_experts\.empty\(\)\)\) "
        r"\{ zero_d_input\(\); \} else if \([^)]*DType == ETensorDType::BF16\) \{ " + re.escape(gemm),
        body,
    )
    assert branch, f"{path.name}: the skip no longer guards zero_d_input() against the base dgrad"
    source = path.read_text(encoding="utf-8")
    assert '#include "surogate_fork/gpt_oss_prequant_dgrad.h"' in source


# --- CPU emulation of one GPT-OSS expert, with the branch the probe picks -------------------------------------------


def gpt_oss_act(gate_up, alpha=1.702, limit=7.0):
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1) * gate * (gate * alpha).sigmoid()


def emulated_backward(x, dy, w_gu, b_gu, a_gu, bb_gu, w_d, b_d, a_d, bb_d, s, skip):
    """The gradients surogate's dispatch computes: base dgrad (unless skipped) plus the LoRA terms, per op."""
    import torch

    gate_up = x @ w_gu.T + s * (x @ a_gu.T) @ bb_gu.T + b_gu
    gate_up = gate_up.detach().requires_grad_(True)
    act = gpt_oss_act(gate_up)
    # down backward: d_act = [base dgrad] + LoRA dX; dA, dB of the down LoRA from (act, dy)
    h_d = act.detach() @ a_d.T
    d_bb_d = s * dy.T @ h_d
    d_a_d = s * (dy @ bb_d).T @ act.detach()
    d_act = s * (dy @ bb_d) @ a_d
    if not skip:
        d_act = d_act + dy @ w_d
    # activation backward
    (d_gate_up,) = torch.autograd.grad(act, gate_up, d_act)
    # gate_up backward: d_x = [base dgrad] + LoRA dX; dA, dB of the gate_up LoRA from (x, d_gate_up)
    h_gu = x @ a_gu.T
    d_bb_gu = s * d_gate_up.T @ h_gu
    d_a_gu = s * (d_gate_up @ bb_gu).T @ x
    d_x = s * (d_gate_up @ bb_gu) @ a_gu
    if not skip:
        d_x = d_x + d_gate_up @ w_gu
    return {"x": d_x, "gate_up.A": d_a_gu, "gate_up.B": d_bb_gu, "down.A": d_a_d, "down.B": d_bb_d}


def autograd_reference(x, dy, w_gu, b_gu, a_gu, bb_gu, w_d, b_d, a_d, bb_d, s):
    import torch

    leaves = [t.detach().clone().requires_grad_(True) for t in (x, a_gu, bb_gu, a_d, bb_d)]
    x_, a_gu_, bb_gu_, a_d_, bb_d_ = leaves
    gate_up = x_ @ w_gu.T + s * (x_ @ a_gu_.T) @ bb_gu_.T + b_gu
    act = gpt_oss_act(gate_up)
    y = act @ w_d.T + s * (act @ a_d_.T) @ bb_d_.T + b_d
    grads = torch.autograd.grad(y, leaves, dy)
    return dict(zip(["x", "gate_up.A", "gate_up.B", "down.A", "down.B"], grads))


@pytest.mark.parametrize("b_scale", [0.0, 2e-2])
def test_the_default_branch_gives_the_exact_lora_gradients(probes, b_scale):
    torch = pytest.importorskip("torch")
    # The case the skip was written for: LoRA, LoRA-only mode, prequantized checkpoint, GPT-OSS; switch unset.
    skip = {site: run_probe(probes[site], None, True, True, True, True) for site in SITES}
    assert skip["gate_up"] == skip["down"]
    gen = torch.Generator().manual_seed(0)
    T, C, M, r, s = 64, 96, 48, 8, 2.0

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, dtype=torch.float64) * scale

    x, dy = rnd(T, C), rnd(T, C)
    w_gu, b_gu, w_d, b_d = rnd(2 * M, C, scale=C**-0.5), rnd(2 * M, scale=0.1), rnd(C, M, scale=M**-0.5), rnd(C, scale=0.1)
    a_gu, a_d = rnd(r, C, scale=C**-0.5), rnd(r, M, scale=M**-0.5)
    bb_gu, bb_d = rnd(2 * M, r, scale=b_scale), rnd(C, r, scale=b_scale)  # B = 0 is LoRA's init
    args = (x, dy, w_gu, b_gu, a_gu, bb_gu, w_d, b_d, a_d, bb_d, s)
    got = emulated_backward(*args, skip=skip["down"])
    want = autograd_reference(*args)
    assert want["gate_up.B"].norm() > 0 and want["x"].norm() > 0
    errors = {}
    for name in want:
        if want[name].norm() == 0:  # dA while B is zero
            errors[name] = float(got[name].norm())
        else:
            errors[name] = float((got[name] - want[name]).norm() / want[name].norm())
    bad = {name: f"{err:.3g}" for name, err in errors.items() if not err < 1e-10}
    assert not bad, f"relative gradient errors with skip={skip['down']}, |B|~{b_scale}: {bad}"
