"""Ledger topic T16 (GQA shapes the fork registers in the serve engine), runnable without CUDA.

csrc/src/serve/ops/kernel/gqa_attention_geometry.cuh registers Gqa128_32q8, Gqa128_40q8 and Gqa128_32q32 for the
fork; fork/native/serve holds their small-T decode units and fork/native/include/surogate_fork/gqa_fork_shapes.h the
switch, SUROGATE_SERVE_GQA_FORK_SHAPES=1. The routing decisions are made by three upstream functions the topic
edits: optimized_decode_shape and generic_takes (wrapper/gqa_attention.cpp) and dispatch_prompt_geometry
(launcher/gqa_attention_prefill.cu). The probe below compiles those functions' own text, cut out of the sources,
against the real registry header and the fork header, with host stubs for the dtype enum and the block mask, and
runs it with the switch unset, off and on. So these tests check the routing production compiles, not a copy of it:
the registry carries the shapes, the switch off leaves every shape on the route it took before (the new ones on the
generic kernel and the KV geometry's prompt dispatch), on sends a BF16 or e4m3 cache to the registered kernels and
keeps int8 on the generic kernel, and no other shape moves. The GPU half (kernels, numerics, speed) is
sinfer_gqa_fork_shapes_test (fork/native/tests/test-fork-gqa-shapes.cpp), upstream's sinfer_gqa_attention_test run
with the switch on as well, and the bench in fork/docs/performance.md; the last tests here pin their wiring.
"""

import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
SERVE = REPO / "csrc/src/serve"
GEOMETRY = SERVE / "ops/kernel/gqa_attention_geometry.cuh"
KV_QUANT = SERVE / "ops/kernel/gqa_attention_kv_quant.cuh"
WRAPPER = SERVE / "ops/wrapper/gqa_attention.cpp"
GENERIC = SERVE / "ops/launcher/gqa_attention_generic.cu"
PREFILL = SERVE / "ops/launcher/gqa_attention_prefill.cu"
LAUNCHER = SERVE / "ops/launcher"
FORK_INCLUDE = REPO / "fork/native/include"
FORK_SERVE = REPO / "fork/native/serve"
FORK_CMAKE = REPO / "fork/native/cmake/fork_native.cmake"
CSRC_CMAKE = REPO / "csrc/CMakeLists.txt"
LEDGER = REPO / "fork/patches/ledger.toml"
PERFORMANCE = REPO / "fork/docs/performance.md"
SWITCH = "SUROGATE_SERVE_GQA_FORK_SHAPES"

FORK_SHAPES = {"Gqa128_32q8": (128, 32, 8), "Gqa128_40q8": (128, 40, 8), "Gqa128_32q32": (128, 32, 32)}
DTYPES = ("BF16", "FP8_E4M3FN", "I8")
#: Shapes the probe routes: the fork's, upstream's registered ones that share a width or a KV count with them (and
#: the one int8 exception upstream makes), and unregistered query groups over registered KV counts.
UPSTREAM_REGISTERED = [(128, 16, 8), (128, 32, 4), (128, 32, 16), (128, 16, 2), (64, 32, 8), (256, 8, 4), (256, 24, 4)]
UNREGISTERED = [(128, 28, 4), (128, 64, 8), (128, 24, 8), (256, 12, 4), (64, 16, 8)]


def definition(path: Path, signature: str) -> str:
    """The text of the first definition matching `signature` (a regex), through its closing brace."""
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth = 1
    for end in range(match.end(), len(text)):
        depth += {"{": 1, "}": -1}.get(text[end], 0)
        if depth == 0:
            return text[match.start() : end + 1]
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def registry_names() -> list[str]:
    macro = re.search(r"#define SINFER_GQA_FOR_EACH_GEOMETRY\(X\)(.*?)\n\n", GEOMETRY.read_text(encoding="utf-8"), re.S)
    assert macro, "gqa_attention_geometry.cuh: no SINFER_GQA_FOR_EACH_GEOMETRY"
    return re.findall(r"X\((\w+)\)", macro.group(1))


def probe_source() -> str:
    quant = re.search(r"inline constexpr int kGqaKvQuantGroup = (\d+);", KV_QUANT.read_text(encoding="utf-8"))
    assert quant, "gqa_attention_kv_quant.cuh: no kGqaKvQuantGroup"
    serves = definition(GENERIC, r"bool gqa_generic_attention_serves\(")
    dispatch = definition(PREFILL, r"template <typename Visitor>\s*void dispatch_prompt_geometry\(")
    optimized = definition(WRAPPER, r"bool optimized_decode_shape\(")
    generic = definition(WRAPPER, r"bool generic_takes\(")
    shapes = UPSTREAM_REGISTERED + UNREGISTERED + list(FORK_SHAPES.values())
    rows = ", ".join("{%d, %d, %d}" % shape for shape in shapes)
    return f"""
// Host stand-ins for the CUDA qualifiers the registry header puts on one device helper.
#define __device__
#define __forceinline__ inline
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "surogate_fork/gqa_fork_shapes.h"
#include <cstdio>

namespace sinfer {{
enum class DType {{ BF16, FP8_E4M3FN, I8 }};
}}
namespace sinfer::ops {{
struct GqaBlockMask {{
    const std::uint32_t* words = nullptr;
    std::int32_t image_begin = 0;
    std::int32_t image_end = 0;
}};
inline constexpr int kGqaKvQuantGroup = {quant.group(1)};
namespace detail {{
{serves}
{dispatch}
}} // namespace detail
{optimized}
{generic}

int run() {{
#define SINFER_PROBE_REGISTERED(Name) \\
    std::printf("registry %s %d %d %d\\n", #Name, Name::HeadDim, Name::QHeads, Name::KVHeads);
    SINFER_GQA_FOR_EACH_GEOMETRY(SINFER_PROBE_REGISTERED)
#undef SINFER_PROBE_REGISTERED
    for (const auto& shape : surogate_fork::kGqaForkShapes) {{
        std::printf("fork %d %d %d\\n", shape.head_dim, shape.q_heads, shape.kv_heads);
    }}
    // The append kernels dispatch on (head dim, KV heads) and take the first registered match.
    gqa_dispatch_kv_geometry(128, 8, []<typename G>() {{ std::printf("append 128 8 %d\\n", G::QHeads); }});
    std::printf("enabled %d\\n", surogate_fork::gqa_fork_shapes_enabled() ? 1 : 0);
    const int shapes[][3] = {{{rows}}};
    const DType dtypes[] = {{DType::BF16, DType::FP8_E4M3FN, DType::I8}};
    const char* names[] = {{"BF16", "FP8_E4M3FN", "I8"}};
    for (const auto& s : shapes) {{
        for (int d = 0; d < 3; ++d) {{
            int prompt[3] = {{0, 0, 0}};
            detail::dispatch_prompt_geometry(s[0], s[1], s[2], dtypes[d], [&]<typename G>() {{
                prompt[0] = G::HeadDim;
                prompt[1] = G::QHeads;
                prompt[2] = G::KVHeads;
            }});
            std::printf("route %d %d %d %s registered=%d optimized=%d generic=%d prompt=%d,%d,%d\\n", s[0], s[1],
                        s[2], names[d], gqa_shape_is_registered(s[0], s[1], s[2]) ? 1 : 0,
                        optimized_decode_shape(s[0], s[1], s[2], dtypes[d]) ? 1 : 0,
                        generic_takes(s[0], s[1], s[2], dtypes[d], {{}}) ? 1 : 0, prompt[0], prompt[1], prompt[2]);
        }}
    }}
    return 0;
}}
}} // namespace sinfer::ops

int main() {{ return sinfer::ops::run(); }}
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("no C++ compiler (g++, c++ or clang++) to build the routing probe")
    work = tmp_path_factory.mktemp("gqa_fork_shapes")
    source, binary = work / "probe.cpp", work / "probe"
    source.write_text(probe_source(), encoding="utf-8")
    build = subprocess.run(
        [compiler, "-std=c++20", "-O0", f"-I{SERVE}", f"-I{FORK_INCLUDE}", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=110,
    )
    assert build.returncode == 0, f"routing probe does not compile:\n{build.stderr[-4000:]}"
    return binary


def run_probe(binary: Path, switch: str | None) -> list[str]:
    env = {k: v for k, v in os.environ.items() if k != SWITCH}
    if switch is not None:
        env[SWITCH] = switch
    result = subprocess.run([str(binary)], capture_output=True, text=True, env=env, timeout=30, check=True)
    return result.stdout.splitlines()


def routes(lines: list[str]) -> dict[tuple[int, int, int, str], dict[str, object]]:
    table = {}
    for line in lines:
        if not line.startswith("route "):
            continue
        _, dim, q, kv, dtype, *fields = line.split()
        values = dict(field.split("=") for field in fields)
        table[(int(dim), int(q), int(kv), dtype)] = {
            "registered": values["registered"] == "1",
            "optimized": values["optimized"] == "1",
            "generic": values["generic"] == "1",
            "prompt": tuple(int(x) for x in values["prompt"].split(",")),
        }
    return table


def test_the_registry_carries_the_fork_shapes_once_each(probe):
    lines = run_probe(probe, None)
    registry = {name: tuple(map(int, rest)) for _, name, *rest in (line.split() for line in lines if line.startswith("registry "))}
    for name, shape in FORK_SHAPES.items():
        assert registry.get(name) == shape, f"{name} is not registered as {shape}"
    assert len(registry) == len(registry_names()), "a registry name appears twice"
    fork = {tuple(map(int, line.split()[1:])) for line in lines if line.startswith("fork ")}
    assert fork == set(FORK_SHAPES.values()), "surogate_fork::kGqaForkShapes and the fork's registry lines differ"
    # Gqa128_16q8 stays the first (128, 8) entry, so the KV-cache append kernels keep their instantiation.
    assert "append 128 8 16" in lines


@pytest.mark.parametrize("switch", [None, "0", "", "true", "on"])
def test_with_the_switch_off_the_fork_shapes_keep_their_old_routes(probe, switch):
    lines = run_probe(probe, switch)
    assert "enabled 0" in lines
    table = routes(lines)
    for dim, q, kv in FORK_SHAPES.values():
        for dtype in DTYPES:
            row = table[(dim, q, kv, dtype)]
            assert row["registered"]
            # As an unregistered query group: no small-T kernel, the generic split-KV kernel for decode and narrow
            # prompts, and wide prompts on the prompt kernel of the first shape registered for the KV count.
            assert not row["optimized"] and row["generic"], (dim, q, kv, dtype, row)
            assert row["prompt"] == ((128, 16, 8) if kv == 8 else (128, 32, 32)), (dim, q, kv, dtype, row)


def test_with_the_switch_on_bf16_and_e4m3_take_the_registered_route_and_int8_does_not(probe):
    lines = run_probe(probe, "1")
    assert "enabled 1" in lines
    table = routes(lines)
    for dim, q, kv in FORK_SHAPES.values():
        for dtype in ("BF16", "FP8_E4M3FN"):
            row = table[(dim, q, kv, dtype)]
            assert row["optimized"] and not row["generic"], (dim, q, kv, dtype, row)
            assert row["prompt"] == (dim, q, kv), (dim, q, kv, dtype, row)
        # The int8 small-T decode kernel serves only 256-wide heads (kGqaI8DecodeRegistered) and would throw.
        row = table[(dim, q, kv, "I8")]
        assert not row["optimized"] and row["generic"], (dim, q, kv, "I8", row)
        assert row["prompt"] == ((128, 16, 8) if kv == 8 else (128, 32, 32)), (dim, q, kv, "I8", row)


def test_the_switch_moves_no_other_shape(probe):
    off, on = routes(run_probe(probe, None)), routes(run_probe(probe, "1"))
    others = UPSTREAM_REGISTERED + UNREGISTERED
    for dim, q, kv in others:
        for dtype in DTYPES:
            assert off[(dim, q, kv, dtype)] == on[(dim, q, kv, dtype)], (dim, q, kv, dtype)
    for dim, q, kv in UPSTREAM_REGISTERED:
        for dtype in DTYPES:
            row = off[(dim, q, kv, dtype)]
            int8_prompt_kernel = (dim, q, kv, dtype) == (256, 8, 4, "I8")  # upstream's own exception
            assert row["registered"] and row["optimized"] is not int8_prompt_kernel and not row["generic"]
            assert row["prompt"] == (dim, q, kv)
    for dim, q, kv in UNREGISTERED:
        for dtype in DTYPES:
            row = off[(dim, q, kv, dtype)]
            assert not row["registered"] and not row["optimized"] and row["generic"], (dim, q, kv, dtype)


def decode_units() -> dict[str, Path]:
    """Every gqa_attention_decode_<shape>.cu a CMake source list names, keyed by the file name."""
    csrc = CSRC_CMAKE.read_text(encoding="utf-8")
    listed = {m: LAUNCHER / m for m in re.findall(r"\$\{SERVE_ROOT\}/ops/launcher/(gqa_attention_decode_gqa\w+\.cu)", csrc)}
    hook = re.search(r"function\(_surogate_fork_attach_serve\)(.*?)endfunction\(\)", FORK_CMAKE.read_text(encoding="utf-8"), re.S)
    assert hook, "fork_native.cmake: no _surogate_fork_attach_serve()"
    assert "if(TARGET sinfer_ops)" in hook.group(1)
    assert 'target_include_directories(sinfer_ops PRIVATE "${SUROGATE_FORK_NATIVE_DIR}/include")' in hook.group(1)
    fork = re.findall(r'"\$\{SUROGATE_FORK_NATIVE_DIR\}/serve/(gqa_attention_decode_gqa\w+\.cu)"', hook.group(1))
    assert sorted(fork) == sorted(p.name for p in FORK_SERVE.glob("*.cu")), "a fork serve unit is not attached"
    assert not set(fork) & set(listed), "a decode unit is both upstream's and the fork's"
    listed.update({m: FORK_SERVE / m for m in fork})
    deferred = 'cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL _surogate_fork_attach_serve)'
    assert deferred in FORK_CMAKE.read_text(encoding="utf-8")
    return listed


def test_every_registered_shape_has_one_compiled_decode_unit():
    """The link-time rule of the registry, checked without a build: each X() line has its decode unit in a source list."""
    units = decode_units()
    names = registry_names()
    for name in names:
        unit = units.get(f"gqa_attention_decode_{name.lower()}.cu")
        assert unit and unit.is_file(), f"{name}: no decode unit in a CMake source list"
        text = unit.read_text(encoding="utf-8")
        for mode in ("GqaAppendInput", "GqaCachedInput"):
            assert f"template void gqa_attention_small_t_launch_for<{name}, {mode}>(" in text, (unit, mode)
    assert len(units) == len(names), "a decode unit is listed for a shape the registry does not carry"
    for name in FORK_SHAPES:
        unit = units[f"gqa_attention_decode_{name.lower()}.cu"]
        assert unit.parent == FORK_SERVE, f"{name}: its unit belongs under fork/native/serve"
        assert f"surogate_fork::gqa_is_fork_shape({name}::HeadDim" in unit.read_text(encoding="utf-8")


def test_the_three_routing_points_read_the_fork_switch():
    for path in (WRAPPER, PREFILL):
        assert '#include "surogate_fork/gqa_fork_shapes.h"' in path.read_text(encoding="utf-8"), path
    optimized = " ".join(definition(WRAPPER, r"bool optimized_decode_shape\(").split())
    assert "!surogate_fork::gqa_fork_shape_keeps_generic_route(dim, queries, kv, dtype == DType::I8)" in optimized
    generic = " ".join(definition(WRAPPER, r"bool generic_takes\(").split())
    assert (
        "(!gqa_shape_is_registered(head_dim, q_heads, kv_heads) || "
        "surogate_fork::gqa_fork_shape_keeps_generic_route(head_dim, q_heads, kv_heads, cache_dtype == DType::I8))"
    ) in generic
    prefill = PREFILL.read_text(encoding="utf-8")
    calls = re.findall(r"(?<!void )dispatch_prompt_geometry\((.*?)\[&\]", prefill, re.S)
    assert len(calls) == 3 and all(" ".join(c.split()) == "q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype," for c in calls), calls
    # Every other upstream use of the registry predicate answers the same for a fork shape either way.
    wrapper = WRAPPER.read_text(encoding="utf-8")
    assert wrapper.count("gqa_shape_is_registered(") == 2, "the wrapper gained a registry check: route it too"
    assert prefill.count("gqa_shape_is_registered(") == 1, "the prompt launcher gained a registry check: route it too"


def test_ctest_runs_the_gpu_cases_with_the_switch_off_and_on():
    hook = re.search(r"function\(_surogate_fork_attach_serve\)(.*?)endfunction\(\)", FORK_CMAKE.read_text(encoding="utf-8"), re.S)
    assert hook, "fork_native.cmake: no _surogate_fork_attach_serve()"
    cmake = " ".join(hook.group(1).split())
    for test, binary, value in (
        ("sinfer_gqa_fork_shapes_test", "sinfer_gqa_fork_shapes_test", "0"),
        ("sinfer_gqa_fork_shapes_on_test", "sinfer_gqa_fork_shapes_test", "1"),
        ("sinfer_gqa_attention_fork_shapes_on_test", "sinfer_gqa_attention_test", "1"),
    ):
        assert f"add_test(NAME {test} COMMAND {binary})" in cmake, test
        props = re.findall(r"set_tests_properties\(([^)]*?) PROPERTIES ENVIRONMENT \"(\w+)=(\d)\"\)", cmake)
        assert any(test in names.split() and name == SWITCH and v == value for names, name, v in props), test
    assert "add_dependencies(serve-tests sinfer_gqa_fork_shapes_test)" in cmake
    gpu = (REPO / "fork/native/tests/test-fork-gqa-shapes.cpp").read_text(encoding="utf-8")
    assert '#define main sinfer_gqa_attention_test_main\n#include "ops/test_gqa_attention.cpp"' in gpu
    for dim, q, kv in FORK_SHAPES.values():
        assert re.search(rf'run_fork_shape\(Geometry\{{"\w+", {q}, {kv}, {dim}\}}', gpu), (dim, q, kv)


def test_the_switch_and_its_gpu_checks_are_documented():
    ledger = tomllib.loads(LEDGER.read_text(encoding="utf-8"))
    topics = [t for t in ledger["topic"] if SWITCH in t.get("switch", "")]
    assert len(topics) == 1, "no ledger topic names the switch"
    topic = topics[0]
    for path in ("csrc/src/serve/ops/kernel/gqa_attention_geometry.cuh", "csrc/src/serve/ops/wrapper/gqa_attention.cpp",
                 "csrc/src/serve/ops/launcher/gqa_attention_prefill.cu"):
        assert path in topic["files"], path
    for path in FORK_SERVE.glob("*.cu"):
        assert str(path.relative_to(REPO)) in topic["fork_files"], path
    for test in ("fork/patches/tests/native/test_gqa_fork_shapes.py", "fork/native/tests/test-fork-gqa-shapes.cpp"):
        assert test in topic["tests"] and test in topic["fork_files"], test
    doc = PERFORMANCE.read_text(encoding="utf-8")
    for needle in (f"{SWITCH}=1", "sinfer_gqa_attention_test", "sinfer_gqa_fork_shapes_test",
                   "sinfer_causal_softmax_attention_bench", "--geometry d128-h32-kv8"):
        assert needle in doc, needle
