"""Ledger topic T23, second part: a 64-bit grid-stride index in the FP8 abs_max / quantize kernels (no CUDA needed).

quant.cu's reduce_abs_max_kernel and quantize_with_abs_max_kernel walk their `long N` elements with a grid-stride loop
whose index was an int: `for (int i = vec * (blockIdx.x * blockDim.x + threadIdx.x); i < N; i += vec * blockDim.x *
gridDim.x)`. nvcc advances it with add.s32 and sign-extends it for the 64-bit compare, so a thread whose next index
passes 2^31 - 1 wraps to about -2^31 and keeps going: for N > 2^31 the loop never ends (every int is < N); below 2^31
it ends, after one or more passes that read (and, in the quantizer, write) up to 2^31 elements below the buffers.
That happens once N >= 2^31 - S + vec, S = vec x the launch's thread count (RTX 5090, BF16: S = 2,088,960, so from
14,121 rows at V = 151936). The fused LM-head loss runs both kernels on the whole M x V dlogits chunk when recipe
fp8_hybrid has primed the transposed FP8 lm_head cache (lmhead_chunks <= 1; lm_head_dx_matmul in
fused_lm_head_loss.cpp).

The probe compiles the kernels, fp8_scale_from_amax, the launchers' block and grid formulas from quant.cu and the FP8
constants from utils.h as host C++ with -fwrapv, and runs one thread of a real launch shape. Both buffers are PROT_NONE
reservations of N elements with another 2^31 elements below them; only the 16-byte vectors that thread should touch
(every S-th one from its start, 64-bit) are opened, so any other access faults and the probe reports the offset. The
loads and stores are recorded and compared with that list, the abs-max with the inputs, the stored codes with the
inputs. The FP8 conversion is a stub (the stored byte is the integer input, scale 1): the probe checks which elements
are read and written, not the rounding. On upstream's quant.cu (488e5f00) every case past the threshold faults; with
the fix they pass, and the last shape below the threshold passes on both. Not covered: the INT8 and the FP32-to-BF16
variants (same loop, widened by the same change; their bodies are left as upstream wrote them). GPU behaviour is
unverified here; the errata section gives the 5090 check.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
QUANT = REPO / "csrc/src/recipes/fp8_hybrid/kernels/quant.cu"
UTILS_H = REPO / "csrc/src/utilities/utils.h"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the probe reserves address space with mmap"
)


def definition(path: Path, signature: str) -> str:
    """The first definition matching `signature` (a regex) through its closing brace; comments and literals skipped."""
    text = path.read_text(encoding="utf-8")
    match = re.search(signature + r"[^;{]*\{", text)
    assert match, f"{path.relative_to(REPO)}: no definition matching {signature!r}"
    depth, i = 1, match.end()
    while i < len(text):
        two = text[i : i + 2]
        if two == "//":
            i = text.index("\n", i)
            continue
        if two == "/*":
            i = text.index("*/", i) + 2
            continue
        c = text[i]
        if c in "\"'":
            j = i + 1
            while text[j] != c:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth == 0:
            return text[match.start() : i + 1]
        i += 1
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def matching(path: Path, pattern: str, text: str | None = None) -> str:
    text = path.read_text(encoding="utf-8") if text is None else text
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"{path.relative_to(REPO)}: nothing matching {pattern!r}"
    return match.group(0)


BLOCK_SIZE = r"^\s*int block_size = dp\.maxThreadsPerMultiProcessor == 2048 \? 1024 : 768;$"
N_BLOCKS = r"^\s*int n_blocks = dp\.maxThreadsPerMultiProcessor / block_size \* dp\.multiProcessorCount;$"
REDUCE_LAUNCHER = r"template <class floatX>\s*void reduce_abs_max_launcher\("
QUANTIZE_LAUNCHER = r"template <class floatY, class floatX>\s*void quantize_with_abs_max_launcher\("
REDUCE_KERNEL = r"template <class floatX>\s*__global__ void reduce_abs_max_kernel\("
FP8_KERNEL = (
    r"template <\s*typename FloatOut,\s*typename FloatIn,\s*typename _ =\s*std::enable_if_t<[^;{]*>>\s*"
    r"__global__ void quantize_with_abs_max_kernel\("
)

PRELUDE = r"""
#include <math.h>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

// ---- CUDA host stubs: one thread of a launch ---------------------------------------------------------------------
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
struct probe_dim3 { unsigned int x, y, z; };
static probe_dim3 blockIdx{0, 0, 0}, threadIdx{0, 0, 0}, blockDim{1, 1, 1}, gridDim{1, 1, 1};
static inline void __syncthreads() {}
struct cudaDeviceProp { int maxThreadsPerMultiProcessor, multiProcessorCount; };

struct nv_bfloat16 {  // round to nearest even, as __float2bfloat16 (the probe has no NaN)
    std::uint16_t bits;
    nv_bfloat16() = default;
    nv_bfloat16(float value) {
        std::uint32_t u;
        std::memcpy(&u, &value, 4);
        u += 0x7FFFu + ((u >> 16) & 1u);
        bits = static_cast<std::uint16_t>(u >> 16);
    }
    operator float() const {
        std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
        float value;
        std::memcpy(&value, &u, 4);
        return value;
    }
};
struct __nv_fp8_e4m3 { unsigned char __x; };
struct __nv_fp8_e5m2 { unsigned char __x; };
enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 };
enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE };
// Stub conversion: the probe's inputs are integers in [-100, 100] and its scale is 1, so the byte is the input.
static inline unsigned char __nv_cvt_float_to_fp8(float x, __nv_saturation_t, __nv_fp8_interpretation_t) {
    return static_cast<unsigned char>(static_cast<int>(lrintf(x)) & 0xFF);
}

// Every load and store goes through GenericVector; record the element offsets from the buffer bases.
static const char* g_in = nullptr;
static char* g_out = nullptr;
static long long g_in_elem = 1, g_out_elem = 1;
static std::vector<long long> g_loads, g_stores;
static std::size_t g_cap = 0;
static void probe_record(std::vector<long long>& log, long long offset) {
    if (log.size() >= g_cap) {
        std::printf("RUNAWAY %lld\n", offset);
        std::fflush(stdout);
        _exit(7);
    }
    log.push_back(offset);
}

template <class T, std::size_t N> struct GenericVector {  // vec.cuh: element access, size, load, store
    T values[N];
    static constexpr const std::size_t size = N;
    T& operator[](int index) { return values[index]; }
    const T& operator[](int index) const { return values[index]; }
    static GenericVector load(const T* address) {
        probe_record(g_loads, (reinterpret_cast<const char*>(address) - g_in) / g_in_elem);
        GenericVector result;
        std::memcpy(result.values, address, sizeof result.values);
        return result;
    }
    void store(T* address) {
        probe_record(g_stores, (reinterpret_cast<char*>(address) - g_out) / g_out_elem);
        std::memcpy(address, values, sizeof values);
    }
};

// kernel_utils.cuh's warp/block/atomic reduction, for one thread
static inline void handle_absmax_reduction(float* result, float* block_max, float thread_max) {
    if (result) {
        *block_max = fmaxf(*block_max, thread_max);
        *result = fmaxf(*result, *block_max);
    }
}
"""

PROBE_MAIN = r"""
// ---- probe ------------------------------------------------------------------------------------------------------
static void launch_shape(int threads_per_sm, int sms, int* block, int* grid) {
    cudaDeviceProp dp{threads_per_sm, sms};
    BLOCK_SIZE_LINE
    N_BLOCKS_LINE
    *block = block_size;
    *grid = n_blocks;
}

static void on_fault(int, siginfo_t* info, void*) {
    char text[96];
    const char* address = static_cast<const char*>(info->si_addr);
    long long in_off = (address - g_in) / g_in_elem, out_off = (address - g_out) / g_out_elem;
    int n = std::snprintf(text, sizeof text, "FAULT %lld %lld\n", in_off, out_off);
    if (write(1, text, n) < 0) {}
    _exit(3);
}

static long long input_value(long long e) {  // integers in [-100, 100], exact in BF16 and FP32
    return static_cast<long long>((static_cast<unsigned long long>(e) * 2654435761ULL >> 7) % 201ULL) - 100;
}

static char* reserve(long long bytes, long long below) {
    void* r = mmap(nullptr, below + bytes + 65536, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (r == MAP_FAILED) return nullptr;
    return static_cast<char*>(r) + below;
}

static bool open_range(char* base, long long lo, long long hi) {
    const long long page = sysconf(_SC_PAGESIZE);
    lo = lo / page * page;
    hi = (hi + page - 1) / page * page;
    return mprotect(base + lo, hi - lo, PROT_READ | PROT_WRITE) == 0;
}

template <class TIn, class TOut>
static int run(bool quantize, long long N, int threads_per_sm, int sms, long long thread, const char* out_path) {
    int block = 0, grid = 0;
    launch_shape(threads_per_sm, sms, &block, &grid);
    if (thread < 0 || thread >= static_cast<long long>(block) * grid) return 2;
    const long long vec = 16 / sizeof(TIn);
    const long long stride = vec * block * grid;
    g_in_elem = sizeof(TIn);
    g_out_elem = sizeof(TOut);
    // a wrapped int index is >= -2^31: keep 2^31 elements below each base inside the reservation
    char* in = reserve(N * g_in_elem, (1LL << 31) * g_in_elem + 65536);
    char* out = reserve(N * g_out_elem, (1LL << 31) * g_out_elem + 65536);
    if (!in || !out) {
        std::printf("NOMAP\n");
        return 4;
    }
    g_in = in;
    g_out = out;
    std::vector<long long> expected;
    for (long long e = vec * thread; e < N; e += stride) {
        expected.push_back(e);
        if (!open_range(in, e * g_in_elem, (e + vec) * g_in_elem) ||
            (quantize && !open_range(out, e * g_out_elem, (e + vec) * g_out_elem))) {
            std::printf("NOMAP\n");
            return 4;
        }
        for (long long k = 0; k < vec; ++k) {
            reinterpret_cast<TIn*>(in)[e + k] = static_cast<TIn>(static_cast<float>(input_value(e + k)));
        }
    }
    g_cap = expected.size() + 1;
    g_loads.reserve(g_cap);
    g_stores.reserve(g_cap);

    struct sigaction action {};
    action.sa_sigaction = on_fault;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);

    blockDim.x = static_cast<unsigned>(block);
    gridDim.x = static_cast<unsigned>(grid);
    blockIdx.x = static_cast<unsigned>(thread / block);
    threadIdx.x = static_cast<unsigned>(thread % block);
    float result = 0.0f;
    float scale = -1.0f;
    if (quantize) {
        const float abs_max = FP8_MAX_OF(TOut);  // scale 1: the stub stores the integer input
        quantize_with_abs_max_kernel<TOut, TIn>(reinterpret_cast<TOut*>(out), &scale,
                                                reinterpret_cast<const TIn*>(in), &abs_max, N, false);
    } else {
        reduce_abs_max_kernel<TIn>(&result, reinterpret_cast<const TIn*>(in), N);
    }

    FILE* f = std::fopen(out_path, "w");
    if (!f) return 5;
    std::fprintf(f, "launch %d %d\nabsmax %.9g\nscale %.9g\n", block, grid, result, scale);
    for (long long e : g_loads) std::fprintf(f, "load %lld\n", e);
    for (long long e : g_stores) {
        std::fprintf(f, "store %lld", e);
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(out);
        for (long long k = 0; k < vec; ++k) std::fprintf(f, " %d", bytes[(e + k) * g_out_elem]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::printf("OK\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 8) return 2;
    const char* kernel = argv[1];
    const char* types = argv[2];
    const long long N = std::atoll(argv[3]);
    const int threads_per_sm = std::atoi(argv[4]), sms = std::atoi(argv[5]);
    const long long thread = std::atoll(argv[6]);
    const char* out_path = argv[7];
    const bool quantize = !std::strcmp(kernel, "quantize");
    if (!quantize && std::strcmp(kernel, "absmax")) return 2;
    if (!std::strcmp(types, "e5m2-bf16")) {
        return run<nv_bfloat16, __nv_fp8_e5m2>(quantize, N, threads_per_sm, sms, thread, out_path);
    }
    if (!std::strcmp(types, "e4m3-bf16")) {
        return run<nv_bfloat16, __nv_fp8_e4m3>(quantize, N, threads_per_sm, sms, thread, out_path);
    }
    if (!std::strcmp(types, "e4m3-f32")) {
        return run<float, __nv_fp8_e4m3>(quantize, N, threads_per_sm, sms, thread, out_path);
    }
    return 2;
}
"""


def probe_source() -> str:
    reduce_launcher = definition(QUANT, REDUCE_LAUNCHER)
    quantize_launcher = definition(QUANT, QUANTIZE_LAUNCHER)
    # the probe takes the launch shape from these two lines; both launchers must use them
    block_size = matching(QUANT, BLOCK_SIZE, reduce_launcher).strip()
    n_blocks = matching(QUANT, N_BLOCKS, reduce_launcher).strip()
    assert block_size == matching(QUANT, BLOCK_SIZE, quantize_launcher).strip()
    assert n_blocks == matching(QUANT, N_BLOCKS, quantize_launcher).strip()
    parts = [
        PRELUDE,
        "// ---- utils.h ----",
        matching(
            UTILS_H,
            r"^template <typename FP8Type>\ninline __nv_fp8_interpretation_t fp8_interpretation_v;\n"
            r"(?:template <>\ninline constexpr __nv_fp8_interpretation_t fp8_interpretation_v<\w+> = [\w:]+;\n)+",
        ),
        matching(
            UTILS_H,
            r"^template <typename FP8Type>\ninline constexpr float fp8_max_v = 0\.f;\n"
            r"(?:template <>\ninline constexpr float fp8_max_v<\w+> = [\d.]+f;\n)+",
        ),
        "#define FP8_MAX_OF(T) fp8_max_v<T>",
        "// ---- quant.cu ----",
        definition(QUANT, r"__device__ __forceinline__ float fp8_scale_from_amax\("),
        definition(QUANT, REDUCE_KERNEL),
        definition(QUANT, FP8_KERNEL),
        PROBE_MAIN.replace("BLOCK_SIZE_LINE", block_size).replace("N_BLOCKS_LINE", n_blocks),
    ]
    return "\n\n".join(parts)


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("no C++ compiler (g++, c++ or clang++) to build the quantize probe")
    work = tmp_path_factory.mktemp("fp8_quant_grid_stride")
    source = work / "fp8_quant_probe.cpp"
    source.write_text(probe_source(), encoding="utf-8")
    binary = work / "fp8_quant_probe"
    # -fwrapv: int arithmetic wraps, as the add.s32 nvcc emits for the old loop index
    built = subprocess.run(
        [compiler, "-std=c++20", "-O2", "-fwrapv", "-Wall", "-Wno-unused-function", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stderr[-4000:]
    return binary


# (threads per SM, SMs): the launchers size the grid from these
RTX5090 = (1536, 170)
H100 = (2048, 132)
QWEN_V = 151936  # Qwen2.5/3
VEC = {"e5m2-bf16": 8, "e4m3-bf16": 8, "e4m3-f32": 4}


def launch(device: tuple[int, int]) -> tuple[int, int]:
    """The launchers' block and grid (kept equal to the source by the probe's own launch line, checked per run)."""
    threads_per_sm, sms = device
    block = 1024 if threads_per_sm == 2048 else 768
    return block, threads_per_sm // block * sms


def input_value(e: int) -> int:
    return ((e * 2654435761) >> 7) % 201 - 100


def run_probe(probe: Path, tmp_path: Path, kernel: str, types: str, N: int, device, thread: int):
    out_path = tmp_path / "probe.txt"
    args = [str(probe), kernel, types, str(N), str(device[0]), str(device[1]), str(thread), str(out_path)]
    out = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if out.stdout.startswith("NOMAP"):
        pytest.skip("cannot reserve the probe's address space")
    if out.stdout.startswith("FAULT"):
        in_off, out_off = (int(x) for x in out.stdout.split()[1:3])
        pytest.fail(
            f"{kernel}<{types}> with N = {N}, thread {thread}: access outside the thread's elements "
            f"(input offset {in_off}, output offset {out_off}): the grid-stride index wrapped in 32 bits"
        )
    if out.stdout.startswith("RUNAWAY"):
        pytest.fail(f"{kernel}<{types}> with N = {N}, thread {thread}: more iterations than elements ({out.stdout})")
    assert out.returncode == 0 and out.stdout.startswith("OK"), out.stdout + out.stderr
    lines = out_path.read_text().splitlines()
    block, grid = (int(x) for x in lines[0].split()[1:])
    assert (block, grid) == launch(device), "the probe's launch shape no longer matches launch(); update the test"
    absmax = float(lines[1].split()[1])
    scale = float(lines[2].split()[1])
    loads = [int(line.split()[1]) for line in lines[3:] if line.startswith("load ")]
    stores = [[int(x) for x in line.split()[1:]] for line in lines[3:] if line.startswith("store ")]
    return block, grid, absmax, scale, loads, stores


def check(probe, tmp_path, kernel, types, N, device, thread):
    vec = VEC[types]
    block, grid, absmax, scale, loads, stores = run_probe(probe, tmp_path, kernel, types, N, device, thread)
    expected = list(range(vec * thread, N, vec * block * grid))
    assert expected, "thread has no elements"
    assert loads == expected
    if kernel == "absmax":
        assert stores == []
        assert absmax == max(abs(input_value(e + k)) for e in expected for k in range(vec))
    else:
        assert [s[0] for s in stores] == expected
        for e, *codes in stores:
            assert codes == [input_value(e + k) & 0xFF for k in range(vec)], e
        if thread == 0:
            assert scale == 1.0


def last_vector_thread(N: int, types: str, device) -> int:
    """The thread that loads the last vector [N - vec, N): the first one whose next index passes 2^31 - 1."""
    vec = VEC[types]
    block, grid = launch(device)
    return ((N - vec) % (vec * block * grid)) // vec


def wrap_threshold(types: str, device) -> int:
    """Smallest N (a multiple of vec) at which the int index wraps: N - vec + S > 2^31 - 1."""
    vec = VEC[types]
    block, grid = launch(device)
    return 2**31 - vec * block * grid + vec


# lm_head_dx_matmul's chunk with lmhead_chunks 1: 16,384 rows (e.g. batch 1, seq 16384) x V = 151936, > 2^31
LM_HEAD_N = 16384 * QWEN_V


@pytest.mark.parametrize("types", ["e5m2-bf16", "e4m3-f32"])
@pytest.mark.parametrize("device", [RTX5090, H100], ids=["rtx5090", "h100"])
@pytest.mark.parametrize("which", ["first", "last"])
def test_abs_max_past_2_31_elements(probe, tmp_path, types, device, which):
    block, grid = launch(device)
    thread = 0 if which == "first" else block * grid - 1
    check(probe, tmp_path, "absmax", types, LM_HEAD_N, device, thread)


@pytest.mark.parametrize("types", ["e5m2-bf16", "e4m3-bf16", "e4m3-f32"])
@pytest.mark.parametrize("which", ["first", "last"])
def test_quantize_past_2_31_elements(probe, tmp_path, types, which):
    block, grid = launch(RTX5090)
    thread = 0 if which == "first" else block * grid - 1
    check(probe, tmp_path, "quantize", types, LM_HEAD_N, RTX5090, thread)


@pytest.mark.parametrize("kernel", ["absmax", "quantize"])
def test_below_2_31_elements_the_last_vector_still_wraps(probe, tmp_path, kernel):
    """14,121 x 151936 < 2^31, but the thread holding the last vector steps past 2^31 - 1 (RTX 5090, BF16)."""
    types = "e5m2-bf16"
    N = 14121 * QWEN_V
    assert wrap_threshold(types, RTX5090) <= N < 2**31
    check(probe, tmp_path, kernel, types, N, RTX5090, last_vector_thread(N, types, RTX5090))


@pytest.mark.parametrize("kernel", ["absmax", "quantize"])
def test_the_last_shape_below_the_threshold_is_unaffected(probe, tmp_path, kernel):
    """Passes on upstream's quant.cu too: 14,120 x 151936 leaves every thread's next index below 2^31."""
    types = "e5m2-bf16"
    N = 14120 * QWEN_V
    assert N < wrap_threshold(types, RTX5090)
    check(probe, tmp_path, kernel, types, N, RTX5090, last_vector_thread(N, types, RTX5090))
