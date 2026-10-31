"""Ledger topic T23 (64-bit row offsets in the cross-entropy kernels), runnable without CUDA.

fused_classifier.cu's cross_entropy_forward_kernel / cross_entropy_backward_kernel (V <= 65536) and
chunked_cross_entropy_forward_kernel / chunked_cross_entropy_backward_kernel (V > 65536) indexed logits and dlogits
with `row * P + i` in int. nvcc emits that as a wrapping 32-bit multiply-add (mul.lo.s32 / mad.lo.s32) followed by
mul.wide.s32 to a byte offset, so once a row's offset passes 2^31 - 1 elements the access lands up to 2^31 elements
below the buffer (rows x V > 2^31 elements: from 14,135 rows at V = 151936, 32,769 rows at V = 65536), or inside an
earlier row once it passes 2^32. The fused LM-head loss runs these kernels on one LM-head chunk of rows, which
reaches that size with lmhead_chunks 1 or 2 from 14k-28k tokens per micro-batch, or 8 at 128k (upstream-errata, T23).

The probe compiles the four kernels, prepare_softmax_blockwide3, SoftmaxParams and maybe_softcap_value as they stand
in the sources, plus the chunk sizes from kernels.h, as host C++ with -fwrapv, so int arithmetic wraps as the PTX
does. It runs one block per call with one thread (blockReduce is then the identity), which makes that thread touch
every element the GPU block would. The logits buffer is a PROT_NONE reservation of rows x V elements, with another
2^31 elements below it, and only the row under test is read-write: any access outside that row faults, and the probe
reports the faulting element offset. On upstream's sources (488e5f00) each overflow case below faults; with the fix
the values match a float64 reference. The in-range cases pass on both. GPU behaviour is unverified here; the T23
errata section gives the 5090 check.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[4]
KERNEL = REPO / "csrc/src/kernels/fused_classifier.cu"
KERNELS_H = REPO / "csrc/src/kernels/kernels.h"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the probe reserves address space with mmap"
)


def definition(path: Path, signature: str, terminator: str = "") -> str:
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
            return text[match.start() : i + 1] + terminator
        i += 1
    raise AssertionError(f"{path.relative_to(REPO)}: unbalanced braces after {signature!r}")


def line_matching(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match, f"{path.relative_to(REPO)}: no line matching {pattern!r}"
    return match.group(0)


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
#include <vector>

// ---- CUDA host stubs: one thread per block ----------------------------------------------------------------------
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__ static
struct probe_dim3 { unsigned int x, y, z; };
static probe_dim3 blockIdx{0, 0, 0}, threadIdx{0, 0, 0}, blockDim{1, 1, 1};
static inline void __syncthreads() {}
template <class T> static inline T __shfl_down_sync(unsigned, T value, int) { return value; }
static inline int atomicAdd(int* address, int value) { int old = *address; *address += value; return old; }
static inline float warpReduceMax(float value) { return value; }
static inline float warpReduceSum(float value) { return value; }

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

template <class T, std::size_t N> struct GenericVector {  // vec.cuh: element access, size, load
    T values[N];
    static constexpr const std::size_t size = N;
    T& operator[](int index) { return values[index]; }
    const T& operator[](int index) const { return values[index]; }
    static GenericVector load(const T* address) {
        GenericVector result;
        for (std::size_t k = 0; k < N; ++k) result.values[k] = address[k];
        return result;
    }
};
"""

PROBE_MAIN = r"""
// ---- probe ------------------------------------------------------------------------------------------------------
static char* g_base = nullptr;
static long long g_elem = 1;

static void on_fault(int, siginfo_t* info, void*) {
    char text[64];
    long long bytes = static_cast<long long>(static_cast<char*>(info->si_addr) - g_base);
    int n = std::snprintf(text, sizeof text, "FAULT %lld\n", bytes / g_elem);
    if (write(1, text, n) < 0) {}
    _exit(3);
}

template <class T>
static int run(const char* kernel, int rows, int V, int row, int target, int first, int last, float lse, float dloss,
               const char* in_path, const char* out_path) {
    const int P = V;
    const long long elem = sizeof(T);
    const long long page = sysconf(_SC_PAGESIZE);
    const long long below = (1LL << 31) * elem + page;  // a wrapped int row offset lands at most 2^31 elements below
    const long long buffer = static_cast<long long>(rows) * P * elem;
    void* reservation = mmap(nullptr, below + buffer + page, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                             -1, 0);
    if (reservation == MAP_FAILED) {
        std::printf("NOMAP\n");
        return 4;
    }
    g_base = static_cast<char*>(reservation) + below;
    g_elem = elem;
    const long long row_lo = static_cast<long long>(row) * P * elem;
    const long long open_lo = row_lo / page * page;
    const long long open_hi = (row_lo + P * elem + page - 1) / page * page;
    if (mprotect(g_base + open_lo, open_hi - open_lo, PROT_READ | PROT_WRITE) != 0) {
        std::printf("NOMAP\n");
        return 4;
    }
    T* logits = reinterpret_cast<T*>(g_base);
    T* logits_row = reinterpret_cast<T*>(g_base + row_lo);
    std::vector<float> in(V);
    FILE* f = std::fopen(in_path, "rb");
    if (!f || std::fread(in.data(), 4, V, f) != static_cast<size_t>(V)) return 5;
    std::fclose(f);
    for (int i = 0; i < V; ++i) logits_row[i] = static_cast<T>(in[i]);

    const int n_chunks = (V + CROSS_ENTROPY_MAX_FUSED_SIZE - 1) / CROSS_ENTROPY_MAX_FUSED_SIZE;
    std::vector<int> targets(rows, 0);
    targets[row] = target;
    std::vector<float> losses(rows, 0.25f), logsumexp(rows, lse), dlosses(rows, dloss);
    std::vector<float> chunk_lse(static_cast<size_t>(rows) * n_chunks, NAN);
    int valid = 0;

    struct sigaction action {};
    action.sa_sigaction = on_fault;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);

    blockIdx.x = static_cast<unsigned>(row);
    for (int b = first; b <= last; ++b) {
        blockIdx.y = static_cast<unsigned>(b);
        if (!std::strcmp(kernel, "chunked_forward")) {
            chunked_cross_entropy_forward_kernel<T>(logits, losses.data(), chunk_lse.data(), targets.data(), &valid,
                                                    rows, V, P, n_chunks, CROSS_ENTROPY_MAX_FUSED_SIZE, 0.0f);
        } else if (!std::strcmp(kernel, "chunked_backward")) {
            chunked_cross_entropy_backward_kernel<T>(logits, logits, logsumexp.data(), dlosses.data(), targets.data(),
                                                     rows, V, P, CROSS_ENTROPY_BACKWARD_CHUNK_SIZE, 0.0f);
        } else if (!std::strcmp(kernel, "forward")) {
            cross_entropy_forward_kernel<T>(logits, losses.data(), logsumexp.data(), targets.data(), &valid, nullptr,
                                            rows, V, P, 0.0f);
        } else if (!std::strcmp(kernel, "backward")) {
            cross_entropy_backward_kernel<T>(logits, logits, logsumexp.data(), dlosses.data(), targets.data(), rows,
                                             V, P, 0.0f);
        } else {
            return 6;
        }
    }

    // row after the run, then loss, valid count, logsumexp and the row's chunk logsumexps
    std::vector<float> out(V + 3 + n_chunks);
    for (int i = 0; i < V; ++i) out[i] = static_cast<float>(logits_row[i]);
    out[V] = losses[row];
    out[V + 1] = static_cast<float>(valid);
    out[V + 2] = logsumexp[row];
    for (int c = 0; c < n_chunks; ++c) out[V + 3 + c] = chunk_lse[static_cast<size_t>(row) * n_chunks + c];
    f = std::fopen(out_path, "wb");
    if (!f || std::fwrite(out.data(), 4, out.size(), f) != out.size()) return 5;
    std::fclose(f);
    std::printf("OK\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 13) return 2;
    const char* kernel = argv[1];
    const bool bf16 = !std::strcmp(argv[2], "bf16");
    const int rows = std::atoi(argv[3]), V = std::atoi(argv[4]), row = std::atoi(argv[5]), target = std::atoi(argv[6]);
    const int first = std::atoi(argv[7]), last = std::atoi(argv[8]);
    const float lse = std::strtof(argv[9], nullptr), dloss = std::strtof(argv[10], nullptr);
    return bf16 ? run<nv_bfloat16>(kernel, rows, V, row, target, first, last, lse, dloss, argv[11], argv[12])
                : run<float>(kernel, rows, V, row, target, first, last, lse, dloss, argv[11], argv[12]);
}
"""


def probe_source() -> str:
    parts = [
        PRELUDE,
        "// ---- kernels.h ----",
        line_matching(KERNELS_H, r"^constexpr int CROSS_ENTROPY_MAX_FUSED_SIZE = \d+;$"),
        line_matching(KERNELS_H, r"^constexpr int CROSS_ENTROPY_BACKWARD_CHUNK_SIZE = \d+;$"),
        "// ---- fused_classifier.cu ----",
        line_matching(KERNEL, r"^using reduction_func_t = float \(\*\)\(float\);$"),
        "template <reduction_func_t> static inline float blockReduce(float value, bool = false, float = 0.0f) "
        "{ return value; }",
        definition(KERNEL, r"struct SoftmaxParams", ";"),
        definition(KERNEL, r"__device__ __forceinline__ float maybe_softcap_value\("),
        definition(KERNEL, r"template <class floatX>\s*__device__ SoftmaxParams\s*prepare_softmax_blockwide3\("),
        definition(KERNEL, r"template <class floatX>\s*__global__ void cross_entropy_forward_kernel\("),
        definition(KERNEL, r"template <class floatX>\s*__global__ void cross_entropy_backward_kernel\("),
        definition(KERNEL, r"template <class floatX>\s*__global__ void chunked_cross_entropy_forward_kernel\("),
        definition(KERNEL, r"template <class floatX>\s*__global__ void chunked_cross_entropy_backward_kernel\("),
        PROBE_MAIN,
    ]
    return "\n\n".join(parts)


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("no C++ compiler (g++, c++ or clang++) to build the cross-entropy probe")
    work = tmp_path_factory.mktemp("ce_row_offsets")
    source = work / "ce_row_offsets_probe.cpp"
    source.write_text(probe_source(), encoding="utf-8")
    binary = work / "ce_row_offsets_probe"
    # -fwrapv: int overflow wraps, as the mul.lo.s32 / add.s32 nvcc emits for these expressions
    built = subprocess.run(
        [compiler, "-std=c++20", "-O2", "-fwrapv", "-Wall", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert built.returncode == 0, built.stderr[-4000:]
    return binary


MAX_FUSED = int(
    re.search(r"\d+", line_matching(KERNELS_H, r"^constexpr int CROSS_ENTROPY_MAX_FUSED_SIZE = \d+;$")).group(0)
)
BACKWARD_CHUNK = int(
    re.search(r"\d+", line_matching(KERNELS_H, r"^constexpr int CROSS_ENTROPY_BACKWARD_CHUNK_SIZE = \d+;$")).group(0)
)
DLOSS = np.float32(1.0 / 1024)


def row_values(V: int) -> np.ndarray:
    """k/64 for k in [-255, 255]: exact in BF16, so both dtypes see the same logits."""
    k = (np.arange(V, dtype=np.int64) * 7919 + 13) % 511 - 255
    return (k / 64.0).astype(np.float32)


def run_probe(
    probe: Path,
    tmp_path: Path,
    kernel: str,
    dtype: str,
    rows: int,
    V: int,
    row: int,
    target: int,
    blocks: tuple[int, int],
    lse: float = 0.0,
):
    values = row_values(V)
    src, dst = tmp_path / "row.f32", tmp_path / "out.f32"
    values.tofile(src)
    args = [
        str(probe),
        kernel,
        dtype,
        str(rows),
        str(V),
        str(row),
        str(target),
        str(blocks[0]),
        str(blocks[1]),
        float(np.float32(lse)).hex(),
        float(DLOSS).hex(),
        str(src),
        str(dst),
    ]
    out = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if out.stdout.startswith("NOMAP"):
        pytest.skip("cannot reserve the probe's address space")
    if out.stdout.startswith("FAULT"):
        offset = int(out.stdout.split()[1])
        pytest.fail(
            f"{kernel}<{dtype}> on row {row} of {rows} x {V}: access at element offset {offset} from the logits base, "
            f"outside the row [{row * V}, {(row + 1) * V}): the row offset wrapped in 32 bits"
        )
    assert out.returncode == 0 and out.stdout.startswith("OK"), out.stdout + out.stderr
    n_chunks = -(-V // MAX_FUSED)
    result = np.fromfile(dst, dtype=np.float32)
    assert result.size == V + 3 + n_chunks
    return values, result[:V], result[V], int(result[V + 1]), result[V + 2], result[V + 3 :]


def logsumexp(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


def rtol(dtype: str) -> float:
    return 2.0**-8 if dtype == "bf16" else 1e-5


def check_chunked_forward(probe, tmp_path, dtype, rows, V, row, target, chunks):
    values, after, loss, valid, _, chunk_lse = run_probe(
        probe, tmp_path, "chunked_forward", dtype, rows, V, row, target, chunks
    )
    assert np.array_equal(after, values)
    for c in range(chunks[0], chunks[1] + 1):
        ref = logsumexp(values[c * MAX_FUSED : min(V, (c + 1) * MAX_FUSED)])
        assert np.isclose(chunk_lse[c], ref, rtol=1e-5, atol=0), (c, chunk_lse[c], ref)
    if chunks[0] == 0:
        assert loss == np.float32(0.25) - values[target]  # losses[row] -= logit[target], exact for these values
        assert valid == 1


def check_backward(probe, tmp_path, kernel, dtype, rows, V, row, target, blocks, span):
    values = row_values(V)
    lse = np.float32(logsumexp(values))
    _, after, *_ = run_probe(probe, tmp_path, kernel, dtype, rows, V, row, target, blocks, lse=float(lse))
    lo, hi = blocks[0] * span, min(V, (blocks[1] + 1) * span)
    expected = values.astype(np.float64).copy()
    if target == -100:
        expected[lo:hi] = 0.0
    else:
        grad = (np.exp(values[lo:hi].astype(np.float64) - float(lse)) - (np.arange(lo, hi) == target)) * float(DLOSS)
        expected[lo:hi] = grad
    assert np.allclose(after, expected, rtol=rtol(dtype), atol=1e-30), np.abs(after - expected).max()


# V = 151936 (Qwen2.5/3): 14,134 rows fit in 2^31 elements, row 14,134 wraps from element 20,224 on.
QWEN_V = 151936
# V = 100000: chunk 0 of row 21,474 ends below 2^31, its target 99,999 lies above.
SPLIT_V, SPLIT_ROW = 100000, 21474
# V = 65536 (LFM2): the non-chunked kernels; 32,768 rows fit, row 32,768 starts at exactly 2^31.
SMALL_V = 65536
DTYPES = ["bf16", "f32"]


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunked_forward_reads_rows_past_2_31_elements(probe, tmp_path, dtype):
    check_chunked_forward(probe, tmp_path, dtype, 14135, QWEN_V, 14134, 5, (0, 2))


@pytest.mark.parametrize("dtype", DTYPES)
def test_chunked_forward_reads_a_target_logit_past_2_31_elements(probe, tmp_path, dtype):
    assert SPLIT_ROW * SPLIT_V + MAX_FUSED - 1 < 2**31 <= SPLIT_ROW * SPLIT_V + SPLIT_V - 1
    check_chunked_forward(probe, tmp_path, dtype, SPLIT_ROW + 1, SPLIT_V, SPLIT_ROW, SPLIT_V - 1, (0, 0))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("target", [7, -100], ids=["target", "masked"])
def test_chunked_backward_writes_rows_past_2_31_elements(probe, tmp_path, dtype, target):
    blocks = (0, -(-QWEN_V // BACKWARD_CHUNK) - 1)
    check_backward(probe, tmp_path, "chunked_backward", dtype, 14135, QWEN_V, 14134, target, blocks, BACKWARD_CHUNK)


def test_chunked_backward_does_not_alias_an_earlier_row_past_2_32_elements(probe, tmp_path):
    """Past 2^32 elements the wrapped offset lands inside the buffer (silent corruption on a GPU, not a fault)."""
    rows = 28270
    assert (rows - 1) * QWEN_V >= 2**32
    blocks = (0, -(-QWEN_V // BACKWARD_CHUNK) - 1)
    check_backward(probe, tmp_path, "chunked_backward", "bf16", rows, QWEN_V, rows - 1, 11, blocks, BACKWARD_CHUNK)


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_reads_the_target_logit_past_2_31_elements(probe, tmp_path, dtype):
    values, after, loss, valid, lse, _ = run_probe(probe, tmp_path, "forward", dtype, 32769, SMALL_V, 32768, 3, (0, 0))
    ref = logsumexp(values)
    assert np.array_equal(after, values)
    assert np.isclose(lse, ref, rtol=1e-5, atol=0)
    assert np.isclose(loss, 0.25 + ref - float(values[3]), rtol=1e-5, atol=0)
    assert valid == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("target", [3, -100], ids=["target", "masked"])
def test_backward_writes_rows_past_2_31_elements(probe, tmp_path, dtype, target):
    check_backward(probe, tmp_path, "backward", dtype, 32769, SMALL_V, 32768, target, (0, 0), SMALL_V)


@pytest.mark.parametrize("kernel", ["chunked_forward", "chunked_backward", "forward", "backward"])
def test_the_last_row_below_2_31_elements_is_unaffected(probe, tmp_path, kernel):
    """Passes on upstream's sources too: the probe faults only where the row offset wraps."""
    V = SMALL_V if kernel in ("forward", "backward") else QWEN_V
    rows = 2**31 // V
    assert rows * V <= 2**31 < (rows + 1) * V
    if kernel == "chunked_forward":
        check_chunked_forward(probe, tmp_path, "bf16", rows, V, rows - 1, V - 1, (0, -(-V // MAX_FUSED) - 1))
    elif kernel == "forward":
        values, _, loss, _, lse, _ = run_probe(probe, tmp_path, kernel, "bf16", rows, V, rows - 1, V - 1, (0, 0))
        assert np.isclose(loss, 0.25 + logsumexp(values) - float(values[V - 1]), rtol=1e-5, atol=0)
    else:
        span = BACKWARD_CHUNK if kernel == "chunked_backward" else V
        blocks = (0, -(-V // span) - 1)
        check_backward(probe, tmp_path, kernel, "bf16", rows, V, rows - 1, V - 1, blocks, span)
