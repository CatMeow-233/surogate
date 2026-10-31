// NVFP4 Four Over Six quantization and fresh SR noise on graph replays (fork ledger T24).
// Built into integration-tests by fork/native/cmake/fork_native.cmake. Needs an SM100+ GPU (the RTX 5090 is
// sm_120a); skipped otherwise. The CPU side of these topics is fork/patches/tests/native/test_nvfp4_4o6.py; this
// file checks that the kernels compute what that emulation computes:
//   - SR: the codes and scales of one seed match a host model built from surogate_fork/nvfp4_4o6_arith.h (the
//     only differences allowed come from rcp.approx being within 1 ulp of 1/x), and the average over 1024 seeds
//     converges to the input as 1/sqrt(seeds). Upstream's kernel (SUROGATE_NVFP4_4O6_SR_LEGACY=1) plateaus near
//     5e-2 on this tensor.
//   - RN: the codes and scales match the host model; blocks below upstream's 1/128 clamp decode at their size.
//   - Graphs: a replayed capture of the 4/6 SR and the standard SR quantizer gives new codes on every replay,
//     eager calls with one seed give identical codes. With SUROGATE_NVFP4_4O6_SR_LEGACY=1 the 4/6 codes repeat
//     (upstream's kernel takes no generation); the standard codes still follow SUROGATE_NVFP4_SR_GRAPH_LEGACY.
#include <catch2/catch_test_macros.hpp>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <string_view>
#include <vector>

#include "kernels/kernels.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "surogate_fork/nvfp4_4o6.h"
#include "surogate_fork/nvfp4_4o6_arith.h"

namespace {

namespace q4 = surogate_fork::nvfp4;

void require_cuda(cudaError_t status) {
    REQUIRE(status == cudaSuccess);
}

bool have_sm100(cudaDeviceProp& prop) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return false;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
    return prop.major >= 10;
}

bool env_is_one(const char* name) {
    const char* value = std::getenv(name);
    return value && std::string_view(value) == "1";
}

template <typename T> struct Buffer {
    T* ptr = nullptr;
    size_t count = 0;
    explicit Buffer(size_t n) : count(n) {
        require_cuda(cudaMalloc(&ptr, n * sizeof(T)));
        require_cuda(cudaMemset(ptr, 0, n * sizeof(T)));
    }
    ~Buffer() { cudaFree(ptr); }
    std::vector<T> get() const {
        std::vector<T> out(count);
        require_cuda(cudaMemcpy(out.data(), ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
        return out;
    }
};

//! A dout^T-like BF16 tensor: rows (output channels) with log-normal scales; `outliers` makes 0.2% of the columns
//! (tokens) 3e4 times larger, which puts most blocks below upstream's 1/128 scale clamp.
std::vector<float> make_input(int rows, int cols, uint32_t seed, bool outliers) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::vector<float> big(cols);
    for (int c = 0; c < cols; ++c)
        big[c] = outliers && uniform(gen) < 0.002f ? 3e4f : 1.0f;
    std::vector<float> x(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        const float scale = std::exp(normal(gen));
        for (int c = 0; c < cols; ++c)
            x[static_cast<size_t>(r) * cols + c] = __bfloat162float(__float2bfloat16(normal(gen) * scale * big[c]));
    }
    return x;
}

std::vector<nv_bfloat16> to_bf16(const std::vector<float>& x) {
    std::vector<nv_bfloat16> out(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = __float2bfloat16(x[i]);
    return out;
}

size_t scale_index(int row, int col, int cols) {
    const int atoms_per_row = (cols + 3) / 4;
    const size_t atom = static_cast<size_t>(row / 128) * atoms_per_row + col / 4;
    const int r = row % 128;
    return atom * 512 + (r % 32) * 16 + (r / 32) * 4 + col % 4;
}

float e2m1_value(uint32_t nibble) {
    static const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return (nibble & 8) ? -grid[nibble & 7] : grid[nibble & 7];
}

uint32_t e2m1_code(float q) {
    static const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint32_t i = 0; i < 8; ++i)
        if (std::fabs(q) == grid[i]) return i | (std::signbit(q) ? 8u : 0u);
    return 0xFF;
}

float e2m1_rn(float y) {
    const float a = std::fabs(y);
    float m = 6.0f;
    if (a <= 0.25f) m = 0.0f;
    else if (a < 0.75f) m = 0.5f;
    else if (a <= 1.25f) m = 1.0f;
    else if (a < 1.75f) m = 1.5f;
    else if (a <= 2.5f) m = 2.0f;
    else if (a < 3.5f) m = 3.0f;
    else if (a <= 5.0f) m = 4.0f;
    return std::copysign(m, y);
}

struct Quantized {
    std::vector<uint8_t> codes;   // [M, K/2], element 2j in the low nibble
    std::vector<uint8_t> scales;  // CUTLASS layout
    float amax = 0.0f;
};

std::vector<double> dequantize(const Quantized& q, int M, int K) {
    const int cols = K / 16;
    std::vector<double> out(static_cast<size_t>(M) * K);
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c) {
            const uint8_t byte = q.codes[static_cast<size_t>(r) * (K / 2) + c / 2];
            const uint32_t nibble = (c % 2) ? byte >> 4 : byte & 0xF;
            const float sf = q4::e4m3_value(q.scales[scale_index(r, c / 16, cols)]);
            out[static_cast<size_t>(r) * K + c] = static_cast<double>(e2m1_value(nibble)) * sf * q.amax / 1536.0;
        }
    return out;
}

//! The MSE 4-vs-6 choice with the multiply-adds fused as nvcc compiles them.
bool select_four(const float* v, float enc6, float enc4, float recon6, float recon4) {
    float err6 = 0.0f, err4 = 0.0f;
    for (int g = 0; g < 2; ++g) {
        float e6 = 0.0f, e4 = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const float x = v[g * 8 + i];
            const float d6 = std::fma(e2m1_rn(x * enc6), recon6, -x), d4 = std::fma(e2m1_rn(x * enc4), recon4, -x);
            e6 = std::fma(d6, d6, e6);
            e4 = std::fma(d4, d4, e4);
        }
        err6 += e6;
        err4 += e4;
    }
    return err4 < err6;
}

//! Host model of the fork kernels (rcp.approx modelled as 1/x). Returns codes as E2M1 nibbles [M, K].
void model(const std::vector<float>& x, int M, int K, bool stochastic, uint32_t seed,
           std::vector<uint32_t>& nibbles, std::vector<uint8_t>& block_codes) {
    float g = 0.0f;
    for (float v : x)
        g = std::fmax(g, std::fabs(v));
    const float genc = g > 0.0f ? 1536.0f * (1.0f / std::fmax(g, 1e-12f)) : 1.0f;
    const int cols = K / 16;
    nibbles.assign(x.size(), 0);
    block_codes.assign(static_cast<size_t>(M) * cols, 0);
    for (int r = 0; r < M; ++r)
        for (int b = 0; b < cols; ++b) {
            const float* v = &x[static_cast<size_t>(r) * K + b * 16];
            float amax = 0.0f;
            for (int i = 0; i < 16; ++i)
                amax = std::fmax(amax, std::fabs(v[i]));
            if (amax == 0.0f) continue;
            float q[16];
            uint8_t code;
            if (stochastic) {
                const float inv = 1.0f / std::fmax(amax, 1e-12f);
                const float recon6 = amax * (1.0f / 6.0f), recon4 = amax * 0.25f;
                const bool four = select_four(v, 6.0f * inv, 4.0f * inv, recon6, recon4);
                code = q4::e4m3_code_no_saturation(q4::clamp_block_scale((four ? recon4 : recon6) * genc));
                const float enc = genc / q4::e4m3_value(code);
                const q4::Philox4x32 rnd = q4::sr_block_random(static_cast<uint32_t>(r * cols + b), 0, seed);
                for (int i = 0; i < 16; ++i)
                    q[i] = q4::e2m1_stochastic(v[i] * enc, q4::sr_byte(rnd, i));
            } else {
                const uint8_t c6 = q4::e4m3_code_rn(q4::clamp_block_scale(amax * (1.0f / 6.0f) * genc));
                const uint8_t c4 = q4::e4m3_code_rn(q4::clamp_block_scale(amax * 0.25f * genc));
                const float s6 = q4::e4m3_value(c6), s4 = q4::e4m3_value(c4);
                const bool four = select_four(v, genc / s6, genc / s4, s6 / genc, s4 / genc);
                code = four ? c4 : c6;
                const float enc = genc / q4::e4m3_value(code);
                for (int i = 0; i < 16; ++i)
                    q[i] = e2m1_rn(v[i] * enc);
            }
            block_codes[static_cast<size_t>(r) * cols + b] = code;
            for (int i = 0; i < 16; ++i)
                nibbles[static_cast<size_t>(r) * K + b * 16 + i] = e2m1_code(q[i]);
        }
}

//! Fractions of element codes and of block scales that differ from the host model.
std::pair<double, double> mismatch(const Quantized& q, const std::vector<float>& x, int M, int K, bool stochastic,
                                   uint32_t seed) {
    std::vector<uint32_t> nibbles;
    std::vector<uint8_t> codes;
    model(x, M, K, stochastic, seed, nibbles, codes);
    const int cols = K / 16;
    size_t bad_elements = 0, bad_scales = 0;
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c) {
            const uint8_t byte = q.codes[static_cast<size_t>(r) * (K / 2) + c / 2];
            const uint32_t nibble = (c % 2) ? byte >> 4 : byte & 0xF;
            const uint32_t want = nibbles[static_cast<size_t>(r) * K + c];
            // -0 and +0 are both zero for the GEMM
            if (nibble != want && !((nibble & 7) == 0 && (want & 7) == 0)) ++bad_elements;
            if (c % 16 == 0 && q.scales[scale_index(r, c / 16, cols)] != codes[static_cast<size_t>(r) * cols + c / 16])
                ++bad_scales;
        }
    return {static_cast<double>(bad_elements) / x.size(), static_cast<double>(bad_scales) / (static_cast<size_t>(M) * cols)};
}

double relative_error(const std::vector<double>& a, const std::vector<float>& x) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        num += (a[i] - x[i]) * (a[i] - x[i]);
        den += static_cast<double>(x[i]) * x[i];
    }
    return std::sqrt(num / den);
}

struct Device {
    int M, K;
    Buffer<nv_bfloat16> in;
    Buffer<uint8_t> codes, scales;
    Buffer<float> amax;
    Device(int M, int K, const std::vector<float>& x)
        : M(M),
          K(K),
          in(x.size()),
          codes(static_cast<size_t>(M) * K / 2),
          scales(compute_nvfp4_cutlass_scale_size(M, K)),
          amax(1) {
        const auto bf = to_bf16(x);
        require_cuda(cudaMemcpy(in.ptr, bf.data(), bf.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    }
    Quantized fetch() const {
        Quantized q{codes.get(), scales.get(), amax.get()[0]};
        return q;
    }
};

}  // namespace

TEST_CASE("4/6 SR matches the host model and its seed average converges", "[nvfp4-4o6][cuda]") {
    cudaDeviceProp prop{};
    if (!have_sm100(prop)) SKIP("needs an SM100+ GPU");
    if (!q4::four_over_six_sr_fixed()) SKIP("SUROGATE_NVFP4_4O6_SR_LEGACY=1 selects upstream's kernel");
    constexpr int M = 32, K = 4096;
    const std::vector<float> x = make_input(M, K, 17, false);
    Device dev(M, K, x);
    std::vector<double> sum(x.size(), 0.0);
    double err256 = 0.0, err1024 = 0.0;
    for (int n = 0; n < 1024; ++n) {
        const uint32_t seed = 1000u + 7919u * n;
        quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(dev.codes.ptr, dev.scales.ptr, dev.amax.ptr, dev.in.ptr, M,
                                                         K, recipes::FourOverSixErrorMetric::MSE, seed, prop, nullptr);
        require_cuda(cudaDeviceSynchronize());
        const Quantized q = dev.fetch();
        if (n < 2) {
            const auto [elements, scales] = mismatch(q, x, M, K, true, seed);
            INFO("seed " << seed << ": element mismatch " << elements << ", scale mismatch " << scales);
            CHECK(elements <= 1e-3);
            CHECK(scales <= 1e-3);
        }
        const std::vector<double> deq = dequantize(q, M, K);
        for (size_t i = 0; i < sum.size(); ++i)
            sum[i] += deq[i];
        if (n + 1 == 256 || n + 1 == 1024) {
            std::vector<double> mean(sum.size());
            for (size_t i = 0; i < sum.size(); ++i)
                mean[i] = sum[i] / (n + 1);
            (n + 1 == 256 ? err256 : err1024) = relative_error(mean, x);
        }
    }
    INFO("||E[x_hat] - x|| / ||x||: " << err256 << " at 256 seeds, " << err1024 << " at 1024 (upstream ~5e-2)");
    CHECK(err1024 <= 6e-3);
    CHECK(err256 / err1024 >= 1.6);
}

TEST_CASE("4/6 RN matches the host model and decodes small blocks at their size", "[nvfp4-4o6][cuda]") {
    cudaDeviceProp prop{};
    if (!have_sm100(prop)) SKIP("needs an SM100+ GPU");
    if (!q4::four_over_six_rn_fixed()) SKIP("SUROGATE_NVFP4_4O6_RN_LEGACY=1 selects upstream's kernel");
    constexpr int M = 32, K = 4096;
    for (bool outliers : {false, true}) {
        const std::vector<float> x = make_input(M, K, 23, outliers);
        Device dev(M, K, x);
        quantize_nvfp4_4o6_cutlass_auto_scale(dev.codes.ptr, dev.scales.ptr, dev.amax.ptr, dev.in.ptr, M, K,
                                              recipes::FourOverSixErrorMetric::MSE, prop, nullptr);
        require_cuda(cudaDeviceSynchronize());
        const Quantized q = dev.fetch();
        const auto [elements, scales] = mismatch(q, x, M, K, false, 0);
        INFO("outliers=" << outliers << ": element mismatch " << elements << ", scale mismatch " << scales);
        CHECK(elements <= 1e-3);
        CHECK(scales <= 1e-3);
        if (!outliers) continue;
        // Least-squares gain of the blocks whose exact decode scale is below 1/128 (upstream: ~1.9x inflated).
        const std::vector<double> deq = dequantize(q, M, K);
        double xy = 0.0, xx = 0.0;
        size_t small = 0;
        for (size_t b = 0; b < x.size() / 16; ++b) {
            float amax = 0.0f;
            for (int i = 0; i < 16; ++i)
                amax = std::fmax(amax, std::fabs(x[b * 16 + i]));
            if (amax == 0.0f || amax * 256.0 / q.amax >= 1.0 / 128.0) continue;
            ++small;
            for (int i = 0; i < 16; ++i) {
                xy += x[b * 16 + i] * deq[b * 16 + i];
                xx += static_cast<double>(x[b * 16 + i]) * x[b * 16 + i];
            }
        }
        INFO(small << " small blocks, gain " << xy / xx);
        REQUIRE(small > x.size() / 32);
        CHECK(std::fabs(xy / xx - 1.0) <= 0.03);
    }
}

TEST_CASE("NVFP4 SR draws fresh noise on every CUDA graph replay", "[nvfp4-4o6][sr-graph][cuda]") {
    cudaDeviceProp prop{};
    if (!have_sm100(prop)) SKIP("needs an SM100+ GPU");
    const bool fresh = !env_is_one("SUROGATE_NVFP4_SR_GRAPH_LEGACY");
    // SUROGATE_NVFP4_4O6_SR_LEGACY=1 runs upstream's 4/6 SR kernel, which takes no generation: its replays reuse the
    // captured noise whatever the graph switch says. The standard SR kernels follow the graph switch alone.
    const bool four_fresh = fresh && q4::four_over_six_sr_fixed();
    constexpr int M = 128, K = 1024;
    const std::vector<float> x = make_input(M, K, 29, false);
    Device four(M, K, x), standard(M, K, x);
    cudaStream_t stream;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    auto quantize = [&] {
        quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(four.codes.ptr, four.scales.ptr, four.amax.ptr, four.in.ptr,
                                                         M, K, recipes::FourOverSixErrorMetric::MSE, 7u, prop, stream);
        quantize_nvfp4_stochastic_cutlass_auto_scale(standard.codes.ptr, standard.scales.ptr, standard.amax.ptr,
                                                     standard.in.ptr, M, K, 7u, prop, stream);
    };
    // Eager: one seed, one result.
    quantize();
    require_cuda(cudaStreamSynchronize(stream));
    const auto eager_four = four.codes.get(), eager_standard = standard.codes.get();
    quantize();
    require_cuda(cudaStreamSynchronize(stream));
    CHECK(four.codes.get() == eager_four);
    CHECK(standard.codes.get() == eager_standard);
    // Captured: every replay draws new noise (scales and amax are deterministic).
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    quantize();
    require_cuda(cudaStreamEndCapture(stream, &graph));
    require_cuda(cudaGraphInstantiate(&exec, graph, 0));
    std::vector<std::vector<uint8_t>> four_codes, standard_codes, four_scales;
    for (int replay = 0; replay < 3; ++replay) {
        require_cuda(cudaGraphLaunch(exec, stream));
        require_cuda(cudaStreamSynchronize(stream));
        four_codes.push_back(four.codes.get());
        standard_codes.push_back(standard.codes.get());
        four_scales.push_back(four.scales.get());
    }
    for (int replay = 1; replay < 3; ++replay) {
        INFO("replay " << replay << ", fresh noise expected: 4/6 " << four_fresh << ", standard " << fresh);
        CHECK((four_codes[replay] != four_codes[replay - 1]) == four_fresh);
        CHECK((standard_codes[replay] != standard_codes[replay - 1]) == fresh);
        CHECK(four_scales[replay] == four_scales[0]);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
}
