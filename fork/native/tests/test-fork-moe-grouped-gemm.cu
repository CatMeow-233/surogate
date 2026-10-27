// MoE base expert weight gradients: conditional zero-fill before the beta=0 GEMMs (fork ledger
// T3). Built into integration-tests by fork/native/cmake/fork_native.cmake. The helpers below are
// copies of the ones in csrc/src/testing/kernels/test-moe-grouped-gemm.cu; the anonymous
// namespace keeps both copies local to their translation unit.
#include <catch2/catch_test_macros.hpp>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "kernels/kernels.h"
#include "surogate_fork/moe_weight_grad_policy.h"

namespace {

std::vector<int> offsets_of(const std::vector<int>& t) {
    std::vector<int> off(t.size() + 1, 0);
    for (std::size_t e = 0; e < t.size(); ++e)
        off[e + 1] = off[e] + t[e];
    return off;
}

// Member functions check through require_cuda rather than REQUIRE: CUDA 13.0's cudafe++ (the DGX
// Spark's toolkit) never finishes on some REQUIREs inside member functions of this file's classes.
void require_cuda(cudaError_t status) { REQUIRE(status == cudaSuccess); }
void require_cublas(cublasStatus_t status) { REQUIRE(status == CUBLAS_STATUS_SUCCESS); }

struct Bf16 {
    std::vector<float> host;  // the bf16-rounded values
    nv_bfloat16* dev = nullptr;
    Bf16(std::size_t n, float scale, std::mt19937& rng)
        : host(n) {
        std::uniform_real_distribution<float> u(-scale, scale);
        std::vector<nv_bfloat16> h(n);
        for (std::size_t i = 0; i < n; ++i) {
            h[i] = __float2bfloat16(scale == 0.0f ? 0.0f : u(rng));
            host[i] = __bfloat162float(h[i]);
        }
        require_cuda(cudaMalloc(&dev, std::max<std::size_t>(n, 1) * sizeof(nv_bfloat16)));
        require_cuda(cudaMemcpy(dev, h.data(), n * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
    }
    Bf16(const Bf16&) = delete;
    ~Bf16() {
        cudaFree(dev);
    }
    std::vector<float> read() const {
        std::vector<nv_bfloat16> h(host.size());
        require_cuda(cudaDeviceSynchronize());
        require_cuda(cudaMemcpy(h.data(), dev, h.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<float> out(h.size());
        for (std::size_t i = 0; i < h.size(); ++i)
            out[i] = __bfloat162float(h[i]);
        return out;
    }
};

struct Offsets {
    std::vector<int> host;
    int* dev = nullptr;
    explicit Offsets(std::vector<int> off)
        : host(std::move(off)) {
        require_cuda(cudaMalloc(&dev, host.size() * sizeof(int)));
        require_cuda(cudaMemcpy(dev, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    ~Offsets() {
        cudaFree(dev);
    }
    int total() const {
        return host.back();
    }
};

struct Handles {
    cublasHandle_t cublas{};
    cudaStream_t stream{};
    Handles() {
        require_cuda(cudaStreamCreate(&stream));
        require_cublas(cublasCreate(&cublas));
    }
    ~Handles() {
        cublasDestroy(cublas);
        cudaStreamDestroy(stream);
    }
};

}  // namespace

TEST_CASE("Base expert weight gradients preserve skipped and accumulated regions", "[moe][gemm][wgrad-zero]") {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) SKIP("no CUDA device");
    Handles h;
    std::mt19937 rng(20261024);
    constexpr int experts = 4, rows = 16, cols = 24;
    struct RoutingCase {
        std::vector<int> counts;
        bool have_empty;
    };
    for (const auto& routing :
         {RoutingCase{{2, 1, 3, 2}, false}, RoutingCase{{2, 0, 3, 0}, true}, RoutingCase{{0, 0, 0, 0}, true}}) {
        const auto& counts = routing.counts;
        Offsets off(offsets_of(counts));
        Bf16 input(off.total() * cols, 0.25f, rng), dout(off.total() * rows, 0.25f, rng);
        for (const bool accumulate : {false, true}) {
            INFO("tokens=" << off.total() << " accumulate=" << accumulate);
            Bf16 dw(experts * rows * cols, 1.0f, rng);
            const auto initial = dw.host;
            const bool clear = dsl::moe_weight_grad_needs_zero(off.host.data(), experts, accumulate);
            // Full overwrite avoids a redundant HBM pass; any skipped region
            // still requires initialization, and accumulate must preserve it.
            CHECK(clear == (!accumulate && routing.have_empty));
            if (clear)
                REQUIRE(cudaMemsetAsync(dw.dev, 0, dw.host.size() * sizeof(nv_bfloat16), h.stream) == cudaSuccess);
            moe_grouped_gemm_weight_grad(dw.dev,
                                         dout.dev,
                                         input.dev,
                                         off.dev,
                                         experts,
                                         rows,
                                         cols,
                                         h.cublas,
                                         h.stream,
                                         off.host.data(),
                                         1.0f,
                                         accumulate ? 1.0f : 0.0f,
                                         nullptr,
                                         false,
                                         -1);
            std::vector<double> expected(initial.size(), 0.0);
            if (accumulate) expected.assign(initial.begin(), initial.end());
            for (int e = 0; e < experts; ++e) {
                for (int t = off.host[e]; t < off.host[e + 1]; ++t) {
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            expected[(static_cast<std::size_t>(e) * rows + i) * cols + j] +=
                                static_cast<double>(dout.host[t * rows + i]) * input.host[t * cols + j];
                        }
                    }
                }
            }
            const auto actual = dw.read();
            // Elementwise absolute+relative tolerance also covers the all-empty
            // overwrite case, whose zero reference has no meaningful relative L2.
            for (std::size_t i = 0; i < actual.size(); ++i)
                CHECK(std::abs(actual[i] - expected[i]) <= 0.004 * std::abs(expected[i]) + 0.0001);
            for (int e = 0; e < experts; ++e) {
                if (counts[e] != 0) continue;
                for (int j = 0; j < rows * cols; ++j) {
                    const auto i = static_cast<std::size_t>(e) * rows * cols + j;
                    CHECK(actual[i] == (accumulate ? initial[i] : 0.0f));
                }
            }
        }
    }
}
