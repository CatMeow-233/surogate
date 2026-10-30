// MoE base expert weight gradients: conditional zero-fill before the beta=0 GEMMs (fork ledger
// T3). Built into integration-tests by fork/native/cmake/fork_native.cmake.
//
// The cases call dsl::moe_base_weight_grad (fork/native/src/moe_weight_grad.cpp, compiled into
// surogate-common), which CompiledExecutor::backward_moe_base_weights calls for every full
// fine-tuning MoE expert weight gradient. Skipping the zero-fill is correct only while each expert
// with tokens gets a beta=0 GEMM over its whole rows x cols block, so the overwrite cases start from a
// NaN-filled dW: any element the GEMMs leave unwritten fails the comparison. That models the
// cpu_training gradient slots, which are reused across layers without being cleared.
#include <catch2/catch_test_macros.hpp>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "surogate_fork/moe_weight_grad.h"
#include "surogate_fork/moe_weight_grad_policy.h"
#include "utilities/tensor.h"

namespace {

constexpr int kExperts = 4, kRows = 16, kCols = 24;

std::vector<int> offsets_of(const std::vector<int>& counts) {
    std::vector<int> off(counts.size() + 1, 0);
    for (std::size_t e = 0; e < counts.size(); ++e)
        off[e + 1] = off[e] + counts[e];
    return off;
}

// Member functions check through require_cuda rather than REQUIRE: CUDA 13.0's cudafe++ (the DGX
// Spark's toolkit) never finishes on some REQUIREs inside member functions of this file's classes.
void require_cuda(cudaError_t status) {
    REQUIRE(status == cudaSuccess);
}
void require_cublas(cublasStatus_t status) {
    REQUIRE(status == CUBLAS_STATUS_SUCCESS);
}

// Values on the BF16 grid, so BF16 and FP32 tensors hold the same numbers.
std::vector<float> random_values(std::size_t n, float scale, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-scale, scale);
    std::vector<float> values(n);
    for (auto& v : values)
        v = __bfloat162float(__float2bfloat16(u(rng)));
    return values;
}

std::vector<float> nan_values(std::size_t n) {
    return std::vector<float>(n, std::numeric_limits<float>::quiet_NaN());
}

// A BF16 or FP32 device tensor and the values it holds.
struct DeviceTensor {
    ETensorDType dtype;
    std::vector<long> shape;
    std::vector<float> host;
    std::byte* dev = nullptr;

    DeviceTensor(ETensorDType type, std::vector<long> dims)
        : dtype(type),
          shape(std::move(dims)) {
        std::size_t n = 1;
        for (long d : shape)
            n *= static_cast<std::size_t>(d);
        host.assign(n, 0.0f);
        require_cuda(cudaMalloc(&dev, std::max<std::size_t>(bytes(), 1)));
    }
    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;
    ~DeviceTensor() {
        cudaFree(dev);
    }

    std::size_t bytes() const {
        return host.size() * (dtype == ETensorDType::BF16 ? sizeof(nv_bfloat16) : sizeof(float));
    }

    void upload(const std::vector<float>& values) {
        host = values;
        if (dtype == ETensorDType::BF16) {
            std::vector<nv_bfloat16> h(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) {
                h[i] = __float2bfloat16(values[i]);
                host[i] = __bfloat162float(h[i]);
            }
            require_cuda(cudaMemcpy(dev, h.data(), bytes(), cudaMemcpyHostToDevice));
        } else {
            require_cuda(cudaMemcpy(dev, values.data(), bytes(), cudaMemcpyHostToDevice));
        }
    }

    std::vector<float> read() const {
        std::vector<float> out(host.size());
        require_cuda(cudaDeviceSynchronize());
        if (dtype == ETensorDType::BF16) {
            std::vector<nv_bfloat16> h(host.size());
            require_cuda(cudaMemcpy(h.data(), dev, bytes(), cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i < h.size(); ++i)
                out[i] = __bfloat162float(h[i]);
        } else {
            require_cuda(cudaMemcpy(out.data(), dev, bytes(), cudaMemcpyDeviceToHost));
        }
        return out;
    }

    Tensor view() const {
        return Tensor::from_pointer(dev, 0, dtype, shape);
    }

    // Rows [start, start + count) of a [tokens, width] tensor, as the FFN tiling narrows them.
    Tensor row_window(long start, long count) const {
        Tensor t = view();
        t.Data += start * shape[1] * static_cast<long>(get_dtype_size(dtype));
        t.Sizes[0] = count;
        return t;
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

bool have_cuda_device() {
    int ndev = 0;
    return cudaGetDeviceCount(&ndev) == cudaSuccess && ndev > 0;
}

// dW[e] = sum over the expert's tokens of dY[t]^T X[t], in double.
std::vector<double> reference(const DeviceTensor& dout, const DeviceTensor& input, const std::vector<int>& off) {
    std::vector<double> dw(static_cast<std::size_t>(kExperts) * kRows * kCols, 0.0);
    for (int e = 0; e < kExperts; ++e)
        for (int t = off[e]; t < off[e + 1]; ++t)
            for (int i = 0; i < kRows; ++i)
                for (int j = 0; j < kCols; ++j)
                    dw[(static_cast<std::size_t>(e) * kRows + i) * kCols + j] +=
                        static_cast<double>(dout.host[t * kRows + i]) * input.host[t * kCols + j];
    return dw;
}

// Elementwise absolute+relative tolerance (BF16 rounding of the result). NaN never passes, and the
// all-empty case's zero reference needs the absolute term.
void check_close(const std::vector<float>& actual, const std::vector<double>& expected) {
    REQUIRE(actual.size() == expected.size());
    std::size_t bad = 0;
    std::ostringstream first;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        if (std::abs(actual[i] - expected[i]) <= 0.004 * std::abs(expected[i]) + 0.0001) continue;
        if (bad++ == 0) first << "first mismatch at " << i << ": " << actual[i] << " vs " << expected[i];
    }
    INFO(bad << " of " << actual.size() << " elements wrong; " << first.str());
    CHECK(bad == 0);
}

// The blocks of experts without tokens hold exactly `expected`.
void check_empty_experts(const std::vector<float>& actual,
                         const std::vector<int>& off,
                         const std::vector<float>& expected) {
    std::size_t bad = 0;
    for (int e = 0; e < kExperts; ++e) {
        if (off[e + 1] > off[e]) continue;
        for (int j = 0; j < kRows * kCols; ++j) {
            const auto i = static_cast<std::size_t>(e) * kRows * kCols + j;
            if (!(actual[i] == expected[i])) ++bad;
        }
    }
    CHECK(bad == 0);
}

void run(Handles& h,
         DeviceTensor& dw,
         const Tensor& dout,
         const Tensor& input,
         const std::vector<int>& off,
         bool accumulate) {
    Tensor dw_view = dw.view();
    dsl::moe_base_weight_grad(dw_view,
                              dout,
                              input,
                              off.data(),
                              kExperts,
                              accumulate,
                              h.cublas,
                              h.stream,
                              "experts_weight_grad");
}

}  // namespace

TEST_CASE("Expert weight gradients zero-fill only when an expert has no tokens", "[moe][wgrad-zero]") {
    const std::vector<int> all_routed = offsets_of({2, 1, 3, 2}), one_empty = offsets_of({2, 0, 3, 2}),
                           none = offsets_of({0, 0, 0, 0});
    CHECK_FALSE(dsl::moe_weight_grad_needs_zero(all_routed.data(), kExperts, false));
    CHECK(dsl::moe_weight_grad_needs_zero(one_empty.data(), kExperts, false));
    CHECK(dsl::moe_weight_grad_needs_zero(none.data(), kExperts, false));
    for (const auto* off : {&all_routed, &one_empty, &none})
        CHECK_FALSE(dsl::moe_weight_grad_needs_zero(off->data(), kExperts, true));
}

TEST_CASE("Expert weight gradients check the destination shape before writing", "[moe][wgrad-zero]") {
    // The policy relies on this check: the per-expert GEMMs cover dW only when it is [experts, rows, cols].
    const std::vector<int> off = offsets_of({2, 1, 3, 2});
    const Tensor dout = Tensor::from_pointer(nullptr, 0, ETensorDType::BF16, std::vector<long>{8, kRows});
    const Tensor input = Tensor::from_pointer(nullptr, 0, ETensorDType::BF16, std::vector<long>{8, kCols});
    for (const auto& shape : {std::vector<long>{kExperts, kCols, kRows},
                              std::vector<long>{kExperts - 1, kRows, kCols},
                              std::vector<long>{kExperts * kRows, kCols}}) {
        Tensor dw = Tensor::from_pointer(nullptr, 0, ETensorDType::FP32, shape);
        for (const bool accumulate : {false, true}) {
            try {
                dsl::moe_base_weight_grad(dw,
                                          dout,
                                          input,
                                          off.data(),
                                          kExperts,
                                          accumulate,
                                          nullptr,
                                          nullptr,
                                          "experts_weight_grad");
                FAIL("no shape error");
            } catch (const std::runtime_error& error) {
                CHECK(std::string(error.what()) == "Expert weight gradient shape mismatch: experts_weight_grad");
            }
        }
    }
}

TEST_CASE("Base expert weight gradients overwrite stale buffers and preserve accumulated ones",
          "[moe][gemm][wgrad-zero]") {
    if (!have_cuda_device()) SKIP("no CUDA device");
    Handles h;
    std::mt19937 rng(20261024);
    struct DtypeCase {
        ETensorDType operands, grad;
    };
    struct RoutingCase {
        std::vector<int> counts;
        int zero_dout_expert;  // an expert with tokens whose dY rows are all zero, or -1
    };
    // BF16 activations with an FP32 gradient is the usual full fine-tuning case.
    for (const auto& types : {DtypeCase{ETensorDType::BF16, ETensorDType::BF16},
                              DtypeCase{ETensorDType::BF16, ETensorDType::FP32},
                              DtypeCase{ETensorDType::FP32, ETensorDType::FP32}}) {
        for (const auto& routing : {RoutingCase{{2, 1, 3, 2}, -1},
                                    RoutingCase{{2, 1, 3, 2}, 1},
                                    RoutingCase{{2, 0, 3, 0}, -1},
                                    RoutingCase{{0, 0, 0, 0}, -1}}) {
            const std::vector<int> off = offsets_of(routing.counts);
            const long tokens = off.back();
            const bool have_empty =
                std::any_of(routing.counts.begin(), routing.counts.end(), [](int c) { return c == 0; });
            DeviceTensor input(types.operands, {tokens, kCols}), dout(types.operands, {tokens, kRows});
            input.upload(random_values(input.host.size(), 0.25f, rng));
            auto dy = random_values(dout.host.size(), 0.25f, rng);
            if (routing.zero_dout_expert >= 0) {
                const int e = routing.zero_dout_expert;
                std::fill(dy.begin() + off[e] * kRows, dy.begin() + off[e + 1] * kRows, 0.0f);
            }
            dout.upload(dy);
            const auto grad = reference(dout, input, off);
            for (const bool accumulate : {false, true}) {
                INFO("grad " << (types.grad == ETensorDType::BF16 ? "bf16" : "fp32") << ", operands "
                             << (types.operands == ETensorDType::BF16 ? "bf16" : "fp32") << ", tokens " << tokens
                             << ", zero-dY expert " << routing.zero_dout_expert << ", accumulate " << accumulate);
                // Without empty experts the overwrite path skips the zero-fill.
                CHECK(dsl::moe_weight_grad_needs_zero(off.data(), kExperts, accumulate) == (!accumulate && have_empty));
                DeviceTensor dw(types.grad, {kExperts, kRows, kCols});
                dw.upload(accumulate ? random_values(dw.host.size(), 1.0f, rng) : nan_values(dw.host.size()));
                const auto initial = dw.host;
                run(h, dw, dout.view(), input.view(), off, accumulate);
                std::vector<double> expected = grad;
                if (accumulate)
                    for (std::size_t i = 0; i < expected.size(); ++i)
                        expected[i] += initial[i];
                const auto actual = dw.read();
                check_close(actual, expected);
                check_empty_experts(actual, off, accumulate ? initial : std::vector<float>(initial.size(), 0.0f));
            }
        }
    }
}

TEST_CASE("Base expert weight gradients reuse a streamed gradient slot", "[moe][gemm][wgrad-zero]") {
    // Under cpu_training, block gradients take accumulate=false and are bound to two rotating GPU slots
    // that are never cleared, so each layer writes over the gradient of the layer two before it.
    if (!have_cuda_device()) SKIP("no CUDA device");
    Handles h;
    std::mt19937 rng(20261030);
    DeviceTensor slot(ETensorDType::FP32, {kExperts, kRows, kCols});
    slot.upload(nan_values(slot.host.size()));
    for (const auto& counts :
         {std::vector<int>{3, 1, 2, 2}, std::vector<int>{1, 4, 2, 1}, std::vector<int>{0, 5, 0, 3}}) {
        const std::vector<int> off = offsets_of(counts);
        const long tokens = off.back();
        DeviceTensor input(ETensorDType::BF16, {tokens, kCols}), dout(ETensorDType::BF16, {tokens, kRows});
        input.upload(random_values(input.host.size(), 0.25f, rng));
        dout.upload(random_values(dout.host.size(), 0.25f, rng));
        INFO("layer with expert tokens " << counts[0] << "," << counts[1] << "," << counts[2] << "," << counts[3]);
        run(h, slot, dout.view(), input.view(), off, false);
        const auto actual = slot.read();
        check_close(actual, reference(dout, input, off));
        check_empty_experts(actual, off, std::vector<float>(actual.size(), 0.0f));
    }
}

TEST_CASE("Base expert weight gradients of FFN tiles add up to the untiled gradient", "[moe][gemm][wgrad-zero]") {
    // compiled_ops_tiled_mlp.cpp runs the backward once per row tile: narrowed dY and X, host offsets
    // clamped to the tile, accumulate only after the first tile. The first tile decides between the
    // overwrite and the zero-fill path.
    if (!have_cuda_device()) SKIP("no CUDA device");
    Handles h;
    std::mt19937 rng(20261031);
    struct TileCase {
        std::vector<int> counts;
        long tile;
        bool first_tile_routes_all;
    };
    for (const auto& tiling : {TileCase{{3, 0, 4, 2}, 4, false}, TileCase{{1, 2, 1, 3}, 5, true}}) {
        const std::vector<int> off = offsets_of(tiling.counts);
        const long tokens = off.back();
        DeviceTensor input(ETensorDType::BF16, {tokens, kCols}), dout(ETensorDType::BF16, {tokens, kRows});
        input.upload(random_values(input.host.size(), 0.25f, rng));
        dout.upload(random_values(dout.host.size(), 0.25f, rng));
        DeviceTensor dw(ETensorDType::FP32, {kExperts, kRows, kCols});
        dw.upload(nan_values(dw.host.size()));
        INFO("tile " << tiling.tile << " over " << tokens << " tokens");
        for (long start = 0; start < tokens; start += tiling.tile) {
            const long count = std::min(tiling.tile, tokens - start);
            std::vector<int> tile_off(kExperts + 1);
            for (int e = 0; e <= kExperts; ++e)
                tile_off[e] = static_cast<int>(std::clamp<long>(off[e] - start, 0, count));
            const bool accumulate = start > 0;
            if (!accumulate)
                CHECK(dsl::moe_weight_grad_needs_zero(tile_off.data(), kExperts, false) ==
                      !tiling.first_tile_routes_all);
            run(h, dw, dout.row_window(start, count), input.row_window(start, count), tile_off, accumulate);
        }
        check_close(dw.read(), reference(dout, input, off));
    }
}
