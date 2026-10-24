#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cmath>
#include <array>
#include <memory>
#include <vector>

#include "kernels/kernels.h"
#include "kernels/column_scale.h"
#include "runtime/lora/lora_model_utils.h"
#include "utilities/tensor.h"

namespace {
// Member functions check through require_cuda rather than REQUIRE: CUDA 13.0's cudafe++ (the DGX
// Spark's toolkit) never finishes on some REQUIREs inside member functions of this file's classes.
void require_cuda(cudaError_t status) { REQUIRE(status == cudaSuccess); }

struct DeviceTensor {
    Tensor tensor;
    DeviceTensor(ETensorDType dtype, std::vector<long> shape) {
        long count = 1;
        for (long n : shape) count *= n;
        void* data = nullptr;
        require_cuda(cudaMalloc(&data, count * get_dtype_size(dtype)));
        tensor = Tensor::from_pointer(static_cast<std::byte*>(data), 0, dtype, shape);
        require_cuda(cudaMemset(data, 0, tensor.bytes()));
    }
    ~DeviceTensor() { cudaFree(tensor.Data); }
    template <typename T> void put(const std::vector<T>& values) {
        REQUIRE(values.size() * sizeof(T) == tensor.bytes());
        require_cuda(cudaMemcpy(tensor.Data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice));
    }
    template <typename T> std::vector<T> get() {
        std::vector<T> result(tensor.nelem());
        require_cuda(cudaMemcpy(result.data(), tensor.Data, tensor.bytes(), cudaMemcpyDeviceToHost));
        return result;
    }
};
bool have_gpu() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}
}

TEST_CASE("Random initialization preserves storage after a scalar or odd-length parameter", "[shared-generation][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    for (int count : {0, 1, 3, 5, 13}) {
        DeviceTensor data(ETensorDType::BF16, {count + 8});
        data.put(std::vector<nv_bfloat16>(count + 8, nv_bfloat16(42.f)));
        fill_normal(data.tensor, count, 0.f, .02f, 42, 0, nullptr);
        const auto values = data.get<nv_bfloat16>();
        for (int i = 0; i < count; ++i) REQUIRE(std::isfinite(float(values[i])));
        for (int i = count; i < values.size(); ++i) REQUIRE(float(values[i]) == 42.f);
    }
}

TEST_CASE("Per-channel scales and small embedding tables support arbitrary widths", "[shared-generation][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    for (int C : {1, 3, 32, 128}) {
        constexpr int N = 13, V = 7;
        INFO("channels=" << C);
        DeviceTensor data(ETensorDType::BF16, {N, C}), scale(ETensorDType::BF16, {C});
        DeviceTensor out(ETensorDType::BF16, {N, C}), grad(ETensorDType::BF16, {C});
        std::vector<nv_bfloat16> x(N * C), s(C);
        for (int i = 0; i < x.size(); ++i) x[i] = static_cast<nv_bfloat16>((i % 17 - 8) * .0625f);
        for (int i = 0; i < C; ++i) s[i] = static_cast<nv_bfloat16>((i % 5 + 1) * .25f);
        data.put(x); scale.put(s);
        column_scale(out.tensor, data.tensor, scale.tensor, N, C, nullptr);
        const auto y = out.get<nv_bfloat16>();
        for (int i = 0; i < y.size(); ++i) REQUIRE(float(y[i]) == float(nv_bfloat16(float(x[i]) * float(s[i % C]))));
        column_scale_gradient(grad.tensor, data.tensor, out.tensor, N, C, nullptr);
        const auto ds = grad.get<nv_bfloat16>();
        for (int c = 0; c < C; ++c) {
            float expected = 0;
            for (int row = 0; row < N; ++row) expected += float(x[row * C + c]) * float(y[row * C + c]);
            REQUIRE(float(ds[c]) == float(nv_bfloat16(expected)));
        }
        DeviceTensor table(ETensorDType::BF16, {V, C}), indices(ETensorDType::INT32, {N});
        std::vector<nv_bfloat16> weights(V * C);
        std::vector<int> ids(N);
        for (int i = 0; i < weights.size(); ++i) weights[i] = static_cast<nv_bfloat16>(i * .03125f);
        for (int i = 0; i < N; ++i) ids[i] = (i * 3) % V;
        table.put(weights); indices.put(ids);
        encoder_forward(out.tensor, indices.tensor, table.tensor, std::nullopt, 1, N, C, V, nullptr);
        const auto embedded = out.get<nv_bfloat16>();
        for (int i = 0; i < N * C; ++i) REQUIRE(float(embedded[i]) == float(weights[ids[i / C] * C + i % C]));
    }
}

TEST_CASE("Router bias gradients accumulate with narrow and odd channel counts", "[shared-generation][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    cudaDeviceProp prop{};
    REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
    for (int C : {1, 3, 4, 7, 32, 64}) {
        INFO("channels=" << C);
        constexpr int N = 17;
        DeviceTensor input(ETensorDType::BF16, {N, C}), bias(ETensorDType::BF16, {C});
        DeviceTensor scratch(ETensorDType::FP32, {get_bias_backward_scratch_size(ETensorDType::BF16, C, prop) / 4});
        std::vector<nv_bfloat16> x(N * C), initial(C, nv_bfloat16(.5f));
        for (int i = 0; i < x.size(); ++i) x[i] = static_cast<nv_bfloat16>((i % 19 - 9) * .0625f);
        input.put(x); bias.put(initial);
        backward_bias(bias.tensor, input.tensor, nullptr, nullptr, scratch.tensor, 1, N, C, prop, nullptr);
        const auto actual = bias.get<nv_bfloat16>();
        for (int c = 0; c < C; ++c) {
            float sum = .5f;
            for (int row = 0; row < N; ++row) sum += float(x[row * C + c]);
            REQUIRE(float(actual[c]) == float(nv_bfloat16(sum)));
        }
    }
}

TEST_CASE("MRoPE rotates the prefix and preserves its tail in forward and backward", "[qwen35][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    constexpr int B = 2, T = 7, HQ = 2, HK = 1, HEADS = HQ + 2 * HK, D = 64, POS = 32;
    for (int rotary : {16, 64}) {
        for (int planes : {1, 3}) {
            for (bool inplace : {false, true}) {
                INFO("rotary=" << rotary << " planes=" << planes << " inplace=" << inplace);
                DeviceTensor input(ETensorDType::FP32, {B, T, HEADS, D});
                DeviceTensor output(ETensorDType::FP32, {B, T, HEADS, D});
                DeviceTensor freq(ETensorDType::FP32, {POS, rotary});
                DeviceTensor position(ETensorDType::INT32, {planes, B, T});
                std::vector<float> x(input.tensor.nelem()), f(freq.tensor.nelem());
                std::vector<int> p(planes * B * T);
                for (int i = 0; i < x.size(); ++i) x[i] = std::sin(i * .173f);
                for (int i = 0; i < p.size(); ++i) p[i] = (i * 3 + i / T) % POS;
                for (int t = 0; t < POS; ++t) {
                    for (int d = 0; d < rotary / 2; ++d) {
                        const float angle = t * std::pow(10000.f, -2.f * d / rotary);
                        f[t * rotary + 2 * d] = std::cos(angle);
                        f[t * rotary + 2 * d + 1] = std::sin(angle);
                    }
                }
                freq.put(f); position.put(p);
                for (bool backward : {false, true}) {
                    INFO("backward=" << backward);
                    input.put(x);
                    Tensor& dst = inplace ? input.tensor : output.tensor;
                    auto kernel = backward ? mrope_backward : mrope_forward;
                    kernel(dst, input.tensor, freq.tensor, position.tensor.get<int>(), planes,
                           3, 2, 2, nullptr, B, T, HQ, HK, D, rotary, nullptr);
                    const auto actual = inplace ? input.get<float>() : output.get<float>();
                    for (int bt = 0; bt < B * T; ++bt) {
                        for (int h = 0; h < HEADS; ++h) {
                            const int base = (bt * HEADS + h) * D;
                            for (int j = 0; j < D; ++j) {
                                double expected = x[base + j];
                                if (h < HQ + HK && j < rotary) {
                                    const int d = j % (rotary / 2);
                                    int axis = 0;
                                    if (planes == 3 && d < 6 && d % 3 != 0) axis = d % 3;
                                    const int fi = p[axis * B * T + bt] * rotary + 2 * d;
                                    const double cosine = f[fi], sine = f[fi + 1] * (backward ? -1 : 1);
                                    const double real = x[base + d], imag = x[base + d + rotary / 2];
                                    expected = j < rotary / 2 ? real * cosine - imag * sine
                                                               : real * sine + imag * cosine;
                                }
                                REQUIRE(actual[base + j] == Catch::Approx(expected).margin(2e-7));
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE("LoRA preserves alpha over rank for packed and sliced projections", "[qwen35][lora][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    constexpr int BT = 3, IN = 64, RANK = 8;
    cublasLtHandle_t handle;
    REQUIRE(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    DeviceTensor workspace(ETensorDType::BYTE, {32 << 20});
    for (int out : {1024, 3584}) {
        for (bool fused : {false, true}) {
            for (float scale : {.5f, 2.f}) {
                INFO("out=" << out << " fused=" << fused << " scale=" << scale);
                const int total = fused ? 2 * out : out;
                const int offset = fused ? out : 0;
                DeviceTensor input(ETensorDType::BF16, {BT, IN}), a(ETensorDType::BF16, {RANK, IN}),
                    b(ETensorDType::BF16, {out, RANK}), output(ETensorDType::BF16, {BT, total}),
                    intermediate(ETensorDType::BF16, {BT, RANK}), scratch(ETensorDType::BF16, {BT, total});
                std::vector<nv_bfloat16> xv(BT * IN), av(RANK * IN), bv(out * RANK), yv(BT * total);
                for (int i = 0; i < xv.size(); ++i) xv[i] = nv_bfloat16(std::sin(i * .13f) * .2f);
                for (int i = 0; i < av.size(); ++i) av[i] = nv_bfloat16(std::cos(i * .03f) * .1f);
                for (int i = 0; i < bv.size(); ++i) bv[i] = nv_bfloat16(std::sin(i * .09f) * .3f);
                for (int i = 0; i < yv.size(); ++i) yv[i] = nv_bfloat16(.125f);
                input.put(xv); a.put(av); b.put(bv); output.put(yv);
                modules::LoRALayerWeights<Tensor> weights;
                weights.A = a.tensor; weights.B = b.tensor;
                modules::detail::apply_lora_contribution(output.tensor, offset, input.tensor, weights,
                    intermediate.tensor, scratch.tensor, scale, 0.f, 0, false, BT, IN, out, RANK,
                    handle, workspace.tensor, nullptr);
                const auto actual = output.get<nv_bfloat16>();
                for (int t = 0; t < BT; ++t) {
                    float low[RANK];
                    for (int r = 0; r < RANK; ++r) {
                        double sum = 0;
                        for (int i = 0; i < IN; ++i) sum += float(xv[t * IN + i]) * float(av[r * IN + i]);
                        low[r] = float(nv_bfloat16(sum));
                    }
                    for (int j = 0; j < total; ++j) {
                        double expected = .125;
                        if (j >= offset && j < offset + out) {
                            double delta = 0;
                            for (int r = 0; r < RANK; ++r) delta += low[r] * float(bv[(j - offset) * RANK + r]);
                            expected += scale * delta;
                        }
                        REQUIRE(float(actual[t * total + j]) == Catch::Approx(expected).margin(.002));
                    }
                }
            }
        }
    }
    REQUIRE(cublasLtDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}

TEST_CASE("Dense LoRA slices combine dx while preserving adapter gradients", "[lora][fused-dx][kernels]") {
    if (!have_gpu()) SKIP("CUDA device required");
    if (std::getenv("SUROGATE_ENABLE_LORA_TRANSPOSED_A_BWD")) SKIP("Transposed-A debug path selected");
    constexpr int BT = 17, IN = 64, RANK = 8;
    cublasLtHandle_t handle;
    REQUIRE(cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS);
    cudaStream_t stream;
    require_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceTensor workspace(ETensorDType::BYTE, {32 << 20});
    struct Projection {
        DeviceTensor a, b, reference_a, reference_b, fused_a, fused_b;
        int offset, width;
        Projection(int offset, int width, ETensorDType grad_dtype)
            : a(ETensorDType::BF16, {RANK, IN}),
              b(ETensorDType::BF16, {width, RANK}),
              reference_a(grad_dtype, {RANK, IN}),
              reference_b(grad_dtype, {width, RANK}),
              fused_a(grad_dtype, {RANK, IN}),
              fused_b(grad_dtype, {width, RANK}),
              offset(offset),
              width(width) {
        }
    };
    for (int count : {2, 3}) {
        const std::array<int, 3> widths = count == 3 ? std::array<int, 3>{24, 8, 16} : std::array<int, 3>{16, 16, 0};
        const int total = 16 + widths[0] + widths[1] + widths[2];
        const long packed_elements = static_cast<long>(count) * RANK * (BT + IN);
        for (auto grad_dtype : {ETensorDType::BF16, ETensorDType::FP32}) {
            for (bool accumulate : {false, true}) {
                for (float scaling : {0.5f, 2.0f}) {
                    INFO("count=" << count << " grad_dtype=" << static_cast<int>(grad_dtype)
                                  << " accumulate=" << accumulate << " scaling=" << scaling);
                    DeviceTensor input(ETensorDType::BF16, {BT, IN}), dout(ETensorDType::BF16, {BT, total}),
                        reference_dx(ETensorDType::BF16, {BT, IN}), fused_dx(ETensorDType::BF16, {BT, IN}),
                        initial_dx(ETensorDType::BF16, {BT, IN}), intermediate(ETensorDType::BF16, {BT, RANK}),
                        scratch(ETensorDType::BF16, {packed_elements + 32});
                    std::vector<nv_bfloat16> xv(BT * IN), dyv(BT * total), dxv(BT * IN, nv_bfloat16(0.125f));
                    for (int i = 0; i < xv.size(); ++i)
                        xv[i] = nv_bfloat16((i % 13 - 6) * 0.0625f);
                    for (int i = 0; i < dyv.size(); ++i)
                        dyv[i] = nv_bfloat16((i % 17 - 8) * 0.03125f);
                    input.put(xv);
                    dout.put(dyv);
                    reference_dx.put(dxv);
                    fused_dx.put(dxv);
                    initial_dx.put(dxv);
                    std::vector<std::unique_ptr<Projection>> owned;
                    std::vector<std::vector<nv_bfloat16>> host_a, host_g;
                    modules::detail::LoRABackwardSlice projections[3];
                    int offset = 8;
                    for (int i = 0; i < count; ++i) {
                        owned.push_back(std::make_unique<Projection>(offset, widths[i], grad_dtype));
                        auto& p = *owned.back();
                        std::vector<nv_bfloat16> av(RANK * IN), bv(widths[i] * RANK);
                        for (int j = 0; j < av.size(); ++j)
                            av[j] = nv_bfloat16(((j + i * 7) % 19 - 9) * 0.03125f);
                        for (int j = 0; j < bv.size(); ++j)
                            bv[j] = nv_bfloat16(((j + i * 3) % 11 - 5) * 0.0625f);
                        p.a.put(av);
                        p.b.put(bv);
                        if (grad_dtype == ETensorDType::BF16) {
                            p.reference_a.put(std::vector<nv_bfloat16>(RANK * IN, nv_bfloat16(0.25f)));
                            p.reference_b.put(std::vector<nv_bfloat16>(widths[i] * RANK, nv_bfloat16(0.25f)));
                            p.fused_a.put(std::vector<nv_bfloat16>(RANK * IN, nv_bfloat16(0.25f)));
                            p.fused_b.put(std::vector<nv_bfloat16>(widths[i] * RANK, nv_bfloat16(0.25f)));
                        } else {
                            p.reference_a.put(std::vector<float>(RANK * IN, 0.25f));
                            p.reference_b.put(std::vector<float>(widths[i] * RANK, 0.25f));
                            p.fused_a.put(std::vector<float>(RANK * IN, 0.25f));
                            p.fused_b.put(std::vector<float>(widths[i] * RANK, 0.25f));
                        }
                        host_a.push_back(av);
                        projections[i] =
                            {&p.fused_a.tensor, &p.fused_b.tensor, &p.a.tensor, &p.b.tensor, offset, widths[i]};
                        offset += widths[i];
                    }
                    require_cuda(cudaDeviceSynchronize());
                    for (int i = 0; i < count; ++i) {
                        auto& p = *owned[i];
                        modules::detail::backward_lora_layer(p.reference_a.tensor,
                                                             p.reference_b.tensor,
                                                             reference_dx.tensor,
                                                             dout.tensor,
                                                             p.offset,
                                                             input.tensor,
                                                             p.a.tensor,
                                                             p.b.tensor,
                                                             scaling,
                                                             0.f,
                                                             42 + i,
                                                             true,
                                                             intermediate.tensor,
                                                             scratch.tensor,
                                                             BT,
                                                             IN,
                                                             p.width,
                                                             RANK,
                                                             accumulate,
                                                             handle,
                                                             workspace.tensor,
                                                             stream);
                        require_cuda(cudaStreamSynchronize(stream));
                        host_g.push_back(intermediate.get<nv_bfloat16>());
                    }
                    auto fused = [&](float dropout = 0.f, Tensor* scratch_override = nullptr) {
                        return modules::detail::backward_lora_slices_fused_dx(projections,
                                                                              count,
                                                                              fused_dx.tensor,
                                                                              dout.tensor,
                                                                              input.tensor,
                                                                              scaling,
                                                                              dropout,
                                                                              true,
                                                                              intermediate.tensor,
                                                                              scratch_override ? *scratch_override
                                                                                               : scratch.tensor,
                                                                              BT,
                                                                              IN,
                                                                              RANK,
                                                                              accumulate,
                                                                              handle,
                                                                              workspace.tensor,
                                                                              stream);
                    };
                    REQUIRE(fused());
                    require_cuda(cudaStreamSynchronize(stream));
                    const auto actual = fused_dx.get<nv_bfloat16>();
                    const auto actual_bits = fused_dx.get<std::uint16_t>();
                    const auto sequential = reference_dx.get<nv_bfloat16>();
                    for (int row = 0; row < BT; ++row) {
                        for (int col = 0; col < IN; ++col) {
                            double expected = float(dxv[row * IN + col]);
                            for (int i = 0; i < count; ++i)
                                for (int r = 0; r < RANK; ++r)
                                    expected += float(host_g[i][row * RANK + r]) * float(host_a[i][r * IN + col]);
                            REQUIRE(float(actual[row * IN + col]) == Catch::Approx(expected).margin(0.002));
                            // The existing path rounds dx after every projection; the fused
                            // reduction rounds once, so equality is numerical rather than bitwise.
                            REQUIRE(float(actual[row * IN + col]) ==
                                    Catch::Approx(float(sequential[row * IN + col])).margin(0.006));
                        }
                    }
                    for (const auto& p : owned) {
                        if (grad_dtype == ETensorDType::BF16) {
                            REQUIRE(p->fused_a.get<std::uint16_t>() == p->reference_a.get<std::uint16_t>());
                            REQUIRE(p->fused_b.get<std::uint16_t>() == p->reference_b.get<std::uint16_t>());
                        } else {
                            REQUIRE(p->fused_a.get<float>() == p->reference_a.get<float>());
                            REQUIRE(p->fused_b.get<float>() == p->reference_b.get<float>());
                        }
                    }

                    // Unsupported cases must decline before touching any output.
                    REQUIRE_FALSE(fused(0.1f));
                    Tensor too_small = scratch.tensor;
                    too_small.Sizes[0] = packed_elements - 1;
                    REQUIRE_FALSE(fused(0.f, &too_small));
                    auto saved = projections[1];
                    projections[1].offset = projections[0].offset;
                    REQUIRE_FALSE(fused());
                    projections[1] = saved;
                    projections[1].dA = projections[0].dA;
                    REQUIRE_FALSE(fused());
                    projections[1] = saved;
                    Tensor different_rank = *projections[1].A;
                    different_rank.Sizes[0] = RANK / 2;
                    projections[1].A = &different_rank;
                    REQUIRE_FALSE(fused());
                    projections[1] = saved;
                    Tensor aliased_intermediate = Tensor::from_pointer(scratch.tensor.Data,
                                                                       scratch.tensor.Device,
                                                                       scratch.tensor.DType,
                                                                       std::array<long, 2>{BT, RANK});
                    REQUIRE_FALSE(modules::detail::backward_lora_slices_fused_dx(projections,
                                                                                 count,
                                                                                 fused_dx.tensor,
                                                                                 dout.tensor,
                                                                                 input.tensor,
                                                                                 scaling,
                                                                                 0.f,
                                                                                 true,
                                                                                 aliased_intermediate,
                                                                                 scratch.tensor,
                                                                                 BT,
                                                                                 IN,
                                                                                 RANK,
                                                                                 accumulate,
                                                                                 handle,
                                                                                 workspace.tensor,
                                                                                 stream));
                    require_cuda(cudaStreamSynchronize(stream));
                    REQUIRE(fused_dx.get<std::uint16_t>() == actual_bits);

                    if (count == 3 && !accumulate && grad_dtype == ETensorDType::BF16 && scaling == 0.5f) {
                        // All tensors, handles and algorithms are warm before capture.
                        cudaGraph_t graph = nullptr;
                        cudaGraphExec_t executable = nullptr;
                        require_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
                        require_cuda(cudaMemcpyAsync(fused_dx.tensor.Data,
                                                     initial_dx.tensor.Data,
                                                     initial_dx.tensor.bytes(),
                                                     cudaMemcpyDeviceToDevice,
                                                     stream));
                        const bool captured = fused();
                        require_cuda(cudaStreamEndCapture(stream, &graph));
                        REQUIRE(captured);
                        require_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                        require_cuda(cudaGraphLaunch(executable, stream));
                        require_cuda(cudaStreamSynchronize(stream));
                        REQUIRE(fused_dx.get<nv_bfloat16>().size() == actual.size());
                        const auto replay = fused_dx.get<nv_bfloat16>();
                        for (int i = 0; i < replay.size(); ++i)
                            REQUIRE(float(replay[i]) == float(actual[i]));
                        require_cuda(cudaGraphExecDestroy(executable));
                        require_cuda(cudaGraphDestroy(graph));
                    }
                }
            }
        }
    }
    require_cuda(cudaStreamDestroy(stream));
    REQUIRE(cublasLtDestroy(handle) == CUBLAS_STATUS_SUCCESS);
}
