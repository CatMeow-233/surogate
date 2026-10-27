// Fused LoRA input-gradient accumulation, SUROGATE_FUSE_LORA_DX (fork ledger T2). Built into
// integration-tests by fork/native/cmake/fork_native.cmake. The helpers below are copies of the
// ones in csrc/src/testing/kernels/test-qwen35-shared.cu; the anonymous namespace keeps both
// copies local to their translation unit.
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

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
