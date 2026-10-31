// One head geometry's small-T decode instantiations, in the form of upstream's
// csrc/src/serve/ops/launcher/gqa_attention_decode_gqa128_16q2.cu, for a shape the fork
// registers (ledger T16): Gqa128_40q8 (qwen3-14b). The routing switch is
// fork/native/include/surogate_fork/gqa_fork_shapes.h; fork/native/cmake/fork_native.cmake
// compiles this file into sinfer_ops.
#include "ops/launcher/gqa_attention_decode_launch.cuh"
#include "surogate_fork/gqa_fork_shapes.h"

namespace sinfer::ops::detail {

static_assert(surogate_fork::gqa_is_fork_shape(Gqa128_40q8::HeadDim, Gqa128_40q8::QHeads,
                                               Gqa128_40q8::KVHeads),
              "a fork decode unit for a shape surogate_fork::kGqaForkShapes lacks");

template void gqa_attention_small_t_launch_for<Gqa128_40q8, GqaAppendInput>(
    const Tensor&, GqaAppendInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

template void gqa_attention_small_t_launch_for<Gqa128_40q8, GqaCachedInput>(
    const Tensor&, GqaCachedInput, const Tensor&, float, PagedKVBatchLayerView,
    const GqaSmallTInvocation&, GqaExecutionEnvelope, Tensor&, Tensor&, Tensor&, Tensor&,
    cudaStream_t);

} // namespace sinfer::ops::detail
