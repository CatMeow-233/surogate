#pragma once

// The GQA head shapes the fork registers in the serve engine, and the switch that routes them
// (ledger T16).
//
// csrc/src/serve/ops/kernel/gqa_attention_geometry.cuh registers three 128-wide shapes for the fork,
// each with a small-T decode translation unit in fork/native/serve/:
//
//   Gqa128_32q8   32 query heads over 8 KV heads: Qwen3-4B, Qwen3-8B, Llama-3-8B, Mistral-7B
//   Gqa128_40q8   40 over 8: Qwen3-14B
//   Gqa128_32q32  multi-head 32 over 32: Llama-2-7B and other MHA models with a 128-wide head
//
// Before registration the first two ran decode, verify and prompts narrower than the prompt kernel's
// first width (SUROGATE_SERVE_GQA_PROMPT_KERNEL_MIN_COLUMNS, 64) on the generic CUDA-core split-KV
// kernel (gqa_attention_generic.cu) on every GPU but Hopper, and their wider prompts on the prompt
// kernel of the KV geometry (Gqa128_16q8). The third was refused: no registered 128-wide shape
// carried 32 KV heads.
//
// SUROGATE_SERVE_GQA_FORK_SHAPES=1 sends a BF16 or e4m3 cache of these shapes down the registered
// route: the tensor-core small-T kernel, which packs the query group into the MMA rows and reads each
// KV head once for the whole group, with its fixed key partitions and reducer. Its outputs differ
// from the generic kernel's in the low bits (it rounds P to BF16 before P V; the generic kernel keeps
// it in FP32), so the switch is off by default. Off, every shape is treated as it was before the fork
// registered these: the first two take their old routes (gqa_fork_shape_keeps_generic_route), and
// the multi-head shape is refused, as it was, because the fork withholds the (head dim, KV heads)
// pair only it carries from the registry's KV acceptance (gqa_fork_kv_pair_withheld). On, the
// multi-head shape is served: a BF16 or e4m3 cache on the registered route, an int8 cache on the
// generic kernel and the prompt kernel from its first width.
//
// An int8 cache keeps the old routes with the switch on too: the int8 small-T decode kernel serves
// only 256-wide heads with a group of four to eight (kGqaI8DecodeRegistered in
// gqa_attention_decode_launch.cuh) and would throw, where the generic kernel serves int8 at 128.

#include "ops/kernel/gqa_attention_geometry.cuh"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace surogate_fork {

struct GqaShape {
    std::int32_t head_dim;
    std::int32_t q_heads;
    std::int32_t kv_heads;
};

// Kept equal to the fork's lines in SINFER_GQA_FOR_EACH_GEOMETRY: each fork decode translation unit
// asserts its geometry is listed here, and fork/patches/tests/native/test_gqa_fork_shapes.py checks
// the registry against this list.
inline constexpr GqaShape kGqaForkShapes[] = {
    {128, 32, 8},
    {128, 40, 8},
    {128, 32, 32},
};

constexpr bool gqa_is_fork_shape(std::int32_t head_dim, std::int32_t q_heads,
                                 std::int32_t kv_heads) {
    for (const GqaShape& shape : kGqaForkShapes) {
        if (shape.head_dim == head_dim && shape.q_heads == q_heads && shape.kv_heads == kv_heads) {
            return true;
        }
    }
    return false;
}

// Whether a registered shape is to be routed as if it were not registered: a fork shape with the
// switch off, or over an int8 cache. Every other shape answers false, so upstream routing is
// unchanged for it.
constexpr bool gqa_fork_shape_keeps_generic_route(std::int32_t head_dim, std::int32_t q_heads,
                                                  std::int32_t kv_heads, bool int8_cache,
                                                  bool enabled) {
    return gqa_is_fork_shape(head_dim, q_heads, kv_heads) && (!enabled || int8_cache);
}

// SUROGATE_SERVE_GQA_FORK_SHAPES=1 turns the registered route on; unset or any other value leaves it
// off. Read once per process, as the engine's other routing switches are, so a captured decode graph
// and the workspace plan that sized it see the same route.
inline bool gqa_fork_shapes_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_GQA_FORK_SHAPES");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

inline bool gqa_fork_shape_keeps_generic_route(std::int32_t head_dim, std::int32_t q_heads,
                                               std::int32_t kv_heads, bool int8_cache) {
    return gqa_is_fork_shape(head_dim, q_heads, kv_heads) &&
           gqa_fork_shape_keeps_generic_route(head_dim, q_heads, kv_heads, int8_cache,
                                              gqa_fork_shapes_enabled());
}

// Whether a registered shape the fork did not add carries this (head dim, KV heads) pair: the
// registry's KV acceptance (sinfer::ops::gqa_kv_shape_is_registered) as it stands without the fork's
// lines. Derived from the registry rather than listed, so a shape upstream registers later over the
// same pair is accepted whatever the switch says.
inline bool gqa_kv_pair_registered_without_fork(std::int64_t head_dim, std::int64_t kv_heads) {
#define SUROGATE_FORK_GQA_UPSTREAM_KV_MATCH(Name)                                                  \
    if (head_dim == sinfer::ops::Name::HeadDim && kv_heads == sinfer::ops::Name::KVHeads &&       \
        !gqa_is_fork_shape(sinfer::ops::Name::HeadDim, sinfer::ops::Name::QHeads,                 \
                           sinfer::ops::Name::KVHeads)) {                                          \
        return true;                                                                               \
    }
    SINFER_GQA_FOR_EACH_GEOMETRY(SUROGATE_FORK_GQA_UPSTREAM_KV_MATCH)
#undef SUROGATE_FORK_GQA_UPSTREAM_KV_MATCH
    return false;
}

// Whether the wrapper is to refuse a (head dim, KV heads) pair the registry carries: one that only
// fork shapes carry (128 over 32 KV heads, the multi-head shape), with the switch off. Upstream
// refused that pair, so a model with it is refused as before: the workspace plan, every validate
// call and the KV append throw on its geometry. Every other pair answers false, so upstream
// acceptance is unchanged for it; 128 over 8 KV heads is upstream's Gqa128_16q8 pair and is never
// withheld. Read at the wrapper's four KV acceptance checks (supported_attention_shape,
// require_kv_heads, validate_cache, validate_batch_cache) and by upstream's registration contract
// test, which would otherwise demand that every registered shape be accepted.
inline bool gqa_fork_kv_pair_withheld(std::int64_t head_dim, std::int64_t kv_heads) {
    return !gqa_fork_shapes_enabled() && sinfer::ops::gqa_kv_shape_is_registered(head_dim, kv_heads) &&
           !gqa_kv_pair_registered_without_fork(head_dim, kv_heads);
}

} // namespace surogate_fork
