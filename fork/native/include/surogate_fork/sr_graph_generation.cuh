// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0

// Fresh stochastic-rounding noise on CUDA graph replays (fork ledger T24).
//
// The NVFP4 SR kernels take their seed as a host value. The training step is captured into a CUDA graph once
// (py_train.cpp, train_step_graphed) and replayed every step, so upstream's kernels drew the same noise for
// every element at every step after the capture: per element the rounding becomes a fixed threshold, and the
// expectation over steps is no longer the input. This header gives each including translation unit a device
// counter that the captured graph advances after every SR launch, so a replay draws noise its predecessors did
// not. Outside a capture nothing changes: the counter is neither read nor advanced and the kernels compute
// exactly what they computed before, so eager runs keep upstream's seeds (and their reproducibility on resume).
// SUROGATE_NVFP4_SR_GRAPH_LEGACY=1 turns it off (graphs replay the captured noise again).
//
// Use from a launcher:
//     const bool graph = surogate_fork::sr_graph::use_generation(stream);
//     kernel<<<..., stream>>>(..., graph);  // the kernel reads generation(graph) or curand_offset(graph)
//     surogate_fork::sr_graph::advance(graph, stream);
// The counter is per translation unit (internal linkage, no relocatable device code needed); each kernel of a
// unit sees a value that differs from every earlier captured launch of that unit. It is not checkpointed.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace surogate_fork::sr_graph {

namespace {

// Number of captured SR launches of this translation unit that have run (wraps after 2^32).
__device__ uint32_t g_generation = 0;

__global__ void advance_generation_kernel() {
    atomicAdd(&g_generation, 1u);
}

}  // namespace

/// SUROGATE_NVFP4_SR_GRAPH_LEGACY=1: replays reuse the noise drawn at capture time, as upstream.
inline bool fresh_noise_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("SUROGATE_NVFP4_SR_GRAPH_LEGACY");
        return !(v && std::string_view(v) == "1");
    }();
    return enabled;
}

/// True when `stream` is being captured into a CUDA graph and the switch is on.
inline bool use_generation(cudaStream_t stream) {
    if (!fresh_noise_enabled()) return false;
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
        (void)cudaGetLastError();  // the launch that follows reports any real stream error
        return false;
    }
    return status == cudaStreamCaptureStatusActive;
}

/// The generation a kernel launched with `use` mixes into its random stream: 0 outside captures.
__device__ __forceinline__ uint32_t generation(bool use) {
    return use ? __ldcg(&g_generation) : 0u;
}

/// curand_init offset that moves a per-thread Philox stream to generation(use): offset / 4 is added to the low
/// 64 bits of the Philox counter, so the counter's second word grows by the generation and the streams of two
/// generations are disjoint as long as a thread draws fewer than 2^34 values. 0 outside captures.
__device__ __forceinline__ unsigned long long curand_offset(bool use) {
    return static_cast<unsigned long long>(generation(use)) << 34;
}

/// Enqueue the counter step after a captured SR launch (no-op when `use` is false).
inline void advance(bool use, cudaStream_t stream) {
    if (use) advance_generation_kernel<<<1, 1, 0, stream>>>();
}

}  // namespace surogate_fork::sr_graph
