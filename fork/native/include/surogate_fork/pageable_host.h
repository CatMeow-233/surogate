// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Host memory the CUDA runtime does not know about (fork ledger T21).
#pragma once

#include <cuda_runtime.h>

namespace surogate_fork {

//! True when `data` is pageable host memory, such as the std::vector moments of the cpu_training AdamW step.
//! SafeTensorWriter copied every tensor with cudaMemcpy(..., cudaMemcpyDeviceToHost), whose source must be device
//! memory or memory registered with CUDA, and which fails outright on a host without a driver; it copies such
//! bytes with memcpy instead. Device >= 0 (device or managed memory) is never pageable. Pinned and write-combined
//! host memory (Device < 0, registered) keep upstream's cudaMemcpy: a CPU read of write-combined memory is slow.
//! When the query fails (no driver, or a runtime that rejects unregistered pointers) the memory cannot be
//! registered memory that only the GPU should read, and memcpy reads any host memory correctly.
inline bool is_pageable_host(const void* data, int device) {
    if (device >= 0 || data == nullptr) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, data) != cudaSuccess) {
        (void)cudaGetLastError();
        return true;
    }
    return attributes.type == cudaMemoryTypeUnregistered;
}

}  // namespace surogate_fork
