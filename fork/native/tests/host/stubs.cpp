// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// The two symbols the host checks link against without their sources: the GPU dtype conversions of
// csrc/src/kernels/convert.cu (the reader calls them only when the file's dtype differs from the target's,
// which the checks never ask for) and the stack trace of utilities/crash_handler.cpp (which needs backward-cpp).
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define SUROGATE_FORK_HOST_STUB(signature)                                           \
    void convert_dtype signature {                                                   \
        std::fprintf(stderr, "convert_dtype is not available in the host checks\n"); \
        std::abort();                                                                \
    }
SUROGATE_FORK_HOST_STUB((float*, const nv_bfloat16*, std::size_t))
SUROGATE_FORK_HOST_STUB((nv_bfloat16*, const float*, std::size_t))
SUROGATE_FORK_HOST_STUB((nv_bfloat16*, const half*, std::size_t))
SUROGATE_FORK_HOST_STUB((nv_bfloat16*, const __nv_fp8_e4m3*, std::size_t))
#undef SUROGATE_FORK_HOST_STUB

namespace surogate {
std::string capture_stacktrace(int, int) {
    return "(no stack trace in the host checks)";
}
}  // namespace surogate
