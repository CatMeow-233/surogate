// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace dsl {

// The caller validates the complete [experts, rows, cols] destination shape.
// Its beta=0 GEMMs overwrite every element only when every expert has tokens;
// otherwise the skipped experts still need zero gradients. Accumulation keeps
// their existing gradients and must never clear them.
inline bool moe_weight_grad_needs_zero(const int* offsets, int experts, bool accumulate) {
    if (accumulate) return false;
    for (int e = 0; e < experts; ++e) {
        if (offsets[e + 1] <= offsets[e]) return true;
    }
    return false;
}

}  // namespace dsl
