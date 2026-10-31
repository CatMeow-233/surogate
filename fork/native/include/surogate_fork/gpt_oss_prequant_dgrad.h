// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Opt-in switch for the frozen-expert input gradient of GPT-OSS MoE layers trained with LoRA on a prequantized
// checkpoint (fork ledger T25). Read by dispatch_moe_grouped_gemm_gate_up_backward and
// dispatch_moe_grouped_gemm_down_backward (csrc/src/runtime/ops/moe_grouped_gemm_{gate_up,down}.cpp).
//
// Upstream skipped both base dgrad GEMMs for that case (7a8d5fee added the skip with SUROGATE_PREQUANT_BASE_GRAD=1
// as the way back; 1181c715 dropped the variable and made the skip unconditional) and zeroed d_input, so the
// grouped LoRA backward was the only term left in it. The down backward's d_input is d_act, from which the
// activation backward forms the gate_up gradient: the expert gate_up LoRA then saw only the down-LoRA term (exactly
// zero while the down LoRA B is at its zero init), and the MoE branch passed nothing but that term to the
// attention LoRA and the earlier layers. Every other architecture, and GPT-OSS on a BF16 checkpoint, runs the same
// dgrad on the same dequantized BF16 expert weights. The fork computes it by default; setting
// SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD to a non-empty value other than "0" restores upstream's
// approximation (two grouped GEMMs fewer per MoE layer, wrong gradients as described above).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace surogate_fork {

inline constexpr const char* kGptOssPrequantSkipExpertDgradEnv = "SUROGATE_GPT_OSS_PREQUANT_SKIP_EXPERT_DGRAD";

// The convention of upstream's SUROGATE_PREQUANT_BASE_GRAD: unset, empty and "0" are off, anything else is on.
inline bool env_flag_value_is_on(const char* value) {
    return value != nullptr && *value != '\0' && std::strcmp(value, "0") != 0;
}

// Read once per process, so every layer and micro-step of a run takes the same branch. The first call that finds
// the switch on says so on stderr, because it changes gradients rather than only speed.
inline bool gpt_oss_prequant_skip_expert_dgrad_requested() {
    static const bool requested = [] {
        const bool on = env_flag_value_is_on(std::getenv(kGptOssPrequantSkipExpertDgradEnv));
        if (on) {
            std::fprintf(stderr,
                         "[surogate fork] %s is set: GPT-OSS prequantized LoRA skips the frozen-expert input "
                         "gradient (approximate gradients, upstream behaviour)\n",
                         kGptOssPrequantSkipExpertDgradEnv);
        }
        return on;
    }();
    return requested;
}

}  // namespace surogate_fork
