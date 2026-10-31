// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Checkpoints of the CPU AdamW moments of cpu_training full fine-tuning (fork ledger T21); see the header.

#include "surogate_fork/cpu_adamw_checkpoint.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

#include "utilities/safetensors.h"

namespace surogate_fork::cpu_adamw_checkpoint {

namespace {

//! Unset or empty: `fallback`; "0": false; anything else: true.
bool env_flag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return fallback;
    return std::string_view(value) != "0";
}

//! Throws `message`, or with the opt-in prints it as a warning and returns.
void refuse_or_warn(const std::string& message) {
    if (!missing_state_allowed()) {
        throw std::runtime_error(message + " Set " + std::string(kAllowMissingStateEnv) +
                                 "=1 to resume anyway with zero moments for the missing tensors.");
    }
    std::fprintf(stderr,
                 "[surogate fork T21] WARNING: %s Resuming anyway (%s=1): the missing moments start at zero.\n",
                 message.c_str(),
                 kAllowMissingStateEnv);
}

const char* kNoStateCause =
    " It holds no CPU AdamW state: it was written by GPU training, by an optimizer other than adamw_8bit (adamw "
    "and normuon save no optimizer state), with SUROGATE_FORK_CPU_ADAMW_SAVE_STATE=0, or under cpu_training by a "
    "build without fork ledger T21, whose adam.m/adam.v files hold no moments (no tensors, or the unused 8-bit "
    "device state that every save after an earlier resume wrote). Resuming from it restarts the moments at zero "
    "while bias correction continues from the saved step, so the first updates are several times larger than in "
    "an uninterrupted run.";

}  // namespace

bool save_state_enabled() {
    return env_flag(kSaveStateEnv, true);
}

bool missing_state_allowed() {
    return env_flag(kAllowMissingStateEnv, false);
}

std::vector<Slot> build_layout(const std::vector<Param>& params) {
    std::vector<Slot> slots;
    std::unordered_set<const std::byte*> stepped;
    std::size_t offset = 0;
    for (const auto& param : params) {
        if (param.master == nullptr) {
            throw std::logic_error("cpu_adamw_checkpoint::build_layout: no master for " + param.name);
        }
        const Tensor& master = *param.master;
        if (master.Data && !stepped.insert(master.Data).second) continue;
        Slot slot;
        slot.name = param.name;
        slot.shape.assign(master.Sizes.begin(), master.Sizes.begin() + master.Rank);
        slot.offset = offset;
        slot.n = master.nelem();
        offset += slot.n;
        slots.push_back(std::move(slot));
    }
    return slots;
}

MomentContainer::MomentContainer(const char* prefix)
    : mPrefix(prefix) {
}

void MomentContainer::bind(const std::vector<Slot>* slots, std::vector<float>* data) {
    mSlots = slots;
    mData = data;
    mFormat = kFormatVersion;
}

void MomentContainer::set_loading(bool loading) {
    mLoading = loading;
}

void MomentContainer::iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) {
    if (mSlots == nullptr || mData == nullptr) return;
    if (!mLoading && !save_state_enabled()) return;
    callback(kFormatTensor,
             TensorShard(Tensor::from_pointer(reinterpret_cast<std::byte*>(&mFormat),
                                              /*device=*/-1,
                                              ETensorDType::INT32,
                                              std::array<long, 1>{1})));
    for (const auto& slot : *mSlots) {
        if (slot.offset + slot.n > mData->size()) {
            throw std::logic_error("cpu_adamw_checkpoint: the moments of " + slot.name + " lie outside the state");
        }
        callback(mPrefix + slot.name,
                 TensorShard(Tensor::from_pointer(reinterpret_cast<std::byte*>(mData->data() + slot.offset),
                                                  /*device=*/-1,
                                                  ETensorDType::FP32,
                                                  slot.shape)));
    }
}

CpuAdamWCheckpoint::CpuAdamWCheckpoint()
    : mMomentum(kMomentPrefix),
      mVariance(kVariancePrefix) {
}

void CpuAdamWCheckpoint::initialize(optimizers::CPUAdamWState& state, const std::vector<Param>& params) {
    if (mState != nullptr) return;
    if (state.initialized) {
        throw std::logic_error("cpu_adamw_checkpoint: the CPU AdamW state was allocated without its layout");
    }
    mSlots = build_layout(params);
    std::size_t total = 0;
    for (const auto& slot : mSlots) {
        total += slot.n;
    }
    state.m.resize(total, 0.0f);
    state.v.resize(total, 0.0f);
    state.total_params = total;
    state.initialized = true;
    mState = &state;
    mMomentum.bind(&mSlots, &state.m);
    mVariance.bind(&mSlots, &state.v);
}

void CpuAdamWCheckpoint::begin_load() {
    if (mState == nullptr) {
        throw std::logic_error("cpu_adamw_checkpoint::begin_load before initialize");
    }
    mMomentum.set_format(0);
    mVariance.set_format(0);
    mMomentum.set_loading(true);
    mVariance.set_loading(true);
    mMomentum.set_checked(false);
    mVariance.set_checked(false);
    mLoadPending = true;
}

void CpuAdamWCheckpoint::finish_load() {
    if (!mLoadPending) return;
    mLoadPending = false;
    mMomentum.set_loading(false);
    mVariance.set_loading(false);
    const std::int32_t m_format = mMomentum.format();
    const std::int32_t v_format = mVariance.format();
    mMomentum.set_format(kFormatVersion);
    mVariance.set_format(kFormatVersion);
    for (const std::int32_t format : {m_format, v_format}) {
        if (format != 0 && format != kFormatVersion) {
            throw std::runtime_error("The checkpoint's CPU AdamW state has format version " + std::to_string(format) +
                                     "; this build reads version " + std::to_string(kFormatVersion) + " only.");
        }
    }
    // A file without the tag has been refused (or, opted in, reported) by require_complete(); this covers a
    // caller that read the files without it.
    const bool m_unreported = m_format == 0 && !mMomentum.checked();
    const bool v_unreported = v_format == 0 && !mVariance.checked();
    if (m_unreported || v_unreported) {
        refuse_or_warn(std::string("The checkpoint's ") + (m_unreported ? "adam.m" : "adam.v") + " file has no " +
                       kFormatTensor + "." + kNoStateCause);
    }
}

void require_complete(const std::string& file_name, ITensorContainer& container) {
    auto* moments = dynamic_cast<MomentContainer*>(&container);
    if (moments == nullptr) return;

    std::vector<std::string> expected;
    moments->iterate_tensors(
        [&expected](std::string name, const TensorShard&) { expected.push_back(std::move(name)); });
    if (expected.empty()) return;
    moments->set_checked(true);

    std::unordered_set<std::string> present;
    {
        SafeTensorsReader reader(file_name);
        for (const auto& entry : reader.entries()) {
            present.insert(entry.name());
        }
    }
    std::vector<std::string> missing;
    for (const auto& name : expected) {
        if (!present.contains(name)) missing.push_back(name);
    }
    if (missing.empty()) return;

    std::string message = file_name + " lacks " + std::to_string(missing.size()) + " of the " +
                          std::to_string(expected.size()) + " CPU AdamW tensors this cpu_training run resumes into (";
    constexpr std::size_t kNamed = 3;
    for (std::size_t i = 0; i < missing.size() && i < kNamed; ++i) {
        message += (i ? ", " : "") + missing[i];
    }
    message += missing.size() > kNamed ? ", ...)." : ").";
    if (!present.contains(kFormatTensor)) {
        message += kNoStateCause;
    } else {
        message += " Its CPU AdamW state covers other parameters: another model, or another set of trainable "
                   "parameters.";
    }
    refuse_or_warn(message);
}

}  // namespace surogate_fork::cpu_adamw_checkpoint
