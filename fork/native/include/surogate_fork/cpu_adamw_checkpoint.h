// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Checkpoints of the CPU AdamW moments of cpu_training full fine-tuning (fork ledger T21).
//
// Under cpu_training, AdamW8BitOptimizer::step routes every step to step_cpu_streaming, which keeps FP32 first
// and second moments in host memory (optimizers::CPUAdamWState: one flat std::vector each, laid out over the
// masters it updates). Upstream exposed neither through momentum_container()/variance_container(), so adam.m and
// adam.v held no tensors, and on resume prepare_for_checkpoint_load allocated the GPU path's 8-bit state on the
// device (unused, ~17.4 GB at 8B) while the CPU moments restarted at zero. This module owns the layout both the
// step and the resume use, exposes the moments to save_checkpoint/load_checkpoint, and refuses a checkpoint that
// lacks them instead of letting the reader skip the missing names.
//
// Format v1, one copy per local rank (adam.m.shard_<rank>_of_<n>.safetensors and adam.v.shard_...; data-parallel
// ranks hold identical moments, and each writes and reads its own file, as the weights shards do):
//   cpu_adamw.m.<param>   FP32, the master's shape: the first moment of that master
//   cpu_adamw.v.<param>   FP32, the master's shape: the second moment
//   cpu_adamw.format      INT32 [1] = kFormatVersion, in both files
// <param> is the DslGradStore::param_names() name of each master the step updates, in that order. A master two
// names share (a tied lm_head is the embedding's master, ledger T10) has one entry, under the first name. The
// bias-correction step is not stored: the trainer passes step + 1 for the resumed step, as before.
//
// Compatibility rules (fork/docs/performance.md, "cpu_training 的 AdamW 状态 checkpoint"):
//   - The names are disjoint from the GPU path's adamw8bit.state1/scales1/state2/scales2 (BYTE and FP16), so a
//     cpu_training resume from a GPU-training checkpoint finds none of its names and is refused, not silently
//     zeroed. Only that direction is checked: require_complete() returns at once for the GPU optimizers'
//     containers, so a GPU-training resume from a cpu_training checkpoint skips the cpu_adamw.* names and starts
//     the 8-bit state at zero without a warning, as upstream did. A per-parameter entry makes a layout change
//     (other parameters, another order or size) a missing name or a shape error rather than a silent misread; the
//     reader already throws on a dtype or shape mismatch.
//   - require_complete() refuses a file that lacks any expected name, which covers checkpoints written by
//     upstream or by a fork build without T21 (header-only adam.m/adam.v, or the unused 8-bit device state that
//     every save after an earlier resume wrote). Their moments are lost: SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1
//     resumes anyway, with zero moments for the missing tensors, which is what upstream did.
//   - A file whose cpu_adamw.format holds another version is refused whatever the switches say.
//   - A later format (for example 8-bit CPU state, roadmap #9) must use other names or a new version, and must
//     convert or refuse v1 files.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "runtime/optimizers/cpu_adamw.h"
#include "utilities/tensor.h"
#include "utilities/tensor_container.h"

namespace surogate_fork::cpu_adamw_checkpoint {

inline constexpr std::int32_t kFormatVersion = 1;
inline constexpr const char* kFormatTensor = "cpu_adamw.format";
inline constexpr const char* kMomentPrefix = "cpu_adamw.m.";
inline constexpr const char* kVariancePrefix = "cpu_adamw.v.";
//! "0": checkpoints hold no CPU AdamW moments (upstream behaviour; 8 bytes per parameter less on disk).
inline constexpr const char* kSaveStateEnv = "SUROGATE_FORK_CPU_ADAMW_SAVE_STATE";
//! "1": a resume from a checkpoint without (all of) them warns and starts the missing moments at zero.
inline constexpr const char* kAllowMissingStateEnv = "SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE";

//! A trainable parameter and its master, in DslGradStore::param_names() order.
struct Param {
    std::string name;
    const Tensor* master = nullptr;
};

//! One master the CPU step updates, and where its moments live in the flat m/v arrays.
struct Slot {
    std::string name;
    std::vector<long> shape;
    std::size_t offset = 0;
    std::size_t n = 0;
};

//! The masters the CPU step updates, in step order, at consecutive offsets. A master whose storage an earlier
//! parameter already has (a tied lm_head, T10) is skipped, so it takes one step with the gradient both names share.
std::vector<Slot> build_layout(const std::vector<Param>& params);

//! One moment (m or v) as FP32 host tensors, one per slot, plus the format tag.
class MomentContainer final : public ITensorContainer {
public:
    explicit MomentContainer(const char* prefix);

    void bind(const std::vector<Slot>* slots, std::vector<float>* data);
    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

    //! During a load the container always lists its tensors; otherwise only when saving is enabled.
    void set_loading(bool loading);
    //! The format tag: kFormatVersion when bound, 0 from the start of a load until the file provides it.
    [[nodiscard]] std::int32_t format() const {
        return mFormat;
    }
    void set_format(std::int32_t format) {
        mFormat = format;
    }
    //! require_complete() has checked this container against a file since the load began.
    [[nodiscard]] bool checked() const {
        return mChecked;
    }
    void set_checked(bool checked) {
        mChecked = checked;
    }

private:
    const char* mPrefix;
    const std::vector<Slot>* mSlots = nullptr;
    std::vector<float>* mData = nullptr;
    bool mLoading = false;
    bool mChecked = false;
    std::int32_t mFormat = kFormatVersion;
};

//! The checkpoint side of an AdamW8BitOptimizer's CPU state.
class CpuAdamWCheckpoint {
public:
    CpuAdamWCheckpoint();

    //! Lays the moments out over `params` and allocates them, zero-filled, in `state` (what the first CPU step
    //! did), then exposes them. Runs once: later calls return at once.
    void initialize(optimizers::CPUAdamWState& state, const std::vector<Param>& params);
    [[nodiscard]] bool active() const {
        return mState != nullptr;
    }
    [[nodiscard]] const std::vector<Slot>& slots() const {
        return mSlots;
    }
    [[nodiscard]] MomentContainer* momentum() {
        return &mMomentum;
    }
    [[nodiscard]] MomentContainer* variance() {
        return &mVariance;
    }

    //! Before the optimizer files are read: clear the format tags so that finish_load() sees what the files held.
    void begin_load();
    //! After they were read: throws on another format version, and on files without CPU AdamW state unless
    //! SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1 (then warns; the moments stay zero).
    void finish_load();

private:
    optimizers::CPUAdamWState* mState = nullptr;
    std::vector<Slot> mSlots;
    MomentContainer mMomentum;
    MomentContainer mVariance;
    bool mLoadPending = false;
};

//! load_checkpoint calls this before reading each optimizer file into `container`. For a MomentContainer it
//! checks that `file_name` has every tensor the container expects (SafeTensorsReader::load_tensors reads only the
//! names both have and skips the rest without a word); a missing one throws, or with
//! SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE=1 warns. Any other container (the GPU optimizers') is not checked.
void require_complete(const std::string& file_name, ITensorContainer& container);

//! SUROGATE_FORK_CPU_ADAMW_SAVE_STATE is not "0".
[[nodiscard]] bool save_state_enabled();
//! SUROGATE_FORK_CPU_ADAMW_ALLOW_MISSING_STATE is set and not "0".
[[nodiscard]] bool missing_state_allowed();

}  // namespace surogate_fork::cpu_adamw_checkpoint
