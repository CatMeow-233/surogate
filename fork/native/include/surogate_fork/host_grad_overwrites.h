// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// The host gradient buffers that the first micro-step of an optimizer step overwrites (fork ledger T13).
//
// Under cpu_training, DslGradStore::start_micro_step(0) used to memset every host gradient on the rank thread
// (16.4 GB for an 8B model, one thread), although the first micro-step's D2H copies (offload_layer_grads for
// block gradients, offload_non_block_grads for the rest) overwrite each of them in full before anything reads
// it. DslGradStore now records each such copy as it enqueues it and zeroes only the buffers that no copy
// covered, before their first reader: the host additions of the next micro-step (start_micro_step(k > 0)) or
// the CPU optimizer (wait_all_offloads). The host gradients end bit-identical to the memset path: a covered
// buffer holds the copied bytes either way, and any other buffer is zero before its first reader either way.
// Zeroing an uncovered buffer cannot race with the GPU: no copy into it has been enqueued (that would have
// recorded it), and its readers are enqueued only after the zeroing.
//
// The class is plain host code (no CUDA), so the contract test compiles it without a GPU
// (fork/native/openmp-contract/host_overwrites_check.cpp).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace surogate_fork {

class HostGradOverwrites {
public:
    // Registers a host buffer of `bytes` bytes. Names that share one buffer (a tied lm_head and the embedding,
    // ledger T10) register it once.
    void add(void* data, std::size_t bytes) {
        if (!data || bytes == 0) return;
        const auto [it, inserted] = mIndex.emplace(data, mBuffers.size());
        if (inserted) {
            mBuffers.push_back({data, bytes, false});
        } else {
            mBuffers[it->second].bytes = std::max(mBuffers[it->second].bytes, bytes);
        }
    }

    // The first micro-step of an optimizer step starts: nothing has been overwritten in it yet.
    void begin_step() {
        for (auto& buffer : mBuffers)
            buffer.written = false;
        mPending = true;
    }

    // A copy of `bytes` bytes to `data` has been enqueued. It covers a registered buffer that starts at `data`
    // and is no longer than `bytes`; any other copy covers nothing.
    void overwritten(const void* data, std::size_t bytes) {
        const auto it = mIndex.find(data);
        if (it != mIndex.end() && bytes >= mBuffers[it->second].bytes) mBuffers[it->second].written = true;
    }

    // Zeroes every buffer that no copy covered since begin_step(), at most once per step, and returns the number
    // of bytes it zeroed.
    std::size_t zero_rest() {
        if (!mPending) return 0;
        mPending = false;
        std::size_t zeroed = 0;
        for (const auto& buffer : mBuffers) {
            if (buffer.written) continue;
            std::memset(buffer.data, 0, buffer.bytes);
            zeroed += buffer.bytes;
        }
        return zeroed;
    }

    [[nodiscard]] std::size_t buffers() const {
        return mBuffers.size();
    }

private:
    struct Buffer {
        void* data;
        std::size_t bytes;
        bool written;
    };
    std::unordered_map<const void*, std::size_t> mIndex;
    std::vector<Buffer> mBuffers;
    bool mPending = false;  // begin_step() ran and zero_rest() has not
};

}  // namespace surogate_fork
