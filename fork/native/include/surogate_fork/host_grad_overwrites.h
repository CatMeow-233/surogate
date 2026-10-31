// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// The host gradient buffers that the first micro-step of an optimizer step overwrites (fork ledger T13).
//
// Under cpu_training, DslGradStore::start_micro_step(0) used to memset every host gradient on the rank thread
// (16.4 GB for an 8B model, one thread), although the first micro-step's D2H copies (offload_layer_grads for
// block gradients, offload_non_block_grads for the rest) overwrite each of them in full before anything reads
// it. DslGradStore now calls before_copy() just before it enqueues each copy into a host gradient, and
// zero_rest() zeroes the buffers that no copy wrote, before their first reader: the host additions of the next
// micro-step (start_micro_step(k > 0)) or the CPU optimizer (wait_all_offloads).
//
// The host gradients end bit-identical to the memset path for any set of copies, provided every copy into a
// registered buffer calls before_copy() before it is enqueued. For each buffer, the first copy of the step that
// writes to it decides:
//   - it covers the whole buffer: nothing to zero; the memset path zeroes the buffer and then copies the same
//     bytes over it;
//   - it covers part of the buffer (shorter than the buffer, or starting inside it): before_copy() zeroes the
//     whole buffer at once, before that copy is enqueued, which is the memset path's order (zero, then copy), so
//     this copy and any later copy into the buffer land on zeros;
//   - no copy writes to it: zero_rest() zeroes it before its first reader.
// No zeroing races the GPU. before_copy() zeroes a buffer before the copy it precedes is enqueued, and only when
// no earlier copy of the step has written to that buffer. zero_rest() zeroes only buffers that no copy of the step
// has written to. The previous step's copies finished in its wait_all_offloads. Once zero_rest() has run, every
// buffer holds copied bytes or zeros, and before_copy() does nothing until the next begin_step().
//
// Registered buffers do not overlap, except that one buffer can be registered more than once (a tied lm_head and
// the embedding share one, ledger T10); add() throws std::logic_error on any other overlap.
//
// The class is plain host code (no CUDA), so the contract test compiles it without a GPU
// (fork/native/openmp-contract/cpu_step_check.cpp: check_overwrites_rules and check_overwrites_protocol).
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <map>
#include <stdexcept>
#include <utility>

namespace surogate_fork {

class HostGradOverwrites {
public:
    // Registers a host buffer of `bytes` bytes. A buffer registered again (same start) keeps the larger size. On an
    // overlap it throws and leaves the registrations as they were.
    void add(void* data, std::size_t bytes) {
        if (!data || bytes == 0) return;
        const std::uintptr_t start = address(data);
        auto next = mByStart.upper_bound(start);  // the first buffer that starts after this one
        auto same = next == mByStart.begin() ? mByStart.end() : std::prev(next);
        if (same != mByStart.end() && same->first != start) {
            if (end_of(*same) > start) overlap();  // starts inside an earlier buffer
            same = mByStart.end();
        }
        const std::size_t size = same == mByStart.end() ? bytes : std::max(same->second.bytes, bytes);
        if (next != mByStart.end() && start + size > next->first) overlap();  // runs into the next buffer
        if (same == mByStart.end()) {
            mByStart.emplace_hint(next, start, Buffer{data, size, false});
        } else {
            same->second.bytes = size;
        }
    }

    // The first micro-step of an optimizer step starts: no copy has written to any buffer in it yet.
    void begin_step() {
        for (auto& entry : mByStart)
            entry.second.written = false;
        mPending = true;
    }

    // A copy of `bytes` bytes to `dst` is about to be enqueued: call this first. Returns the bytes it zeroed (see
    // the top of this file): the buffers this copy is the first of the step to write to and does not cover.
    std::size_t before_copy(const void* dst, std::size_t bytes) {
        if (!mPending || !dst || bytes == 0) return 0;
        const std::uintptr_t begin = address(dst);
        const std::uintptr_t end = begin + bytes;
        auto it = mByStart.upper_bound(begin);                               // the first buffer that starts after `dst`
        if (it != mByStart.begin() && end_of(*std::prev(it)) > begin) --it;  // the buffer `dst` points into
        std::size_t zeroed = 0;
        for (; it != mByStart.end() && it->first < end; ++it) {
            Buffer& buffer = it->second;
            if (buffer.written) continue;
            buffer.written = true;
            if (begin > it->first || end < end_of(*it)) {
                std::memset(buffer.data, 0, buffer.bytes);
                zeroed += buffer.bytes;
            }
        }
        return zeroed;
    }

    // Zeroes every buffer that no copy has written to since begin_step(), at most once per step, and returns the
    // number of bytes it zeroed.
    std::size_t zero_rest() {
        if (!mPending) return 0;
        mPending = false;
        std::size_t zeroed = 0;
        for (const auto& entry : mByStart) {
            const Buffer& buffer = entry.second;
            if (buffer.written) continue;
            std::memset(buffer.data, 0, buffer.bytes);
            zeroed += buffer.bytes;
        }
        return zeroed;
    }

    [[nodiscard]] std::size_t buffers() const {
        return mByStart.size();
    }

private:
    struct Buffer {
        void* data;
        std::size_t bytes;
        bool written;  // a copy of this step has written to it, or before_copy() has zeroed it
    };

    static std::uintptr_t address(const void* p) {
        return reinterpret_cast<std::uintptr_t>(p);
    }

    static std::uintptr_t end_of(const std::pair<const std::uintptr_t, Buffer>& entry) {
        return entry.first + entry.second.bytes;
    }

    [[noreturn]] static void overlap() {
        throw std::logic_error("HostGradOverwrites: host gradient buffers overlap (fork ledger T13)");
    }

    std::map<std::uintptr_t, Buffer> mByStart;
    bool mPending = false;  // begin_step() ran and zero_rest() has not
};

}  // namespace surogate_fork
