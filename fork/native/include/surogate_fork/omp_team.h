// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// OpenMP team sizes and the ordered reduction for the CPU optimizer regions of
// csrc/src/runtime/optimizers/cpu_adamw.cpp (fork ledger T5).
//
// libgomp sizes its default team once, from the CPU mask of the process. Each trainer rank thread then pins
// itself to its GPU's CPUs (set_cpu_affinity -> nvmlDeviceSetCpuAffinity, which sets the calling thread only),
// team threads inherit that mask, and on a multi-socket or multi-NUMA host a default team runs more threads
// than its CPUs and spin-waits at every barrier. Every region of cpu_adamw.cpp therefore runs OmpTeam(n).threads:
//   - 1 when n < kOmpMinParallelElems: the region's if clause is false and the loop runs on the calling thread,
//     because below that size the fork/join costs more than the parallel loop saves;
//   - otherwise the CPUs in the calling thread's affinity mask, divided by the number of callers whose masks
//     overlap it and that are in a region or left one within kOmpSharerWindowNs (a process's rank threads can
//     all pin to the same CPUs), at least 1 and at most omp_get_max_threads(), so OMP_NUM_THREADS still lowers it.
// The mask is read on every such call (one sched_getaffinity syscall), so a thread that re-pins is followed;
// worker threads libgomp already created for it keep the mask they started with.
// A caller is counted from its first such region until its thread exits; one idle for longer than the window
// stops counting, so a thread that ran the optimizer once does not halve its neighbours' teams forever. The
// registry's mutex is held only for that bookkeeping; a forked child that never runs the optimizer is unaffected.
//
// ordered_block_sum() is the reduction the gradient norm uses. Each kOrderedSumBlock-element block is summed in
// index order and the block sums are added in block order, so the result is the same for every team size: it
// does not depend on pinning, on other ranks, on the threshold, or on whether the build has OpenMP. For
// n <= kOrderedSumBlock it is the plain left-to-right sum of the upstream serial loop.
//
// Without OpenMP (upstream's build of surogate-common) every team is 1 and nothing is registered.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <chrono>
#include <mutex>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#endif

// Benchmarks may override the threshold (fork/native/bench/cpu-optimizer); builds use the default.
#ifndef SUROGATE_FORK_OMP_MIN_PARALLEL_ELEMS
#define SUROGATE_FORK_OMP_MIN_PARALLEL_ELEMS 65536
#endif

namespace surogate_fork {

// Smallest loop that gets a team (fork/docs/performance.md, "CPU 优化器并行", has the measurement).
inline constexpr std::size_t kOmpMinParallelElems = SUROGATE_FORK_OMP_MIN_PARALLEL_ELEMS;
// Block of ordered_block_sum. It fixes the summation order: changing it changes the norm's last bits.
inline constexpr std::size_t kOrderedSumBlock = 16384;
// How long a caller that left its last region still counts as sharing its CPUs.
inline constexpr std::int64_t kOmpSharerWindowNs = 1'000'000'000;

#ifdef _OPENMP
namespace detail {

struct OmpCaller {
    cpu_set_t mask;
    int active = 0;
    std::int64_t last_ns = 0;
};

struct OmpCallers {
    std::mutex mu;
    std::vector<OmpCaller*> live;
};

inline OmpCallers& omp_callers() {
    // Never destroyed: a thread can still leave a region while static destructors run.
    static OmpCallers* callers = new OmpCallers;
    return *callers;
}

inline std::int64_t omp_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// One per calling thread (OmpTeam keeps it thread_local); the contract test creates several in one thread to
// stand for several callers. Destroying it, at thread exit, unregisters the caller.
class OmpCallerSlot {
public:
    OmpCallerSlot() = default;
    OmpCallerSlot(const OmpCallerSlot&) = delete;
    OmpCallerSlot& operator=(const OmpCallerSlot&) = delete;

    ~OmpCallerSlot() {
        if (!mCaller) return;
        auto& callers = omp_callers();
        std::lock_guard<std::mutex> lock(callers.mu);
        callers.live.erase(std::find(callers.live.begin(), callers.live.end(), mCaller));
        delete mCaller;
    }

    // Registers a region on `mask` that starts at `now_ns` and returns its team size.
    int enter(const cpu_set_t& mask, std::int64_t now_ns, int max_threads) {
        auto& callers = omp_callers();
        std::lock_guard<std::mutex> lock(callers.mu);
        if (!mCaller) {
            mCaller = new OmpCaller;
            callers.live.push_back(mCaller);
        }
        mCaller->mask = mask;
        ++mCaller->active;
        mCaller->last_ns = now_ns;
        int sharers = 0;  // includes this caller
        for (const OmpCaller* other : callers.live) {
            cpu_set_t common;
            CPU_AND(&common, &other->mask, &mask);
            if (CPU_COUNT(&common) > 0 && (other->active > 0 || now_ns - other->last_ns <= kOmpSharerWindowNs))
                ++sharers;
        }
        return std::clamp(CPU_COUNT(&mask) / sharers, 1, std::max(1, max_threads));
    }

    void leave(std::int64_t now_ns) {
        auto& callers = omp_callers();
        std::lock_guard<std::mutex> lock(callers.mu);
        --mCaller->active;
        mCaller->last_ns = now_ns;
    }

private:
    OmpCaller* mCaller = nullptr;
};

}  // namespace detail
#endif

// The team for one region over `n` elements; keep it alive until the region has finished.
class OmpTeam {
public:
    explicit OmpTeam(std::size_t n) {
#ifdef _OPENMP
        if (n < kOmpMinParallelElems) return;
        cpu_set_t mask;
        if (pthread_getaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
            threads = omp_get_max_threads();  // a mask wider than cpu_set_t: libgomp's default team
            return;
        }
        thread_local detail::OmpCallerSlot slot;
        mSlot = &slot;
        threads = slot.enter(mask, detail::omp_now_ns(), omp_get_max_threads());
#else
        (void)n;
#endif
    }

    ~OmpTeam() {
#ifdef _OPENMP
        if (mSlot) mSlot->leave(detail::omp_now_ns());
#endif
    }

    OmpTeam(const OmpTeam&) = delete;
    OmpTeam& operator=(const OmpTeam&) = delete;

    int threads = 1;

private:
#ifdef _OPENMP
    detail::OmpCallerSlot* mSlot = nullptr;
#endif
};

// Sum of term(i) for i in [0, n) in a fixed order (see the top of this file), on `threads` threads.
template <class Term>
double ordered_block_sum(std::size_t n, const Term& term, int threads) {
    const std::size_t blocks = (n + kOrderedSumBlock - 1) / kOrderedSumBlock;
    const auto block_sum = [&](std::size_t b) {
        const std::size_t end = std::min(n, (b + 1) * kOrderedSumBlock);
        double sum = 0.0;
        for (std::size_t i = b * kOrderedSumBlock; i < end; ++i)
            sum += term(i);
        return sum;
    };
    double total = 0.0;
    if (threads <= 1 || blocks <= 1) {
        for (std::size_t b = 0; b < blocks; ++b)
            total += block_sum(b);
        return total;
    }
    std::vector<double> partial(blocks);
#pragma omp parallel for schedule(static) num_threads(threads)
    for (std::size_t b = 0; b < blocks; ++b)
        partial[b] = block_sum(b);
    for (const double sum : partial)
        total += sum;
    return total;
}

template <class Term>
double ordered_block_sum(std::size_t n, const Term& term) {
    const OmpTeam team(n);
    return ordered_block_sum(n, term, team.threads);
}

}  // namespace surogate_fork
