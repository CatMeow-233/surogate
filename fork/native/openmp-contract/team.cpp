// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Library side of the team contract: surogate_fork/omp_team.h compiled with the training OpenMP flags, as
// csrc/src/runtime/optimizers/cpu_adamw.cpp is in surogate-common. team_check.cpp drives it.

#include "surogate_fork/omp_team.h"

#include <cmath>
#include <map>
#include <memory>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

#ifndef _OPENMP
#error "The training OpenMP target did not enable OpenMP for its CXX source"
#endif

#include "team_api.h"

std::size_t fork_omp_min_parallel_elems() {
    return surogate_fork::kOmpMinParallelElems;
}

std::size_t fork_ordered_sum_block() {
    return surogate_fork::kOrderedSumBlock;
}

long long fork_omp_sharer_window_ns() {
    return surogate_fork::kOmpSharerWindowNs;
}

int fork_omp_max_threads() {
    return omp_get_max_threads();
}

int fork_omp_thread_limit() {
    return omp_get_thread_limit();
}

bool fork_omp_dynamic() {
    return omp_get_dynamic() != 0;
}

// The same region shape as cpu_adamw.cpp's loops.
RegionReport fork_omp_region(std::size_t n) {
    cpu_set_t caller;
    pthread_getaffinity_np(pthread_self(), sizeof(caller), &caller);
    const surogate_fork::OmpTeam team(n);
    int threads = 0;
    bool inside = true;
#pragma omp parallel num_threads(team.threads) if (team.threads > 1) reduction(&& : inside)
    {
#pragma omp single
        threads = omp_get_num_threads();
        cpu_set_t mine, outside;
        pthread_getaffinity_np(pthread_self(), sizeof(mine), &mine);
        CPU_XOR(&outside, &mine, &caller);
        CPU_AND(&outside, &outside, &mine);  // CPUs this team thread may use and the caller may not
        inside = CPU_COUNT(&outside) == 0;
    }
    return {team.threads, threads, inside};
}

namespace {

double norm_term(const float* d, std::size_t i) {
    const float v = d[i];
    return std::isfinite(v) ? static_cast<double>(v) * static_cast<double>(v) : 0.0;
}

std::map<int, std::unique_ptr<surogate_fork::detail::OmpCallerSlot>>& slots() {
    static std::map<int, std::unique_ptr<surogate_fork::detail::OmpCallerSlot>> s;
    return s;
}

}  // namespace

double fork_ordered_sum(const float* d, std::size_t n, int threads) {
    return surogate_fork::ordered_block_sum(n, [d](std::size_t i) { return norm_term(d, i); }, threads);
}

double fork_ordered_sum_auto(const float* d, std::size_t n) {
    return surogate_fork::ordered_block_sum(n, [d](std::size_t i) { return norm_term(d, i); });
}

int fork_slot_enter(int id, const cpu_set_t& mask, long long now_ns, int max_threads) {
    auto& slot = slots()[id];
    if (!slot) slot = std::make_unique<surogate_fork::detail::OmpCallerSlot>();
    return slot->enter(mask, now_ns, max_threads);
}

void fork_slot_leave(int id, long long now_ns) {
    slots().at(id)->leave(now_ns);
}

void fork_slot_destroy(int id) {
    slots().erase(id);
}
