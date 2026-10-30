// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Functions team.cpp exports to team_check.cpp. The checker does not include surogate_fork/omp_team.h itself:
// compiled without OpenMP, its inline definitions would differ from the library's.
#pragma once

#include <cstddef>
#include <sched.h>

struct RegionReport {
    int planned;  // OmpTeam(n).threads
    int threads;  // omp_get_num_threads() inside the region
    bool inside;  // every team thread's affinity is within the caller's
};

std::size_t fork_omp_min_parallel_elems();
std::size_t fork_ordered_sum_block();
long long fork_omp_sharer_window_ns();
int fork_omp_max_threads();
int fork_omp_thread_limit();
bool fork_omp_dynamic();
RegionReport fork_omp_region(std::size_t n);
double fork_ordered_sum(const float* d, std::size_t n, int threads);
double fork_ordered_sum_auto(const float* d, std::size_t n);
// Synthetic callers for the sharing rule: slot `id` stands for one thread, `now_ns` is its clock.
int fork_slot_enter(int id, const cpu_set_t& mask, long long now_ns, int max_threads);
void fork_slot_leave(int id, long long now_ns);
void fork_slot_destroy(int id);
