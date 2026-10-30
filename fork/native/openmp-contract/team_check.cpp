// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Team contract of the CPU optimizer regions (surogate_fork/omp_team.h, ledger T5):
//   1. loops below the threshold run on the calling thread;
//   2. a thread pinned the way NVML pins a rank thread gets a team no larger than its mask, and every team thread
//      stays inside that mask;
//   3. threads that share CPUs split them, threads on disjoint CPUs do not, and a caller stops counting when its
//      thread exits or after the sharing window;
//   4. ordered_block_sum gives the same bits for every team size, equal to the serial block order, and equal to
//      the plain left-to-right sum up to one block.
// Checks that need more CPUs than the process may use are reported as skipped, and the others still run.

#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

#include "team_api.h"

namespace {

int failures = 0;
int thread_limit = 0;  // omp_get_thread_limit()
bool dynamic = false;  // omp_get_dynamic(): the runtime may start fewer threads than asked

bool started_as_planned(const RegionReport& r) {
    const int want = std::min(r.planned, thread_limit);
    return dynamic ? r.threads >= 1 && r.threads <= want : r.threads == want;
}

void expect(bool ok, const char* what, long long got, long long want) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (got %lld, want %lld)\n", what, got, want);
        ++failures;
    }
}

std::vector<int> process_cpus() {
    cpu_set_t mask;
    sched_getaffinity(0, sizeof(mask), &mask);
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &mask)) cpus.push_back(c);
    return cpus;
}

cpu_set_t mask_of(const std::vector<int>& cpus) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int c : cpus)
        CPU_SET(c, &mask);
    return mask;
}

void pin(const std::vector<int>& cpus) {
    const cpu_set_t mask = mask_of(cpus);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
        std::fprintf(stderr, "FAIL: pthread_setaffinity_np\n");
        ++failures;
    }
}

// On a fresh thread each time: the main thread would stay registered as a caller on every CPU.
void check_threshold(std::size_t big) {
    RegionReport small{}, large{};
    std::thread([&] {
        small = fork_omp_region(fork_omp_min_parallel_elems() - 1);
        large = fork_omp_region(big);
    }).join();
    expect(small.planned == 1 && small.threads == 1, "a loop below the threshold runs serially", small.threads, 1);
    expect(started_as_planned(large), "a region runs the planned team", large.threads, large.planned);
}

void check_pinned(const std::vector<int>& cpus, std::size_t big, int max_threads) {
    for (std::size_t k : {std::size_t{1}, std::size_t{2}, cpus.size()}) {
        if (k > cpus.size()) {
            std::printf("skipped: a %zu-CPU mask (the process may use %zu CPUs)\n", k, cpus.size());
            continue;
        }
        RegionReport r{};
        std::thread([&] {
            pin(std::vector<int>(cpus.begin(), cpus.begin() + static_cast<long>(k)));
            r = fork_omp_region(big);
        }).join();
        const int want = std::min(static_cast<int>(k), max_threads);
        expect(r.planned == want, "a pinned thread's team is its mask's CPU count", r.planned, want);
        expect(started_as_planned(r), "the pinned region runs the planned team", r.threads, r.planned);
        expect(r.inside, "every team thread stays inside the caller's mask", r.inside, 1);
    }
}

void check_threads_sharing(const std::vector<int>& cpus, std::size_t big, int max_threads) {
    if (cpus.size() < 2) {
        std::printf("skipped: two threads on one mask (the process may use 1 CPU)\n");
        return;
    }
    const std::vector<int> shared(cpus.begin(),
                                  cpus.begin() + static_cast<long>(std::min<std::size_t>(cpus.size(), 4)));
    RegionReport second[2]{};
    std::barrier sync(2);
    std::vector<std::thread> ranks;
    for (int r = 0; r < 2; ++r) {
        ranks.emplace_back([&, r] {
            pin(shared);
            fork_omp_region(big);  // registers; the other rank may not have yet
            sync.arrive_and_wait();
            second[r] = fork_omp_region(big);
            sync.arrive_and_wait();  // an exited thread no longer counts: keep both alive until both are done
        });
    }
    for (auto& t : ranks)
        t.join();
    const int want = std::max(1, std::min(static_cast<int>(shared.size()) / 2, max_threads));
    for (const RegionReport& r : second) {
        expect(r.planned == want, "two threads on one mask split it", r.planned, want);
        expect(started_as_planned(r) && r.inside, "the split team runs inside the mask", r.threads, r.planned);
    }
    RegionReport after{};
    std::thread([&] {
        pin(shared);
        after = fork_omp_region(big);
    }).join();
    const int alone = std::min(static_cast<int>(shared.size()), max_threads);
    expect(after.planned == alone, "callers whose threads exited stop counting", after.planned, alone);

    if (cpus.size() < 4) {
        std::printf("skipped: two threads on disjoint masks (the process may use %zu CPUs)\n", cpus.size());
        return;
    }
    ranks.clear();
    for (int r = 0; r < 2; ++r) {
        ranks.emplace_back([&, r] {
            pin({cpus[2 * r], cpus[2 * r + 1]});
            fork_omp_region(big);
            sync.arrive_and_wait();
            second[r] = fork_omp_region(big);
            sync.arrive_and_wait();  // an exited thread no longer counts: keep both alive until both are done
        });
    }
    for (auto& t : ranks)
        t.join();
    for (const RegionReport& r : second) {
        const int want_disjoint = std::min(2, max_threads);
        expect(r.planned == want_disjoint, "threads on disjoint masks do not split", r.planned, want_disjoint);
    }
}

// The sharing rule with synthetic masks and clocks, independent of the host's CPUs.
void check_sharing_rule() {
    const long long window = fork_omp_sharer_window_ns();
    const cpu_set_t low = mask_of({0, 1, 2, 3}), high = mask_of({4, 5, 6, 7}), mid = mask_of({2, 3, 4, 5});
    const cpu_set_t one = mask_of({0});
    const long long t = 1'000'000'000'000;
    int got = fork_slot_enter(1, low, t, 64);
    expect(got == 4, "a caller alone gets its mask", got, 4);
    fork_slot_leave(1, t + 1);
    got = fork_slot_enter(2, low, t + 2, 64);
    expect(got == 2, "a caller that left within the window still counts", got, 2);
    got = fork_slot_enter(1, low, t + 3, 64);
    expect(got == 2, "a caller in a region counts", got, 2);
    got = fork_slot_enter(3, high, t + 4, 64);
    expect(got == 4, "a disjoint mask is not shared", got, 4);
    got = fork_slot_enter(4, mid, t + 5, 64);
    expect(got == 1, "an overlapping mask shares with every caller it overlaps", got, 1);
    for (int id : {1, 2, 3, 4})
        fork_slot_leave(id, t + 6);
    got = fork_slot_enter(5, low, t + 6 + window + 1, 64);
    expect(got == 4, "callers idle for longer than the window stop counting", got, 4);
    fork_slot_leave(5, t + 6 + window + 1);
    got = fork_slot_enter(6, low, t + 7, 64);  // callers 1, 2, 4, 5 and 6 share the 4 CPUs of `low`
    expect(got == 1, "a team is at least one thread", got, 1);
    fork_slot_leave(6, t + 7);
    for (int id : {1, 2, 3, 4, 5, 6})
        fork_slot_destroy(id);
    got = fork_slot_enter(7, low, t + 8, 3);
    expect(got == 3, "omp_get_max_threads caps the team", got, 3);
    got = fork_slot_enter(8, one, t + 8, 64);
    expect(got == 1, "a caller on one CPU shared with another gets one thread", got, 1);
    for (int id : {7, 8})
        fork_slot_destroy(id);
    got = fork_slot_enter(9, low, t + 9, 64);
    expect(got == 4, "destroyed callers stop counting", got, 4);
    fork_slot_destroy(9);
}

std::vector<float> wide_range(std::size_t n) {
    std::vector<float> d(n);
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint64_t h = (i + 1) * 0x9e3779b97f4a7c15ull;
        const float mantissa = 1.0f + static_cast<float>((h >> 11) & 0xfffff) / 1048576.0f;
        d[i] = std::ldexp(mantissa, static_cast<int>((h >> 40) % 40) - 30) * ((h >> 33) & 1 ? -1.0f : 1.0f);
    }
    if (n > 7) {
        d[3] = std::numeric_limits<float>::quiet_NaN();
        d[n - 2] = std::numeric_limits<float>::infinity();
    }
    return d;
}

double term(const std::vector<float>& d, std::size_t i) {
    return std::isfinite(d[i]) ? static_cast<double>(d[i]) * static_cast<double>(d[i]) : 0.0;
}

void check_ordered_sum(const std::vector<int>& cpus) {
    const std::size_t block = fork_ordered_sum_block();
    for (std::size_t n :
         {std::size_t{0}, std::size_t{1}, std::size_t{9}, block - 1, block, block + 1, 3 * block, 37 * block + 5}) {
        const std::vector<float> d = wide_range(n);
        double reference = 0.0, plain = 0.0;
        for (std::size_t b = 0; b * block < n; ++b) {
            double sum = 0.0;
            for (std::size_t i = b * block; i < std::min(n, (b + 1) * block); ++i)
                sum += term(d, i);
            reference += sum;
        }
        for (std::size_t i = 0; i < n; ++i)
            plain += term(d, i);
        for (int threads = 1; threads <= 4; ++threads) {
            const double got = fork_ordered_sum(d.data(), n, threads);
            if (got != reference) {
                std::fprintf(stderr,
                             "FAIL: ordered sum of %zu elements on %d threads is %a, serial order %a\n",
                             n,
                             threads,
                             got,
                             reference);
                ++failures;
            }
        }
        double pinned = 0.0, unpinned = 0.0;
        std::thread([&] {
            pin({cpus.front()});
            pinned = fork_ordered_sum_auto(d.data(), n);
        }).join();
        std::thread([&] { unpinned = fork_ordered_sum_auto(d.data(), n); }).join();
        if (pinned != reference || unpinned != reference) {
            std::fprintf(stderr,
                         "FAIL: ordered sum of %zu elements depends on the caller's mask: %a %a, want %a\n",
                         n,
                         pinned,
                         unpinned,
                         reference);
            ++failures;
        }
        if (n <= block && plain != reference) {
            std::fprintf(stderr, "FAIL: up to one block the order is left to right: %a, want %a\n", reference, plain);
            ++failures;
        }
        // The data must make the order visible, or the checks above could not catch a reordering.
        if (n == 37 * block + 5 && plain == reference) {
            std::fprintf(stderr, "FAIL: the test data does not distinguish summation orders\n");
            ++failures;
        }
    }
}

}  // namespace

int main() {
    const std::vector<int> cpus = process_cpus();
    const int max_threads = fork_omp_max_threads();
    thread_limit = fork_omp_thread_limit();
    dynamic = fork_omp_dynamic();
    const std::size_t big = 4 * fork_omp_min_parallel_elems();
    check_sharing_rule();
    check_threshold(big);
    check_pinned(cpus, big, max_threads);
    check_threads_sharing(cpus, big, max_threads);
    check_ordered_sum(cpus);
    if (failures) {
        std::fprintf(stderr, "%d team contract checks failed\n", failures);
        return 1;
    }
    std::printf("team contract verified on %zu CPUs (omp_get_max_threads %d)\n", cpus.size(), max_threads);
    return 0;
}
