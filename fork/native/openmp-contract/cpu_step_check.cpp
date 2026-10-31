// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// CPU contract of the cpu_training host step (fork ledger T13), without CUDA: surogate_fork::HostGradOverwrites.
// Replaying DslGradStore's host-side protocol (micro-step 0 D2H overwrites, staged additions on later micro-steps,
// non-block copies, gaps in the coverage) gives the same host gradients as the upstream memset of every buffer at
// micro-step 0. The staged addition below is DslGradStore::accumulate_staged_layer's BF16 loop with the host
// __float2bfloat16 of cuda_bf16.hpp, copied.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "surogate_fork/host_grad_overwrites.h"

namespace {

int failures = 0;

void expect(bool ok, const char* what, long long got, long long want) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s (got %lld, want %lld)\n", what, got, want);
        ++failures;
    }
}

// ---- cuda_bf16.hpp host path, copied: __internal_float2bfloat16 and the rounding of __float2bfloat16 ----------

std::uint32_t bits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float from_bits(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

float ref_bf16_to_f32(std::uint16_t h) {
    return from_bits(static_cast<std::uint32_t>(h) << 16U);
}

std::uint16_t ref_f32_to_bf16(float a) {
    const std::uint32_t x = bits(a);
    if ((x & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>(0x7fffU);
    const std::uint32_t remainder = x << 16U;
    auto r = static_cast<std::uint16_t>(x >> 16U);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r & 0x1U) != 0U))) r++;
    return r;
}

// ---- the upstream loop (dsl_grad_store.cpp accumulate_staged_layer), with the copy above --------------------

void ref_add_bf16(std::uint16_t* dst, const std::uint16_t* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        float val = ref_bf16_to_f32(dst[i]) + ref_bf16_to_f32(src[i]);
        dst[i] = ref_f32_to_bf16(val);
    }
}

bool same_bits(const void* a, const void* b, std::size_t bytes) {
    return std::memcmp(a, b, bytes) == 0;
}

// ---- 1. host gradient overwrites (T13) ------------------------------------------------------------------------

// A replay of DslGradStore's host-side protocol over random steps. Buffers 0..L-1 are block gradients (one layer
// each), L the buffer a tied embedding and lm_head share (copied under both names), L+1 a non-block gradient
// whose device buffer is null (never copied). Device values are fresh random bytes per micro-step. `old_host`
// takes upstream's memset at micro-step 0; `new_host` uses HostGradOverwrites as dsl_grad_store.cpp does.
void check_overwrites_protocol() {
    constexpr int L = 6;
    std::mt19937_64 rng(7);
    std::vector<std::size_t> sizes;
    for (int i = 0; i < L; ++i)
        sizes.push_back(1 + rng() % 3000);
    sizes.push_back(2048);  // tied embedding/lm_head
    sizes.push_back(77);    // non-block gradient with no device buffer
    const auto fill = [&](std::vector<std::uint16_t>& b) {
        for (auto& x : b)
            x = static_cast<std::uint16_t>(rng());
    };
    std::vector<std::vector<std::uint16_t>> old_host, new_host;
    for (std::size_t s : sizes) {
        old_host.emplace_back(s);
        fill(old_host.back());  // pinned memory starts with whatever it held
        new_host.push_back(old_host.back());
    }
    surogate_fork::HostGradOverwrites overwrites;
    for (std::size_t b = 0; b < sizes.size(); ++b)
        overwrites.add(new_host[b].data(), sizes[b] * 2);
    overwrites.add(new_host[L].data(), sizes[L] * 2);  // lm_head registers the embedding's buffer again
    expect(overwrites.buffers() == sizes.size(),
           "a shared buffer registers once",
           static_cast<long long>(overwrites.buffers()),
           static_cast<long long>(sizes.size()));

    for (int step = 0; step < 40; ++step) {
        const int ga = 1 + static_cast<int>(rng() % 4);
        for (int micro = 0; micro < ga; ++micro) {
            // start_micro_step
            if (micro == 0) {
                for (auto& h : old_host)
                    std::fill(h.begin(), h.end(), 0);
                overwrites.begin_step();
            } else {
                overwrites.zero_rest();
            }
            // backward: each layer's D2H (sometimes skipped, to model a coverage gap), then the non-block copies
            for (int layer = L - 1; layer >= 0; --layer) {
                if (rng() % 5 == 0) continue;
                std::vector<std::uint16_t> device(sizes[layer]);
                fill(device);
                if (micro == 0) {
                    old_host[layer] = device;
                    new_host[layer] = device;
                    overwrites.overwritten(new_host[layer].data(), sizes[layer] * 2);
                } else {
                    ref_add_bf16(old_host[layer].data(), device.data(), sizes[layer]);
                    ref_add_bf16(new_host[layer].data(), device.data(), sizes[layer]);
                }
            }
            if (rng() % 4 != 0) {  // the tied gradient, copied under both names
                std::vector<std::uint16_t> device(sizes[L]);
                fill(device);
                for (int name = 0; name < 2; ++name) {
                    old_host[L] = device;
                    new_host[L] = device;
                    if (micro == 0) overwrites.overwritten(new_host[L].data(), sizes[L] * 2);
                }
            }
        }
        overwrites.zero_rest();  // wait_all_offloads
        for (std::size_t b = 0; b < sizes.size(); ++b) {
            if (!same_bits(old_host[b].data(), new_host[b].data(), sizes[b] * 2)) {
                std::fprintf(stderr, "FAIL: step %d buffer %zu differs from the memset path\n", step, b);
                ++failures;
                return;
            }
        }
    }
}

void check_overwrites_rules() {
    std::vector<float> a(100, 1.0f), b(50, 2.0f);
    surogate_fork::HostGradOverwrites o;
    o.add(a.data(), a.size() * 4);
    o.add(b.data(), b.size() * 4);
    o.add(nullptr, 16);  // a gradient without storage registers nothing
    expect(o.buffers() == 2, "null buffers do not register", static_cast<long long>(o.buffers()), 2);
    expect(o.zero_rest() == 0, "zero_rest before begin_step zeroes nothing", 0, 0);
    o.begin_step();
    o.overwritten(a.data(), a.size() * 4 - 4);  // a copy shorter than the buffer does not cover it
    o.overwritten(b.data(), b.size() * 4);
    const std::size_t zeroed = o.zero_rest();
    expect(zeroed == a.size() * 4,
           "only the uncovered buffer is zeroed",
           static_cast<long long>(zeroed),
           static_cast<long long>(a.size() * 4));
    expect(a[0] == 0.0f && a[99] == 0.0f && b[0] == 2.0f, "zeroed and kept contents", 0, 0);
    a.assign(a.size(), 3.0f);
    expect(o.zero_rest() == 0 && a[0] == 3.0f, "zero_rest runs once per step", static_cast<long long>(a[0]), 3);
}

}  // namespace

int main() {
    check_overwrites_rules();
    check_overwrites_protocol();
    if (failures) {
        std::fprintf(stderr, "%d cpu step checks failed\n", failures);
        return 1;
    }
    std::printf("cpu step contract verified\n");
    return 0;
}
