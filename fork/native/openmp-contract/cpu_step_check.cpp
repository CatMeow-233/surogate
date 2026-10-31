// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// CPU contract of the cpu_training host step (fork ledgers T13-T15), without CUDA:
//   1. surogate_fork::HostGradOverwrites: replaying DslGradStore's host-side protocol (micro-step 0 D2H
//      overwrites, staged additions on later micro-steps, non-block copies, gaps in the coverage) gives the same
//      host gradients as the upstream memset of every buffer at micro-step 0 (T13);
//   2. host_grad_add_bf16/f32 give the bits of DslGradStore::accumulate_staged_layer's loop, with the host
//      __float2bfloat16 of cuda_bf16.hpp, for every BF16 `dst` value against a structured set of `src` values,
//      at every team size and length (T14);
//   3. adamw_step_* give the bits of cpu_adamw.cpp's three loops, specials included (T15), where the build's
//      baseline ISA has no fused multiply-add; where it has one, SUROGATE_FORK_CPU_ADAMW_SIMD=1 is refused.
//   In 2 and 3 a NaN in an FP32 result may carry another NaN's payload (same_floats explains why).
//   4. the host-function team and the rank thread's team do not halve each other when they share a serial group
//      (T14), and do without it.
// The reference loops below are the upstream expressions. The checker is built without OpenMP (as team_check.cpp)
// but at -O3 and with the compiler's default contraction, as surogate-common's Release build compiles
// cpu_adamw.cpp, so that on a baseline ISA with FMA the references fuse multiply-adds as upstream's loops do there.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <pthread.h>
#include <random>
#include <stdexcept>
#include <sched.h>
#include <thread>
#include <vector>

#include "surogate_fork/cpu_step_kernels.h"
#include "surogate_fork/host_grad_overwrites.h"
#include "team_api.h"

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

// ---- the upstream loops (dsl_grad_store.cpp accumulate_staged_layer, cpu_adamw.cpp), with the copy above ------

void ref_add_bf16(std::uint16_t* dst, const std::uint16_t* src, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        float val = ref_bf16_to_f32(dst[i]) + ref_bf16_to_f32(src[i]);
        dst[i] = ref_f32_to_bf16(val);
    }
}

void ref_adamw_f32(float* p, const float* g, float* m, float* v, std::size_t n, const surogate_fork::AdamWStep& s) {
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = g[i] * s.grad_scale;
        m[i] = s.beta1 * m[i] + (1.0f - s.beta1) * gi;
        v[i] = s.beta2 * v[i] + (1.0f - s.beta2) * gi * gi;
        const float m_hat = m[i] / s.bc1;
        const float v_hat = v[i] / s.bc2;
        p[i] = p[i] * s.wd_factor - s.lr * m_hat / (std::sqrt(v_hat) + s.eps);
    }
}

void ref_adamw_bf16_grad(float* p,
                         const std::uint16_t* g,
                         float* m,
                         float* v,
                         std::size_t n,
                         const surogate_fork::AdamWStep& s) {
    for (std::size_t i = 0; i < n; ++i) {
        const float gi = ref_bf16_to_f32(g[i]) * s.grad_scale;
        m[i] = s.beta1 * m[i] + (1.0f - s.beta1) * gi;
        v[i] = s.beta2 * v[i] + (1.0f - s.beta2) * gi * gi;
        const float m_hat = m[i] / s.bc1;
        const float v_hat = v[i] / s.bc2;
        p[i] = p[i] * s.wd_factor - s.lr * m_hat / (std::sqrt(v_hat) + s.eps);
    }
}

void ref_adamw_bf16_param(std::uint16_t* p,
                          const std::uint16_t* g,
                          float* m,
                          float* v,
                          std::size_t n,
                          const surogate_fork::AdamWStep& s) {
    for (std::size_t i = 0; i < n; ++i) {
        float pi = ref_bf16_to_f32(p[i]);
        const float gi = ref_bf16_to_f32(g[i]) * s.grad_scale;
        m[i] = s.beta1 * m[i] + (1.0f - s.beta1) * gi;
        v[i] = s.beta2 * v[i] + (1.0f - s.beta2) * gi * gi;
        const float m_hat = m[i] / s.bc1;
        const float v_hat = v[i] / s.bc2;
        pi = pi * s.wd_factor - s.lr * m_hat / (std::sqrt(v_hat) + s.eps);
        p[i] = ref_f32_to_bf16(pi);
    }
}

// ---- inputs --------------------------------------------------------------------------------------------------

std::vector<int> process_cpus() {
    cpu_set_t mask;
    sched_getaffinity(0, sizeof(mask), &mask);
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &mask)) cpus.push_back(c);
    return cpus;
}

void pin(const std::vector<int>& cpus) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int c : cpus)
        CPU_SET(c, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
        std::fprintf(stderr, "FAIL: pthread_setaffinity_np\n");
        ++failures;
    }
}

// Every exponent with a few mantissas and both signs, the specials (zeros, infinities, NaNs with payloads,
// subnormals, the largest finite values) and values one ULP around rounding ties.
std::vector<std::uint16_t> structured_bf16() {
    std::vector<std::uint16_t> out;
    for (std::uint32_t sign : {0u, 0x8000u})
        for (std::uint32_t exp = 0; exp < 256; ++exp)
            for (std::uint32_t mant : {0u, 1u, 0x3fu, 0x40u, 0x41u, 0x7eu, 0x7fu})
                out.push_back(static_cast<std::uint16_t>(sign | exp << 7 | mant));
    return out;
}

std::vector<float> wide_floats(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<float> out(n);
    for (auto& x : out) {
        const std::uint64_t r = rng();
        const float mantissa = 1.0f + static_cast<float>(r & 0xffffff) / 16777216.0f;
        x = std::ldexp(mantissa, static_cast<int>((r >> 24) % 60) - 40) * ((r >> 40) & 1 ? -1.0f : 1.0f);
        switch ((r >> 48) % 64) {
            case 0: x = 0.0f; break;
            case 1: x = -0.0f; break;
            case 2: x = std::numeric_limits<float>::infinity(); break;
            case 3: x = std::numeric_limits<float>::quiet_NaN(); break;
            case 4: x = from_bits(0x7f800001u + static_cast<std::uint32_t>(r >> 56)); break;  // signalling NaN
            case 5: x = std::numeric_limits<float>::denorm_min() * static_cast<float>(r >> 56); break;
            case 6: x = std::numeric_limits<float>::max(); break;
            default: break;
        }
    }
    return out;
}

std::vector<std::uint16_t> wide_bf16(std::size_t n, std::uint64_t seed) {
    std::vector<std::uint16_t> out(n);
    const std::vector<float> f = wide_floats(n, seed);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = ref_f32_to_bf16(f[i]);
    return out;
}

bool same_bits(const void* a, const void* b, std::size_t bytes) {
    return std::memcmp(a, b, bytes) == 0;
}

// FP32 results: the same bits, except that two NaNs match whatever their payloads. An operation with two NaN
// operands returns one of them, and which one is not fixed (IEEE 754 leaves it open, and GCC commutes the operands
// of + and *), so the vectorized loop may keep the other payload. BF16 results map every NaN to 0x7fff and are
// compared with same_bits.
bool same_floats(const float* a, const float* b, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (std::isnan(a[i]) ? !std::isnan(b[i]) : bits(a[i]) != bits(b[i])) return false;
    return true;
}

// ---- 1. host gradient overwrites (T13) ------------------------------------------------------------------------

// One copy into a host buffer: `len` elements at element `offset`.
struct Copy {
    std::size_t offset, len;
};

// A replay of DslGradStore's host-side protocol over random steps. Buffers 0..L-1 are block gradients (one layer
// each), L the buffer a tied embedding and lm_head share (copied under both names), L+1 a non-block gradient
// whose device buffer is null (never copied). Device values are fresh random bytes per micro-step. `old_host`
// takes upstream's memset at micro-step 0; `new_host` calls HostGradOverwrites as dsl_grad_store.cpp does. Each
// micro-step-0 copy into a block gradient is, at random, skipped (a coverage gap), whole (what the grad store
// enqueues), or one or two partial copies (shorter than the buffer, or at an offset), which only a future copy
// could make: the memset path gives those the copied ranges over zeros.
void check_overwrites_protocol() {
    constexpr int L = 6;
    std::mt19937_64 rng(7);
    std::vector<std::size_t> sizes;
    for (int i = 0; i < L; ++i)
        sizes.push_back(2 + rng() % 3000);
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

    // The copies of one micro-step-0 D2H into a buffer of n elements.
    const auto copies_of = [&](std::size_t n) {
        std::vector<Copy> copies;
        switch (rng() % 6) {
            case 0: break;                                              // no copy: a coverage gap
            case 1: copies.push_back({0, 1 + rng() % (n - 1)}); break;  // shorter than the buffer
            case 2: {                                                   // inside the buffer, at an offset
                const std::size_t offset = 1 + rng() % (n - 1);
                copies.push_back({offset, 1 + rng() % (n - offset)});
                break;
            }
            case 3: {  // two halves, which cover the buffer together
                const std::size_t half = 1 + rng() % (n - 1);
                copies.push_back({half, n - half});
                copies.push_back({0, half});
                break;
            }
            default: copies.push_back({0, n}); break;  // the whole buffer, as dsl_grad_store.cpp copies today
        }
        return copies;
    };
    // Copies `device` into `old_host[b]` and `new_host[b]` as `copies` say, calling before_copy first.
    const auto copy_into =
        [&](std::size_t b, const std::vector<std::uint16_t>& device, const std::vector<Copy>& copies, bool record) {
            for (const Copy& c : copies) {
                if (record) overwrites.before_copy(new_host[b].data() + c.offset, c.len * 2);
                std::copy_n(device.begin() + static_cast<long>(c.offset),
                            c.len,
                            old_host[b].begin() + static_cast<long>(c.offset));
                std::copy_n(device.begin() + static_cast<long>(c.offset),
                            c.len,
                            new_host[b].begin() + static_cast<long>(c.offset));
            }
        };

    for (int step = 0; step < 200; ++step) {
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
            // backward: each layer's D2H, then the non-block copies
            for (int layer = L - 1; layer >= 0; --layer) {
                const std::size_t n = sizes[layer];
                std::vector<std::uint16_t> device(n);
                fill(device);
                if (micro == 0) {
                    copy_into(layer, device, copies_of(n), true);
                } else if (rng() % 5 != 0) {
                    ref_add_bf16(old_host[layer].data(), device.data(), n);
                    ref_add_bf16(new_host[layer].data(), device.data(), n);
                }
            }
            if (rng() % 4 != 0) {  // the tied gradient, copied whole under both names (later micro-steps too)
                std::vector<std::uint16_t> device(sizes[L]);
                fill(device);
                for (int name = 0; name < 2; ++name)
                    copy_into(L, device, {{0, sizes[L]}}, true);
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
    std::vector<float> a(100, 1.0f), b(50, 2.0f), c(64, 3.0f), d(64, 4.0f);
    surogate_fork::HostGradOverwrites o;
    o.add(a.data(), a.size() * 4);
    o.add(b.data(), b.size() * 4);
    o.add(c.data(), c.size() * 4);
    o.add(d.data(), 32 * 4);  // two adjacent buffers
    o.add(d.data() + 32, 32 * 4);
    o.add(nullptr, 16);  // a gradient without storage registers nothing
    expect(o.buffers() == 5, "null buffers do not register", static_cast<long long>(o.buffers()), 5);
    bool threw = false;
    try {
        o.add(a.data() + 10, 8);  // a view into a registered buffer
    } catch (const std::logic_error&) {
        threw = true;
    }
    expect(threw, "an overlapping registration throws", threw, 1);
    threw = false;
    try {
        o.add(d.data(), 33 * 4);  // the first of the adjacent buffers grown into the second
    } catch (const std::logic_error&) {
        threw = true;
    }
    expect(threw, "a registration grown into the next buffer throws", threw, 1);

    expect(o.zero_rest() == 0, "zero_rest before begin_step zeroes nothing", 0, 0);
    expect(o.before_copy(a.data(), 8) == 0 && a[0] == 1.0f, "before_copy outside a step does nothing", 0, 0);
    o.begin_step();
    std::size_t zeroed = o.before_copy(b.data(), b.size() * 4);
    expect(zeroed == 0 && b[0] == 2.0f, "a copy over a whole buffer zeroes nothing", static_cast<long long>(zeroed), 0);
    zeroed = o.before_copy(a.data(), a.size() * 4 - 4);
    expect(zeroed == a.size() * 4 && a[0] == 0.0f && a[99] == 0.0f,
           "a copy shorter than its buffer zeroes it first",
           static_cast<long long>(zeroed),
           static_cast<long long>(a.size() * 4));
    std::fill(a.begin(), a.end() - 1, 5.0f);  // that copy lands
    zeroed = o.before_copy(a.data() + 50, 40);
    expect(zeroed == 0 && a[0] == 5.0f, "a later copy into the same buffer keeps the earlier one", 0, 0);
    std::fill(a.begin() + 50, a.begin() + 60, 6.0f);
    zeroed = o.before_copy(d.data() + 16, 32 * 4);
    expect(zeroed == 64 * 4 && d[0] == 0.0f && d[63] == 0.0f,
           "a copy across two buffers, covering neither, zeroes both",
           static_cast<long long>(zeroed),
           64 * 4);
    zeroed = o.zero_rest();
    expect(zeroed == c.size() * 4,
           "zero_rest zeroes only the buffer no copy wrote",
           static_cast<long long>(zeroed),
           256);
    expect(c[0] == 0.0f && b[49] == 2.0f && a[0] == 5.0f && a[55] == 6.0f && a[99] == 0.0f,
           "zeroed and kept contents",
           0,
           0);
    c.assign(c.size(), 3.0f);
    expect(o.zero_rest() == 0 && c[0] == 3.0f, "zero_rest runs once per step", static_cast<long long>(c[0]), 3);
    expect(o.before_copy(c.data() + 1, 8) == 0 && c[0] == 3.0f, "before_copy after zero_rest does nothing", 0, 0);
    o.begin_step();
    expect(o.zero_rest() == (100 + 50 + 64 + 64) * 4, "the next step starts over", 0, 0);
}

// ---- 2. host gradient additions (T14) -------------------------------------------------------------------------

void check_add_bf16_values() {
    // Every BF16 dst value against the structured src set, in one long call per src block.
    const std::vector<std::uint16_t> srcs = structured_bf16();
    std::vector<std::uint16_t> dst(65536), want(65536), src(65536);
    for (std::uint16_t s : srcs) {
        for (std::uint32_t d = 0; d < 65536; ++d) {
            dst[d] = static_cast<std::uint16_t>(d);
            src[d] = s;
        }
        want = dst;
        ref_add_bf16(want.data(), src.data(), want.size());
        surogate_fork::host_grad_add_bf16(dst.data(), src.data(), dst.size(), nullptr);
        if (!same_bits(dst.data(), want.data(), dst.size() * 2)) {
            std::size_t i = 0;
            while (dst[i] == want[i])
                ++i;
            std::fprintf(stderr, "FAIL: bf16 add 0x%04zx + 0x%04x = 0x%04x, upstream 0x%04x\n", i, s, dst[i], want[i]);
            ++failures;
            return;
        }
    }
}

void check_add_lengths_and_teams(const std::vector<int>& cpus) {
    const std::size_t big = 4 * 65536 + 17;
    for (std::size_t k = 1; k <= std::min<std::size_t>(cpus.size(), 4); ++k) {
        for (std::size_t n : {std::size_t{0},
                              std::size_t{1},
                              std::size_t{63},
                              std::size_t{64},
                              std::size_t{65},
                              std::size_t{65535},
                              std::size_t{65536},
                              big}) {
            std::vector<std::uint16_t> d16 = wide_bf16(n, n + 1), s16 = wide_bf16(n, n + 2), w16 = d16;
            std::vector<float> d32 = wide_floats(n, n + 3), s32 = wide_floats(n, n + 4), w32 = d32;
            ref_add_bf16(w16.data(), s16.data(), n);
            for (std::size_t i = 0; i < n; ++i)
                w32[i] += s32[i];
            int team16 = 0, team32 = 0;
            std::thread([&] {
                pin(std::vector<int>(cpus.begin(), cpus.begin() + static_cast<long>(k)));
                team16 = surogate_fork::host_grad_add_bf16(d16.data(), s16.data(), n, nullptr);
                team32 = surogate_fork::host_grad_add_f32(d32.data(), s32.data(), n, nullptr);
            }).join();
            if (!same_bits(d16.data(), w16.data(), n * 2) || !same_floats(d32.data(), w32.data(), n)) {
                std::fprintf(stderr, "FAIL: host additions of %zu elements on %zu CPUs differ from upstream\n", n, k);
                ++failures;
            }
            const int want_team = n < 65536 ? 1 : std::min(static_cast<int>(k), fork_omp_max_threads());
            expect(team16 == want_team && team32 == want_team, "host additions run OmpTeam's team", team16, want_team);
        }
    }
}

// ---- 3. AdamW (T15) -------------------------------------------------------------------------------------------

// Whether this file's baseline ISA has a fused multiply-add (the macros cpu_step_kernels.cpp tests): the reference
// loops above then fuse, as upstream's cpu_adamw.cpp does in such a build.
constexpr bool kReferenceHasFma =
#if defined(__FP_FAST_FMAF) || defined(__FMA__) || defined(__FMA4__) || defined(__ARM_FEATURE_FMA)
    true;
#else
    false;
#endif

void check_adamw(const std::vector<int>& cpus) {
    // main() sets SUROGATE_FORK_CPU_ADAMW_SIMD=1 before anything reads it.
    expect(surogate_fork::cpu_adamw_simd_supported() == !kReferenceHasFma,
           "the kernels refuse the switch exactly where the baseline ISA has FMA",
           surogate_fork::cpu_adamw_simd_supported(),
           !kReferenceHasFma);
    expect(surogate_fork::cpu_adamw_simd_enabled() == surogate_fork::cpu_adamw_simd_supported(),
           "SUROGATE_FORK_CPU_ADAMW_SIMD=1 turns the path on where it is supported, and only there",
           surogate_fork::cpu_adamw_simd_enabled(),
           surogate_fork::cpu_adamw_simd_supported());
    if (!surogate_fork::cpu_adamw_simd_supported()) {
        std::printf("skipped: AdamW bits (the baseline ISA has FMA: upstream fuses, and T15 is refused)\n");
        return;
    }
    const surogate_fork::AdamWStep steps[] = {
        {1e-4f,
         0.9f,
         0.999f,
         1.0f - std::pow(0.9f, 3.0f),
         1.0f - std::pow(0.999f, 3.0f),
         1e-8f,
         1.0f - 1e-4f * 0.01f,
         1.0f / 4096.0f},
        {3e-2f, 0.8f, 0.95f, 1.0f - 0.8f, 1.0f - 0.95f, 1e-6f, 1.0f, 0.37f},
    };
    for (const auto& s : steps) {
        for (std::size_t k = 1; k <= std::min<std::size_t>(cpus.size(), 4); ++k) {
            for (std::size_t n : {std::size_t{1}, std::size_t{65}, std::size_t{65535}, std::size_t{3 * 65536 + 5}}) {
                const std::vector<float> g32 = wide_floats(n, 11), p32 = wide_floats(n, 12);
                std::vector<float> m0 = wide_floats(n, 13), v0 = wide_floats(n, 14);
                for (auto& x : v0)
                    x = std::fabs(x);  // the second moment is never negative
                const std::vector<std::uint16_t> g16 = wide_bf16(n, 15), p16 = wide_bf16(n, 16);

                std::vector<float> rp = p32, rm = m0, rv = v0, fp = p32, fm = m0, fv = v0;
                ref_adamw_f32(rp.data(), g32.data(), rm.data(), rv.data(), n, s);
                std::vector<float> rp2 = p32, rm2 = m0, rv2 = v0, fp2 = p32, fm2 = m0, fv2 = v0;
                ref_adamw_bf16_grad(rp2.data(), g16.data(), rm2.data(), rv2.data(), n, s);
                std::vector<std::uint16_t> rp3 = p16, fp3 = p16;
                std::vector<float> rm3 = m0, rv3 = v0, fm3 = m0, fv3 = v0;
                ref_adamw_bf16_param(rp3.data(), g16.data(), rm3.data(), rv3.data(), n, s);
                std::thread([&] {
                    pin(std::vector<int>(cpus.begin(), cpus.begin() + static_cast<long>(k)));
                    surogate_fork::adamw_step_f32(fp.data(), g32.data(), fm.data(), fv.data(), n, s);
                    surogate_fork::adamw_step_bf16_grad(fp2.data(), g16.data(), fm2.data(), fv2.data(), n, s);
                    surogate_fork::adamw_step_bf16_param(fp3.data(), g16.data(), fm3.data(), fv3.data(), n, s);
                }).join();
                const bool ok = same_floats(fp.data(), rp.data(), n) && same_floats(fm.data(), rm.data(), n) &&
                                same_floats(fv.data(), rv.data(), n) && same_floats(fp2.data(), rp2.data(), n) &&
                                same_floats(fm2.data(), rm2.data(), n) && same_floats(fv2.data(), rv2.data(), n) &&
                                same_bits(fp3.data(), rp3.data(), n * 2) && same_floats(fm3.data(), rm3.data(), n) &&
                                same_floats(fv3.data(), rv3.data(), n);
                if (!ok) {
                    std::fprintf(stderr, "FAIL: AdamW of %zu elements on %zu CPUs differs from upstream\n", n, k);
                    ++failures;
                }
            }
        }
    }
}

// ---- 4. serial group on real threads (T14) --------------------------------------------------------------------

// A rank thread runs an optimizer region, then a second thread (the CUDA callback thread) runs a host addition,
// then the rank thread runs another region, all on the same CPUs and never at the same time.
void check_serial_group_teams(const std::vector<int>& cpus) {
    if (cpus.size() < 2) {
        std::printf("skipped: serial group teams (the process may use 1 CPU)\n");
        return;
    }
    const std::vector<int> shared(cpus.begin(),
                                  cpus.begin() + static_cast<long>(std::min<std::size_t>(cpus.size(), 4)));
    const int full = std::min(static_cast<int>(shared.size()), fork_omp_max_threads());
    const int half = std::max(1, std::min(static_cast<int>(shared.size()) / 2, fork_omp_max_threads()));
    const std::size_t n = 4 * 65536;
    std::vector<float> p(n, 1.0f), g(n, 0.5f), m(n, 0.0f), v(n, 0.0f);
    std::vector<std::uint16_t> dst(n, 0x3f80), src(n, 0x3f80);
    const surogate_fork::AdamWStep s{1e-3f, 0.9f, 0.999f, 0.1f, 0.001f, 1e-8f, 1.0f, 1.0f};
    for (const bool grouped : {true, false}) {
        static int store = 0;  // stands for the DslGradStore whose address is the key
        const void* key = grouped ? &store : nullptr;
        int rank_first = 0, callback = 0, rank_second = 0;
        std::thread rank([&] {
            pin(shared);
            fork_set_omp_serial_group(key);
            rank_first = surogate_fork::adamw_step_f32(p.data(), g.data(), m.data(), v.data(), n, s);
            std::thread([&] {
                pin(shared);
                callback = surogate_fork::host_grad_add_bf16(dst.data(), src.data(), n, key);
            }).join();
            rank_second = surogate_fork::adamw_step_f32(p.data(), g.data(), m.data(), v.data(), n, s);
        });
        rank.join();
        expect(rank_first == full, "the rank thread alone gets its CPUs", rank_first, full);
        if (grouped) {
            expect(callback == full, "a host function in the rank's serial group keeps the full team", callback, full);
            expect(rank_second == full, "and the rank thread's next team stays full", rank_second, full);
        } else {
            // Exited threads stop counting, so each call above saw only callers still alive.
            expect(callback == half, "without a group the host function shares with the rank thread", callback, half);
        }
    }
}

}  // namespace

int main() {
    setenv("SUROGATE_FORK_CPU_ADAMW_SIMD", "1", 1);  // read once, by check_adamw
    const std::vector<int> cpus = process_cpus();
    check_overwrites_rules();
    check_overwrites_protocol();
    // On a fresh thread: the main thread would stay registered as a caller on every CPU and halve later teams.
    std::thread(check_add_bf16_values).join();
    check_add_lengths_and_teams(cpus);
    check_adamw(cpus);
    check_serial_group_teams(cpus);
    if (failures) {
        std::fprintf(stderr, "%d cpu step checks failed\n", failures);
        return 1;
    }
    std::printf("cpu step contract verified on %zu CPUs\n", cpus.size());
    return 0;
}
