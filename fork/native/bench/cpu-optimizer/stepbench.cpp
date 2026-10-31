// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Times the host side of a cpu_training optimizer step before and after fork ledgers T13-T15, against the
// trainer's own objects: cpu_adamw.cpp and fork/native/src/cpu_step_kernels.cpp compiled with their commands
// from compile_commands.json (step.sh builds and runs it).
//
// usage: stepbench MODE SHAPE PIN [REPS]
//   MODE   memset   the step-start zeroing of every host gradient: upstream's std::memset of each buffer
//                   (start_micro_step(0)) against HostGradOverwrites when micro-step 0's D2H covers all (T13)
//          accum    the staged BF16 addition of each layer on micro-steps after the first: upstream's
//                   accumulate_staged_layer loop (copied below, pinned by test_cpu_step_hooks.py) on one thread
//                   against host_grad_add_bf16 on an OmpTeam team (T14); run on a second thread, as the CUDA
//                   callback thread runs it
//          adamw    the AdamW pass over every parameter through cpu_adamw.cpp's cpu_adamw_step_bf16_param (BF16
//                   master, the shipped cpu-offload.yaml) and cpu_adamw_step_bf16 (FP32 master); step.sh runs
//                   it with SUROGATE_FORK_CPU_ADAMW_SIMD unset and =1 and compares the hashes (T15)
//          group    a step's host work in order: the callback thread adds three micro-steps of staged layers
//                   (host_grad_add_bf16), then the rank thread runs AdamW (adamw_step_bf16_param), each on an
//                   OmpTeam team; once with the rank's serial group (T14) and once without, reporting team sizes
//          exhaust  every pair of BF16 values through the upstream loop and host_grad_add_bf16: counts
//                   mismatching results (SHAPE and PIN are ignored)
//   SHAPE  llama:L:H:I:V  embedding, L layers (q/o H*H, k/v H*H/4, gate/up/down H*I, two norms), final norm,
//                         lm_head; V may be 0
//   PIN    CPUs in the calling (rank) thread's mask, 0 = the process mask; the callback thread gets the same
//   REPS   timed repetitions after one warm-up (default 5); the median is printed
// Inputs are deterministic, so hashes (FNV-1a over the results) compare across builds and switches.
#include "runtime/optimizers/cpu_adamw.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <semaphore>
#include <string>
#include <thread>
#include <vector>

#include <cuda_bf16.h>

#include "surogate_fork/cpu_step_kernels.h"
#include "surogate_fork/host_grad_overwrites.h"
#include "surogate_fork/omp_team.h"

namespace {

struct Param {
    std::size_t n;
    int layer;  // -1 for non-block parameters
};

std::vector<Param> shape(const std::string& spec) {
    long L = 0, h = 0, im = 0, V = 0;
    if (std::sscanf(spec.c_str(), "llama:%ld:%ld:%ld:%ld", &L, &h, &im, &V) != 4) {
        std::fprintf(stderr, "bad SHAPE %s\n", spec.c_str());
        std::exit(2);
    }
    std::vector<Param> params;
    if (V > 0) params.push_back({static_cast<std::size_t>(V * h), -1});
    for (int l = 0; l < L; ++l)
        for (long n : {h, h * h, h * h / 4, h * h / 4, h * h, h, h * im, h * im, h * im})
            params.push_back({static_cast<std::size_t>(n), l});
    params.push_back({static_cast<std::size_t>(h), -1});
    if (V > 0) params.push_back({static_cast<std::size_t>(V * h), -1});
    return params;
}

std::uint64_t fnv1a(const void* data, std::size_t bytes, std::uint64_t h = 0xcbf29ce484222325ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i)
        h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

std::uint16_t bf16_bits(float f) {
    const nv_bfloat16 b = __float2bfloat16(f);
    std::uint16_t u;
    std::memcpy(&u, &b, sizeof(u));
    return u;
}

// Values spanning 2^-24..2^0 with both signs, distinct per element and tensor.
void fill_bf16(std::vector<std::uint16_t>& out, std::uint64_t seed) {
    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::uint64_t h = (i + 1) * 0x9e3779b97f4a7c15ull ^ (seed + 1) * 0xc2b2ae3d27d4eb4full;
        const float mantissa = 1.0f + static_cast<float>((h >> 8) & 127) / 128.0f;
        const float value = std::ldexp(mantissa, -static_cast<int>((h >> 20) % 25));
        out[i] = bf16_bits((h >> 40) & 1 ? -value : value);
    }
}

void pin(int cpus) {
    if (cpus <= 0) return;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int c = 0; c < cpus; ++c)
        CPU_SET(c, &mask);
    if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
        std::perror("pthread_setaffinity_np");
        std::exit(1);
    }
}

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

double median(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

// dsl_grad_store.cpp DslGradStore::accumulate_staged_layer, BF16 branch, as upstream wrote it (282f9901).
void upstream_accumulate_bf16(void* dst_, const void* src_, std::size_t nelem) {
    auto* dst = static_cast<nv_bfloat16*>(dst_);
    const auto* src = static_cast<const nv_bfloat16*>(src_);
    for (std::size_t i = 0; i < nelem; ++i) {
        float val = static_cast<float>(dst[i]) + static_cast<float>(src[i]);
        dst[i] = static_cast<nv_bfloat16>(val);
    }
}

struct Layer {
    std::vector<std::vector<std::uint16_t>> grads, staged;
    std::size_t elems = 0;
};

std::vector<Layer> make_layers(const std::vector<Param>& params) {
    int layers = 0;
    for (const auto& p : params)
        layers = std::max(layers, p.layer + 1);
    std::vector<Layer> out(static_cast<std::size_t>(layers));
    std::uint64_t seed = 0;
    for (const auto& p : params) {
        if (p.layer < 0) continue;
        auto& layer = out[static_cast<std::size_t>(p.layer)];
        layer.grads.emplace_back(p.n);
        layer.staged.emplace_back(p.n);
        fill_bf16(layer.grads.back(), seed++);
        fill_bf16(layer.staged.back(), seed++);
        layer.elems += p.n;
    }
    return out;
}

void mode_memset(const std::vector<Param>& params, int reps) {
    std::vector<std::vector<std::uint16_t>> host;
    std::size_t bytes = 0;
    for (const auto& p : params) {
        host.emplace_back(p.n, 0x3f80);
        bytes += p.n * 2;
    }
    surogate_fork::HostGradOverwrites overwrites;
    for (auto& h : host)
        overwrites.add(h.data(), h.size() * 2);
    std::vector<double> before, after;
    std::size_t zeroed = 0;
    for (int r = 0; r <= reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        for (auto& h : host)
            std::memset(h.data(), 0, h.size() * 2);
        const double b = ms_since(t0);
        t0 = std::chrono::steady_clock::now();
        overwrites.begin_step();
        for (auto& h : host)  // every buffer gets micro-step 0's D2H
            overwrites.overwritten(h.data(), h.size() * 2);
        zeroed = overwrites.zero_rest();
        const double a = ms_since(t0);
        if (r > 0) {
            before.push_back(b);
            after.push_back(a);
        }
    }
    const double mb = median(before);
    std::printf("memset: %zu buffers, %.1f MB: upstream memset %.2f ms (%.2f GB/s), T13 %.4f ms (%zu bytes zeroed)\n",
                host.size(),
                static_cast<double>(bytes) / 1e6,
                mb,
                static_cast<double>(bytes) / mb / 1e6,
                median(after),
                zeroed);
}

void mode_accum(const std::vector<Param>& params, int pin_cpus, int reps) {
    std::vector<Layer> layers = make_layers(params);
    std::vector<Layer> copy = layers;
    std::size_t elems = 0;
    for (const auto& l : layers)
        elems += l.elems;
    std::vector<double> before, after;
    int team = 0;
    std::thread([&] {
        pin(pin_cpus);
        for (int r = 0; r <= reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            for (auto& l : layers)
                for (std::size_t t = 0; t < l.grads.size(); ++t)
                    upstream_accumulate_bf16(l.grads[t].data(), l.staged[t].data(), l.grads[t].size());
            const double b = ms_since(t0);
            t0 = std::chrono::steady_clock::now();
            for (auto& l : copy)
                for (std::size_t t = 0; t < l.grads.size(); ++t)
                    team = std::max(team,
                                    surogate_fork::host_grad_add_bf16(l.grads[t].data(),
                                                                      l.staged[t].data(),
                                                                      l.grads[t].size(),
                                                                      nullptr));
            const double a = ms_since(t0);
            if (r > 0) {
                before.push_back(b);
                after.push_back(a);
            }
        }
    }).join();
    std::uint64_t hb = 0xcbf29ce484222325ull, ha = hb;
    for (std::size_t i = 0; i < layers.size(); ++i)
        for (std::size_t t = 0; t < layers[i].grads.size(); ++t) {
            hb = fnv1a(layers[i].grads[t].data(), layers[i].grads[t].size() * 2, hb);
            ha = fnv1a(copy[i].grads[t].data(), copy[i].grads[t].size() * 2, ha);
        }
    const double mb = median(before), ma = median(after);
    const double per_layer = static_cast<double>(layers.size());
    std::printf("accum: %zu layers, %zu elements: upstream %.2f ms/layer (%.3f ns/elem, %.2f GB/s at 6 B/elem), "
                "T14 %.2f ms/layer (%.3f ns/elem, %.2f GB/s, team %d), %.2fx, hash %016llx %s\n",
                layers.size(),
                elems,
                mb / per_layer,
                mb * 1e6 / static_cast<double>(elems),
                6.0 * static_cast<double>(elems) / mb / 1e6,
                ma / per_layer,
                ma * 1e6 / static_cast<double>(elems),
                6.0 * static_cast<double>(elems) / ma / 1e6,
                team,
                mb / ma,
                static_cast<unsigned long long>(ha),
                ha == hb ? "(equal)" : "(DIFFER)");
}

void mode_adamw(const std::vector<Param>& params, int pin_cpus, int reps) {
    std::size_t total = 0;
    for (const auto& p : params)
        total += p.n;
    std::vector<std::vector<std::uint16_t>> grad, master16;
    std::vector<std::vector<float>> master32;
    std::uint64_t seed = 100;
    for (const auto& p : params) {
        grad.emplace_back(p.n);
        fill_bf16(grad.back(), seed++);
        master16.emplace_back(p.n, bf16_bits(0.01f));
        master32.emplace_back(p.n, 0.01f);
    }
    std::vector<float> m16(total, 0.0f), v16(total, 0.0f), m32(total, 0.0f), v32(total, 0.0f);
    std::vector<double> t16, t32;
    std::thread([&] {
        pin(pin_cpus);
        for (int r = 0; r <= reps; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            std::size_t off = 0;
            for (std::size_t i = 0; i < params.size(); ++i) {
                optimizers::cpu_adamw_step_bf16_param(master16[i].data(),
                                                      grad[i].data(),
                                                      m16.data() + off,
                                                      v16.data() + off,
                                                      params[i].n,
                                                      1e-4f,
                                                      0.9f,
                                                      0.999f,
                                                      r + 1,
                                                      1e-8f,
                                                      0.01f,
                                                      1.0f / 4096.0f);
                off += params[i].n;
            }
            const double a = ms_since(t0);
            t0 = std::chrono::steady_clock::now();
            off = 0;
            for (std::size_t i = 0; i < params.size(); ++i) {
                optimizers::cpu_adamw_step_bf16(master32[i].data(),
                                                grad[i].data(),
                                                m32.data() + off,
                                                v32.data() + off,
                                                params[i].n,
                                                1e-4f,
                                                0.9f,
                                                0.999f,
                                                r + 1,
                                                1e-8f,
                                                0.01f,
                                                1.0f / 4096.0f);
                off += params[i].n;
            }
            const double b = ms_since(t0);
            if (r > 0) {
                t16.push_back(a);
                t32.push_back(b);
            }
        }
    }).join();
    std::uint64_t h16 = fnv1a(m16.data(), total * 4, fnv1a(v16.data(), total * 4));
    std::uint64_t h32 = fnv1a(m32.data(), total * 4, fnv1a(v32.data(), total * 4));
    for (std::size_t i = 0; i < params.size(); ++i) {
        h16 = fnv1a(master16[i].data(), master16[i].size() * 2, h16);
        h32 = fnv1a(master32[i].data(), master32[i].size() * 4, h32);
    }
    const double a = median(t16), b = median(t32);
    std::printf("adamw simd=%s: %zu elements: BF16 master %.1f ms (%.3f ns/elem, %.2f GB/s at 22 B/elem), "
                "FP32 master %.1f ms (%.3f ns/elem, %.2f GB/s at 28 B/elem), hash %016llx %016llx\n",
                surogate_fork::cpu_adamw_simd_enabled() ? "1" : "0",
                total,
                a,
                a * 1e6 / static_cast<double>(total),
                22.0 * static_cast<double>(total) / a / 1e6,
                b,
                b * 1e6 / static_cast<double>(total),
                28.0 * static_cast<double>(total) / b / 1e6,
                static_cast<unsigned long long>(h16),
                static_cast<unsigned long long>(h32));
}

void mode_group(const std::vector<Param>& params, int pin_cpus, int reps) {
    std::vector<Layer> layers = make_layers(params);
    std::size_t total = 0;
    for (const auto& p : params)
        total += p.n;
    std::vector<std::uint16_t> master(total, bf16_bits(0.01f)), grad(total);
    fill_bf16(grad, 7);
    std::vector<float> m(total, 0.0f), v(total, 0.0f);
    const surogate_fork::AdamWStep s{1e-4f, 0.9f, 0.999f, 0.1f, 0.001f, 1e-8f, 1.0f - 1e-6f, 1.0f / 4096.0f};
    static int store = 0;  // the DslGradStore whose address is the serial group key
    for (const bool grouped : {false, true}) {
        const void* key = grouped ? &store : nullptr;
        std::vector<double> accum_ms, adamw_ms;
        int accum_team = 0, adamw_team = 0;
        std::thread([&] {
            pin(pin_cpus);
            surogate_fork::set_omp_serial_group(key);
            // The callback thread lives as long as the rank thread, as the CUDA driver's does, and takes one
            // step's additions at a time: `go` hands them over, `done` hands the step back.
            std::binary_semaphore go(0), done(0);
            bool stop = false;
            double a = 0.0;
            std::thread callback([&] {
                pin(pin_cpus);
                for (;;) {
                    go.acquire();
                    if (stop) return;
                    const auto t0 = std::chrono::steady_clock::now();
                    for (int micro = 1; micro < 4; ++micro)
                        for (auto& l : layers)
                            for (std::size_t t = 0; t < l.grads.size(); ++t)
                                accum_team = surogate_fork::host_grad_add_bf16(l.grads[t].data(),
                                                                               l.staged[t].data(),
                                                                               l.grads[t].size(),
                                                                               key);
                    a = ms_since(t0);
                    done.release();
                }
            });
            for (int r = 0; r <= reps; ++r) {
                go.release();
                done.acquire();  // wait_all_offloads
                const auto t0 = std::chrono::steady_clock::now();
                for (std::size_t off = 0; off < total;) {  // per-tensor regions, as step_cpu_streaming
                    const std::size_t n = std::min<std::size_t>(total - off, 1u << 24);
                    adamw_team = surogate_fork::adamw_step_bf16_param(master.data() + off,
                                                                      grad.data() + off,
                                                                      m.data() + off,
                                                                      v.data() + off,
                                                                      n,
                                                                      s);
                    off += n;
                }
                if (r > 0) {
                    accum_ms.push_back(a);
                    adamw_ms.push_back(ms_since(t0));
                }
            }
            stop = true;
            go.release();
            callback.join();
        }).join();
        std::printf("group=%s: callback adds (3 micro-steps) %.1f ms on a team of %d, rank AdamW %.1f ms on a team "
                    "of %d\n",
                    grouped ? "rank" : "none",
                    median(accum_ms),
                    accum_team,
                    median(adamw_ms),
                    adamw_team);
    }
}

void mode_exhaust() {
    std::vector<std::uint16_t> want(65536), got(65536), src(65536);
    std::size_t differ = 0;
    for (std::uint32_t d = 0; d < 65536; ++d) {
        for (std::uint32_t s = 0; s < 65536; ++s) {
            want[s] = static_cast<std::uint16_t>(d);
            src[s] = static_cast<std::uint16_t>(s);
        }
        got = want;
        upstream_accumulate_bf16(want.data(), src.data(), want.size());
        surogate_fork::host_grad_add_bf16(got.data(), src.data(), got.size(), nullptr);
        for (std::uint32_t s = 0; s < 65536; ++s)
            differ += want[s] != got[s];
    }
    std::printf("exhaust: 4294967296 BF16 pairs, %zu results differ from upstream's loop\n", differ);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s memset|accum|adamw|group|exhaust SHAPE PIN [REPS]\n", argv[0]);
        return 2;
    }
    const std::string mode = argv[1];
    if (mode == "exhaust") {
        mode_exhaust();
        return 0;
    }
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s %s SHAPE PIN [REPS]\n", argv[0], mode.c_str());
        return 2;
    }
    const std::vector<Param> params = shape(argv[2]);
    const int pin_cpus = std::atoi(argv[3]);
    const int reps = argc > 4 ? std::atoi(argv[4]) : 5;
    std::printf("[%s pin=%d reps=%d] ", argv[2], pin_cpus, reps);
    if (mode == "memset") {
        mode_memset(params, reps);
    } else if (mode == "accum") {
        mode_accum(params, pin_cpus, reps);
    } else if (mode == "adamw") {
        mode_adamw(params, pin_cpus, reps);
    } else if (mode == "group") {
        mode_group(params, pin_cpus, reps);
    } else {
        std::fprintf(stderr, "unknown MODE %s\n", mode.c_str());
        return 2;
    }
    return 0;
}
