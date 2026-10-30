// Copyright (c) 2026, Invergent SA
// SPDX-License-Identifier: Apache-2.0
//
// Times the CPU optimizer step of cpu_training (AdamW8BitOptimizer::step_cpu_streaming:
// cpu_gradient_norm_squared over every host gradient, then one cpu_adamw_step_bf16 per
// parameter) from R rank threads. Each rank thread first pins itself the way
// set_cpu_affinity() does through nvmlDeviceSetCpuAffinity: on the calling thread only.
// Link it with a cpu_adamw.cpp object built from the trainer's compile command (with or
// without -fopenmp); run.sh does that.
//
// usage: pinbench R PIN SOCKETS STEPS SHAPE
//   R        rank threads, all running the step at the same time
//   PIN      CPUs in each rank's mask, 0 = leave the process mask
//   SOCKETS  disjoint masks; rank r uses CPUs [g*PIN, g*PIN+PIN) with g = r % SOCKETS
//   STEPS    optimizer steps; the first is a warm-up and is not timed
//   SHAPE    toy:N         one N-element tensor and 64 tensors of 4096 elements
//            flat:N:K      K tensors of N elements (for the serial threshold)
//            llama:L:H:I:V a Llama-shaped model (embedding, L layers, final norm, lm_head)
// Output, for the slowest rank: the median step time and its norm and update parts, then the
// norm as a hexfloat and an FNV-1a hash of every master weight and moment after the last step,
// so builds can be compared bit for bit. Gradients span 2^-24..2^0 so that the norm's last bits
// depend on its summation order. Every rank steps the same data; when ranks still end with
// different bits (a norm that depends on the team or on thread timing), the line ends with
// "RANKS DIFFER".
#include "runtime/optimizers/cpu_adamw.h"

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_bf16.h>

namespace {

struct Param {
    std::string name;
    std::size_t n;
};

std::vector<Param> shape(const std::string& spec) {
    std::vector<Param> params;
    if (spec.rfind("toy:", 0) == 0) {
        params.push_back({"big", std::strtoull(spec.c_str() + 4, nullptr, 10)});
        for (int i = 0; i < 64; ++i)
            params.push_back({"small" + std::to_string(i), 4096});
        return params;
    }
    if (spec.rfind("flat:", 0) == 0) {
        std::size_t n = 0, k = 0;
        if (std::sscanf(spec.c_str(), "flat:%zu:%zu", &n, &k) != 2) {
            std::fprintf(stderr, "bad SHAPE %s\n", spec.c_str());
            std::exit(2);
        }
        for (std::size_t i = 0; i < k; ++i)
            params.push_back({"t" + std::to_string(i), n});
        return params;
    }
    long L = 0, h = 0, im = 0, V = 0;
    if (std::sscanf(spec.c_str(), "llama:%ld:%ld:%ld:%ld", &L, &h, &im, &V) != 4) {
        std::fprintf(stderr, "bad SHAPE %s\n", spec.c_str());
        std::exit(2);
    }
    params.push_back({"embedding", static_cast<std::size_t>(V * h)});
    for (long l = 0; l < L; ++l) {
        const std::string p = "l" + std::to_string(l) + ".";
        params.push_back({p + "in_norm", static_cast<std::size_t>(h)});
        params.push_back({p + "q", static_cast<std::size_t>(h * h)});
        params.push_back({p + "k", static_cast<std::size_t>(h * h / 4)});
        params.push_back({p + "v", static_cast<std::size_t>(h * h / 4)});
        params.push_back({p + "o", static_cast<std::size_t>(h * h)});
        params.push_back({p + "post_norm", static_cast<std::size_t>(h)});
        params.push_back({p + "gate", static_cast<std::size_t>(h * im)});
        params.push_back({p + "up", static_cast<std::size_t>(h * im)});
        params.push_back({p + "down", static_cast<std::size_t>(h * im)});
    }
    params.push_back({"final_norm", static_cast<std::size_t>(h)});
    params.push_back({"lm_head", static_cast<std::size_t>(V * h)});
    return params;
}

std::uint64_t fnv1a(const void* data, std::size_t bytes, std::uint64_t h) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i)
        h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s R PIN SOCKETS STEPS SHAPE\n", argv[0]);
        return 2;
    }
    const int R = std::atoi(argv[1]);
    const int pin = std::atoi(argv[2]);
    const int sockets = std::atoi(argv[3]);
    const int steps = std::atoi(argv[4]);
    const std::vector<Param> params = shape(argv[5]);
    std::size_t total = 0;
    for (const auto& p : params)
        total += p.n;

    std::barrier sync(R);
    std::vector<double> median(R), median_norm(R), median_update(R);
    std::vector<double> norm(R);
    std::vector<std::uint64_t> hash(R);
    std::vector<std::thread> ranks;
    for (int r = 0; r < R; ++r) {
        ranks.emplace_back([&, r] {
            if (pin > 0) {
                cpu_set_t mask;
                CPU_ZERO(&mask);
                const int first = (r % sockets) * pin;
                for (int c = 0; c < pin; ++c)
                    CPU_SET(first + c, &mask);
                if (pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) != 0) {
                    std::perror("pthread_setaffinity_np");
                    std::exit(1);
                }
            }
            // Gradients vary by element and tensor so that the norm's summation order shows in its bits.
            std::unordered_map<std::string, Tensor> grads;
            std::vector<std::string> names;
            std::vector<std::vector<float>> master;
            std::vector<std::vector<nv_bfloat16>> grad;
            for (std::size_t t = 0; t < params.size(); ++t) {
                const auto& p = params[t];
                master.emplace_back(p.n, 0.01f);
                grad.emplace_back(p.n);
                for (std::size_t i = 0; i < p.n; ++i) {
                    const std::uint64_t h = (i + 1) * 0x9e3779b97f4a7c15ull ^ (t + 1) * 0xc2b2ae3d27d4eb4full;
                    const float mantissa = 1.0f + static_cast<float>((h >> 8) & 127) / 128.0f;
                    const float value = std::ldexp(mantissa, -static_cast<int>((h >> 20) % 25));
                    grad.back()[i] = __float2bfloat16((h >> 40) & 1 ? -value : value);
                }
                Tensor g{};
                g.DType = ETensorDType::BF16;
                g.Rank = 1;
                g.Sizes[0] = static_cast<long>(p.n);
                g.Data = reinterpret_cast<std::byte*>(grad.back().data());
                grads.emplace(p.name, g);
                names.push_back(p.name);
            }
            std::vector<float> m(total, 0.0f), v(total, 0.0f);
            std::vector<double> times, norm_times, update_times;
            double nsq = 0.0;
            for (int s = 0; s < steps; ++s) {
                sync.arrive_and_wait();
                const auto t0 = std::chrono::steady_clock::now();
                nsq = optimizers::cpu_gradient_norm_squared(grads, names);
                const auto tn = std::chrono::steady_clock::now();
                const float scale = 1.0f / (1.0f + static_cast<float>(nsq) * 1e-6f);
                std::size_t off = 0;
                for (std::size_t i = 0; i < params.size(); ++i) {
                    optimizers::cpu_adamw_step_bf16(master[i].data(),
                                                    grad[i].data(),
                                                    m.data() + off,
                                                    v.data() + off,
                                                    params[i].n,
                                                    1e-4f,
                                                    0.9f,
                                                    0.999f,
                                                    s + 1,
                                                    1e-8f,
                                                    0.01f,
                                                    scale);
                    off += params[i].n;
                }
                const auto t1 = std::chrono::steady_clock::now();
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                norm_times.push_back(std::chrono::duration<double, std::milli>(tn - t0).count());
                update_times.push_back(std::chrono::duration<double, std::milli>(t1 - tn).count());
                sync.arrive_and_wait();
            }
            const auto mid = [](std::vector<double>& x) {
                std::sort(x.begin() + 1, x.end());  // the first step is the warm-up
                return x[1 + (x.size() - 1) / 2];
            };
            median[r] = mid(times);
            median_norm[r] = mid(norm_times);
            median_update[r] = mid(update_times);
            norm[r] = nsq;
            std::uint64_t h = 0xcbf29ce484222325ull;
            for (const auto& w : master)
                h = fnv1a(w.data(), w.size() * sizeof(float), h);
            hash[r] = fnv1a(m.data(), m.size() * sizeof(float), fnv1a(v.data(), v.size() * sizeof(float), h));
        });
    }
    for (auto& t : ranks)
        t.join();
    // Every rank steps identical data, so any difference comes from the summation order of the norm.
    int differing = 0;
    for (int r = 1; r < R; ++r)
        differing += norm[r] != norm[0] || hash[r] != hash[0];
    const int slowest = static_cast<int>(std::max_element(median.begin(), median.end()) - median.begin());
    std::printf("R=%d pin=%d sockets=%d %s tensors=%zu elements=%zu: median %.2f ms (norm %.2f, update %.2f), "
                "norm %a, hash %016llx%s\n",
                R,
                pin,
                sockets,
                argv[5],
                params.size(),
                total,
                median[slowest],
                median_norm[slowest],
                median_update[slowest],
                norm[0],
                static_cast<unsigned long long>(hash[0]),
                differing ? ", RANKS DIFFER" : "");
    return 0;
}
