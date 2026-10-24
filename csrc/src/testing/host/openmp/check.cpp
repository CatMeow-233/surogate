// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <iostream>
#include <vector>

int training_openmp_version();
int training_openmp_workers();
void training_openmp_update(int* values, std::size_t count);

int main() {
    const int workers = training_openmp_workers();
    if (workers < 2) {
        std::cout << "SKIP: the OpenMP runtime restricted the requested two-worker team to " << workers << '\n';
        return 77;
    }
    if (training_openmp_version() <= 0 || workers != 2) {
        std::cerr << "The training CXX library did not run the requested OpenMP team\n";
        return 1;
    }
    for (std::size_t count : {0, 1, 17, 4097}) {
        std::vector<int> actual(count), expected(count);
        for (std::size_t i = 0; i < count; ++i) {
            actual[i] = static_cast<int>(i % 31) - 15;
            expected[i] = actual[i] * 3 + static_cast<int>(i % 17);
        }
        training_openmp_update(actual.data(), actual.size());
        if (actual != expected) {
            std::cerr << "Parallel host output differs from the serial reference for count=" << count << '\n';
            return 1;
        }
    }
    std::cout << "OpenMP " << training_openmp_version() << ": two workers and exact host outputs verified\n";
    return 0;
}
