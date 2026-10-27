// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <cstddef>
#include <omp.h>

#ifndef _OPENMP
#error "The training OpenMP target did not enable OpenMP for its CXX source"
#endif

int training_openmp_version() {
    return _OPENMP;
}

int training_openmp_workers() {
    int workers = 0;
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        workers = omp_get_num_threads();
    }
    return workers;
}

// Exact integer arithmetic avoids making this a floating-point reduction test.
// This exercises compiled host parallel work; it is not an AdamW math substitute.
void training_openmp_update(int* values, std::size_t count) {
#pragma omp parallel for schedule(static) num_threads(2)
    for (std::size_t i = 0; i < count; ++i) {
        values[i] = values[i] * 3 + static_cast<int>(i % 17);
    }
}
