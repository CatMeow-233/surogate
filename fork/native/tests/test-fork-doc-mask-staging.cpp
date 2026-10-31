// Fork-owned (ledger T17, T18). Host-only cases for the cap-and-pad document geometry in
// surogate_fork/doc_mask_staging.h; they join dsl-run-state-tests ([doc-mask]) and need no GPU.
// fork/patches/tests/native/test_doc_mask_staging.py runs the same geometry on CPU together with
// GraphExecutor's doc-masking functions as compiled from graph_executor.cpp, and compares the
// attention each captured micro-step reads with the attention eager mode runs.

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

#include "surogate_fork/doc_mask_staging.h"

namespace {

using surogate_fork::doc_mask::PadRoute;

// Position ids of one micro-step: each row lists its documents' lengths; a negative length continues
// the previous row's document (positions keep counting), as the packer writes a long document.
std::vector<std::int32_t> positions(const std::vector<std::vector<int>>& rows, int T) {
    std::vector<std::int32_t> pos;
    int carried = 0;
    for (const auto& row : rows) {
        int filled = 0;
        for (int len : row) {
            const bool continued = len < 0;
            const int n = continued ? -len : len;
            int p = continued ? carried : 0;
            for (int i = 0; i < n; ++i) pos.push_back(p++);
            carried = p;
            filled += n;
        }
        REQUIRE(filled == T);
    }
    return pos;
}

}  // namespace

TEST_CASE("doc-mask scan: every row closes its documents, per-row list without resets", "[doc-mask]") {
    const int B = 3, T = 8;
    // Rows interior to one long document: positions continue across rows, no in-row reset.
    const auto long_doc = positions({{8}, {-8}, {-8}}, T);
    const auto docs = surogate_fork::doc_mask::scan_micro_step_documents(long_doc.data(), B, T);
    CHECK_FALSE(docs.has_in_row_boundaries);
    CHECK(docs.cu_seqlens == surogate_fork::doc_mask::per_row_cu_seqlens(B, T));
    CHECK(docs.cu_seqlens == std::vector<std::int32_t>{0, 8, 16, 24});

    const auto packed = positions({{3, 5}, {-2, 6}, {8}}, T);
    const auto packed_docs = surogate_fork::doc_mask::scan_micro_step_documents(packed.data(), B, T);
    CHECK(packed_docs.has_in_row_boundaries);
    CHECK(packed_docs.cu_seqlens == std::vector<std::int32_t>{0, 3, 8, 10, 16, 24});
    CHECK(packed_docs.num_docs() == 5);
}

TEST_CASE("doc-mask plan: a boundary-free step after a capture with documents stages per-row lists",
          "[doc-mask]") {
    // Upstream assigned {0, B*T} here (one document over every row). The plan keeps the scanned per-row
    // list: B documents, which fit any cap a capture with documents produced (at least B+1).
    const int B = 2, T = 8, cap = 4;
    const auto rows = positions({{8}, {-8}}, T);
    const auto docs = surogate_fork::doc_mask::scan_micro_step_documents(rows.data(), B, T);
    const auto plan = surogate_fork::doc_mask::plan_captured_step(cap, docs.has_in_row_boundaries, docs.num_docs());
    CHECK_FALSE(plan.recapture);
    CHECK(plan.stage_slices);
    CHECK(docs.cu_seqlens == std::vector<std::int32_t>{0, 8, 16});

    // More documents than the cap: recapture. A dense capture: documents recapture, none replays as is.
    CHECK(surogate_fork::doc_mask::plan_captured_step(cap, true, cap + 1).recapture);
    CHECK_FALSE(surogate_fork::doc_mask::plan_captured_step(cap, true, cap).recapture);
    CHECK(surogate_fork::doc_mask::plan_captured_step(0, true, B + 1).recapture);
    const auto dense = surogate_fork::doc_mask::plan_captured_step(0, false, B);
    CHECK_FALSE(dense.recapture);
    CHECK_FALSE(dense.stage_slices);
}

TEST_CASE("doc-mask capture: a boundary-free micro-step stays varlen only under a pad-mode capture",
          "[doc-mask]") {
    CHECK(surogate_fork::doc_mask::rows_stay_varlen_in_capture(/*pad_max_docs=*/4, /*capturing=*/true, /*B=*/2));
    CHECK(surogate_fork::doc_mask::rows_stay_varlen_in_capture(4, true, 4));
    CHECK_FALSE(surogate_fork::doc_mask::rows_stay_varlen_in_capture(4, true, 5));   // would not fit the slice
    CHECK_FALSE(surogate_fork::doc_mask::rows_stay_varlen_in_capture(4, false, 2));  // eager: dense, as upstream
    CHECK_FALSE(surogate_fork::doc_mask::rows_stay_varlen_in_capture(0, true, 2));   // pad mode off
}

TEST_CASE("doc-mask route: overflowing the cap outside a capture takes exact geometry", "[doc-mask]") {
    CHECK(surogate_fork::doc_mask::pad_mode_route(17, 0, false) == PadRoute::Exact);
    CHECK(surogate_fork::doc_mask::pad_mode_route(5, 4, false) == PadRoute::Padded);
    CHECK(surogate_fork::doc_mask::pad_mode_route(5, 4, true) == PadRoute::Padded);
    CHECK(surogate_fork::doc_mask::pad_mode_route(6, 4, false) == PadRoute::ExactEager);
    CHECK(surogate_fork::doc_mask::pad_mode_route(6, 4, true) == PadRoute::OverflowInCapture);
}
