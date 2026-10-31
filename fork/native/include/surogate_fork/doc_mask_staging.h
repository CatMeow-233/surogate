// Fork-owned (ledger T17, T18). Host-side document geometry of the full-step CUDA graph's cap-and-pad
// document masking. Plain C++ with no CUDA dependency: py_train.cpp's train_step_graphed,
// GraphExecutor (graph_executor.cpp) and the tests call the same functions.
//
// Cap-and-pad, as upstream built it: when a full-step graph is captured with packed documents,
// GraphExecutor::enable_doc_masking_pad_to_max(cap, GA) gives every micro-step a slice of cap+1
// int32 offsets in mCuSeqlensCpu. Each captured forward records an H2D copy from its slice, and the
// attention kernels are captured with num_docs = cap and max_seqlen = B*T, so one graph replays any
// topology of at most cap documents per micro-step; train_step_graphed rewrites the slices before
// every launch, padding unused slots with B*T (empty documents).
//
// What T17 fixes (upstream 282f9901):
//  - A step with no in-row position reset, after a capture with documents, staged {0, B*T} for every
//    micro-step: one document over all rows, so row b attended causally to rows 0..b-1 (FA2), or rows
//    1..B-1 got no attention output at all (mem_eff, Hs > 256, which computes min(max_seqlen, T)
//    queries per document). The per-row list {0, T, ..., B*T} the scan already produced is the eager
//    path's geometry; plan_captured_step keeps it.
//  - A micro-step with no in-row reset during the capture went through clear_doc_masking, so its
//    attention was captured dense and never read its slice: a later step whose same micro-step held
//    documents (at most cap, so no recapture) attended across them. rows_stay_varlen_in_capture lets
//    GraphExecutor capture it varlen with one document per row instead.
// What T18 guards: pad mode stays on for the rest of the run, and an eager call (validate) with more
// documents than the cap had its tail documents dropped (their queries got no attention output).
// pad_mode_route sends such a call through exact geometry instead.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace surogate_fork::doc_mask {

/// One micro-step's documents, scanned as train_step_graphed scans them: [B, T] row-major position
/// ids of one plane; a document ends where the position does not advance by exactly one, and every
/// row closes its last document. cu_seqlens therefore holds at least B+1 offsets, ends at B*T, and is
/// exactly the per-row list {0, T, ..., B*T} when no row has an in-row reset.
struct MicroStepDocuments {
    std::vector<std::int32_t> cu_seqlens;
    bool has_in_row_boundaries = false;

    [[nodiscard]] int num_docs() const {
        return cu_seqlens.empty() ? 0 : static_cast<int>(cu_seqlens.size()) - 1;
    }
};

inline MicroStepDocuments scan_micro_step_documents(const std::int32_t* pos, int B, int T) {
    MicroStepDocuments docs;
    docs.cu_seqlens.push_back(0);
    for (int b = 0; b < B; ++b) {
        const int row_base = b * T;
        int doc_start = row_base;
        for (int t = 1; t < T; ++t) {
            const int idx = row_base + t;
            if (pos[idx] - pos[idx - 1] != 1) {
                const int doc_len = idx - doc_start;
                if (doc_len > 0) docs.cu_seqlens.push_back(docs.cu_seqlens.back() + doc_len);
                doc_start = idx;
                docs.has_in_row_boundaries = true;
            }
        }
        const int last_len = (b + 1) * T - doc_start;
        if (last_len > 0) docs.cu_seqlens.push_back(docs.cu_seqlens.back() + last_len);
    }
    return docs;
}

/// {0, T, 2T, ..., B*T}: one document per row, the geometry of dense per-row causal attention.
inline std::vector<std::int32_t> per_row_cu_seqlens(int B, int T) {
    std::vector<std::int32_t> cu;
    cu.reserve(static_cast<std::size_t>(B > 0 ? B : 0) + 1);
    for (int b = 0; b <= B; ++b) cu.push_back(b * T);
    return cu;
}

/// What a step does with an existing full-step graph.
struct CapturedStepDocPlan {
    bool recapture = false;     ///< destroy the graph: it cannot replay this step's documents
    bool stage_slices = false;  ///< (no recapture) rewrite every micro-step's slice before the launch
};

/// captured_doc_cap: the cap the graph was captured with (0 = captured without documents, so its
/// attention is dense and it reads no slice). has_in_row_boundaries / max_num_docs: this step's scan
/// over all micro-steps (max_num_docs counts the per-row documents too, so it is >= B).
/// A graph captured with documents replays every later step that fits its cap, documents or not: a
/// micro-step without in-row resets stages its per-row list. That list always fits, because the
/// capture saw a micro-step with at least B+1 documents and the cap is at least that count.
inline CapturedStepDocPlan plan_captured_step(int captured_doc_cap, bool has_in_row_boundaries, int max_num_docs) {
    CapturedStepDocPlan plan;
    if (captured_doc_cap > 0) {
        plan.recapture = max_num_docs > captured_doc_cap;
        plan.stage_slices = !plan.recapture;
    } else {
        plan.recapture = has_in_row_boundaries;
    }
    return plan;
}

/// During a capture in pad mode, a micro-step without in-row resets is captured varlen with one
/// document per row (B documents, which must fit the cap), so the graph reads its slice on every
/// launch. Outside a capture (validate, the eager warmup) dense attention is what eager mode runs.
inline bool rows_stay_varlen_in_capture(int pad_max_docs, bool capturing, int B) {
    return pad_max_docs > 0 && capturing && B >= 1 && B <= pad_max_docs;
}

/// How GraphExecutor::set_doc_masking stages `count` = num_docs + 1 offsets.
enum class PadRoute {
    Exact,              ///< pad mode off: upstream's exact geometry
    Padded,             ///< pad mode, count <= cap + 1: the micro-step's slice, padded geometry
    ExactEager,         ///< pad mode, count > cap + 1, not capturing: exact geometry from a separate
                        ///< host buffer; the slices (and the captured copies that read them) stay untouched
    OverflowInCapture,  ///< pad mode, count > cap + 1, capturing: cannot throw under an open capture;
                        ///< upstream's truncation, reported. train_step_graphed sizes the cap from the
                        ///< same scan, so its captures never get here.
};

inline PadRoute pad_mode_route(int count, int pad_max_docs, bool capturing) {
    if (pad_max_docs <= 0) return PadRoute::Exact;
    if (count <= pad_max_docs + 1) return PadRoute::Padded;
    return capturing ? PadRoute::OverflowInCapture : PadRoute::ExactEager;
}

}  // namespace surogate_fork::doc_mask
