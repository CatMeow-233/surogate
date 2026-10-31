// GPU cases for the GQA shapes the fork registers in the serve engine (ledger T16), run on
// upstream's GQA test harness.
//
// csrc/src/testing/serve/ops/test_gqa_attention.cpp (sinfer_gqa_attention_test) checks Qwen3-8B's
// 32q/8kv at head dim 128 among its cases, but not Qwen3-14B's 40q/8kv or the 128-wide multi-head
// shape. This file compiles that harness unchanged, its main renamed, and runs the same kinds of
// cases on all three fork shapes against its double-precision oracle: decode and verify rows,
// appending (A1) and over a populated cache (A3); 7-63-column prompt segments and wide prompts;
// sliding windows; batched rows with masked columns and permuted table rows; an e4m3 cache
// (appended against cached bits); batched int8 rows, which stay on the generic kernel either
// way; and packed multi-prompt rounds against the segments' own calls, bit for bit.
//
// fork/native/cmake/fork_native.cmake builds it as sinfer_gqa_fork_shapes_test and registers it
// with ctest twice: with SUROGATE_SERVE_GQA_FORK_SHAPES=0 (the routes the shapes took before
// they were registered) and set to 1 (the registered small-T kernels for BF16 and e4m3).
#define main sinfer_gqa_attention_test_main
#include "ops/test_gqa_attention.cpp"
#undef main

namespace {

int run_fork_shape(const Geometry& geometry, std::uint32_t seed) {
    int failures = 0;
    // Widths: decode, MTP/DFlash verify (the lane step is six for a group of five, so 8 and 16
    // also cover the chunked route), short prompt segments, and prompts past the prompt kernel's
    // first width (64 columns).
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 8, 16, 17, 40, 66}) {
        const AttentionCase near{tokens, 61, 128, seed + static_cast<std::uint32_t>(tokens)};
        failures += run_a1_case(geometry, DType::BF16, near, MappingPattern::Fragmented);
        failures += run_a3_case(geometry, DType::BF16, near, MappingPattern::Fragmented);
    }
    // Histories that cross pages, the 128-key partitions and several splits.
    for (const std::int32_t tokens : {1, 3, 40}) {
        const AttentionCase far{tokens, 1800, 2048, seed + 100u + static_cast<std::uint32_t>(tokens)};
        failures += run_a1_case(geometry, DType::BF16, far, MappingPattern::Fragmented);
        failures += run_a3_case(geometry, DType::BF16, far, MappingPattern::Fragmented);
        const AttentionCase windowed{tokens, 1800, 2048, seed + 200u + static_cast<std::uint32_t>(tokens), 300};
        failures += run_a1_case(geometry, DType::BF16, windowed, MappingPattern::Fragmented);
    }
    failures += run_batch_case(geometry, DType::BF16,
        {1, {2000}, {1}, {0}, MappingPattern::Fragmented, seed + 300u});
    failures += run_batch_case(geometry, DType::BF16,
        {1, {61, 1500, 700, 3000}, {1, 1, 0, 1}, {3, 0, 2, 1}, MappingPattern::Fragmented, seed + 301u});
    failures += run_batch_case(geometry, DType::BF16,
        {4, {300, 1100}, {4, 2}, {1, 0}, MappingPattern::Fragmented, seed + 302u});
    failures += run_batch_case(geometry, DType::BF16,
        {6, {61, 127, 255}, {6, 3, 0}, {2, 0, 1}, MappingPattern::Fragmented, seed + 303u});
    failures += run_batch_case(geometry, DType::BF16,
        {16, {49, 2041}, {16, 7}, {1, 0}, MappingPattern::Fragmented, seed + 304u});
    // int8: the small-T int8 kernel serves only 256-wide heads, so these rows must keep the
    // generic route with the switch on as well (a regression here is an exception, not a
    // mismatch). The batched form is the one upstream runs for int8 on such shapes.
    failures += run_batch_case(geometry, DType::I8,
        {2, {900, 300}, {2, 1}, {1, 0}, MappingPattern::Fragmented, seed + 400u});
    failures += run_batch_case(geometry, DType::I8,
        {3, {61, 127}, {3, 2}, {1, 0}, MappingPattern::Fragmented, seed + 401u});
    failures += verify_fp8_current_tokens_match_cached(geometry.head_dim, geometry.q_heads,
                                                       geometry.kv_heads);
    return failures;
}

// Multi-prompt rounds pack once a shape is registered (ops::gqa_attention_packs_prompt), which the
// switch turns on for these shapes. The packed launch must leave every output and cache bit its
// segments' own calls leave, as upstream's verify_packed_prompt_invariance checks for its shapes;
// this is that check on the fork shapes, over BF16 and e4m3 caches, with fewer and shorter
// chunks. With the switch off nothing may pack.
int verify_fork_shape_packed_prompts(const Geometry& geometry) {
    struct Chunk {
        int history;
        int length;
    };
    const std::vector<Chunk> chunks = {{0, 200}, {127, 41}, {3967, 385}, {300, 7}, {4095, 64},
                                       {0, 63},  {127, 65}, {1000, 200}, {50, 5}};
    const char* raw    = std::getenv("SUROGATE_SERVE_GQA_FORK_SHAPES");
    const bool enabled = raw != nullptr && std::strcmp(raw, "1") == 0;
    constexpr int pages_per_row = 72;
    const int rows = static_cast<int>(chunks.size());
    const int dim = geometry.head_dim, heads = geometry.q_heads, kv_heads = geometry.kv_heads;
    int failures = 0;
    for (const auto dtype : {DType::BF16, DType::FP8_E4M3FN}) {
        DeviceArena arena(1U << 30);
        int columns = 0;
        std::vector<int> column_of(rows);
        for (int r = 0; r < rows; ++r) { column_of[r] = columns; columns += chunks[r].length; }
        Tensor q = arena.alloc(DType::BF16, {dim, heads, columns});
        Tensor k = arena.alloc(DType::BF16, {dim, kv_heads, columns});
        Tensor v = arena.alloc(DType::BF16, {dim, kv_heads, columns});
        Tensor positions = arena.alloc(DType::I32, {columns});
        Tensor reference = arena.alloc(DType::BF16, {dim, heads, columns});
        Tensor packed_out = arena.alloc(DType::BF16, {dim, heads, columns});
        unsigned seed = 5150U + dim + heads + kv_heads;
        const auto fill = [&](Tensor tensor) {
            std::vector<std::uint16_t> values(tensor.numel());
            for (auto& value : values) {
                seed = seed * 1664525U + 1013904223U;
                value = f32_to_bf16((int(seed >> 16) - 32768) / 8192.0F);
            }
            CUDA_CHECK(cudaMemcpy(tensor.data, values.data(), tensor.bytes(), cudaMemcpyHostToDevice));
        };
        fill(q);
        fill(k);
        fill(v);
        std::vector<int> pos(columns);
        for (int r = 0; r < rows; ++r) {
            for (int t = 0; t < chunks[r].length; ++t) { pos[column_of[r] + t] = chunks[r].history + t; }
        }
        CUDA_CHECK(cudaMemcpy(positions.data, pos.data(), positions.bytes(), cudaMemcpyHostToDevice));
        PagedKVBatchLayerView cache;
        cache.head_dim = dim;
        cache.num_kv_heads = kv_heads;
        cache.dtype = dtype;
        std::vector<int> first_page(rows), row_pages(rows);
        int pool = 0;
        for (int r = 0; r < rows; ++r) {
            row_pages[r] = (chunks[r].history + chunks[r].length + kPagedKVPageSize - 1) / kPagedKVPageSize;
            first_page[r] = pool;
            pool += row_pages[r];
        }
        cache.k_pages = arena.alloc(dtype, {dim, kPagedKVPageSize, kv_heads, pool});
        cache.v_pages = arena.alloc(dtype, {dim, kPagedKVPageSize, kv_heads, pool});
        cache.block_tables = arena.alloc(DType::I32, {pages_per_row, rows});
        std::vector<int> table(static_cast<std::size_t>(pages_per_row) * rows), row_ids(rows);
        for (int r = 0; r < rows; ++r) {
            row_ids[r] = r;
            for (int p = 0; p < pages_per_row; ++p) {
                const int local = p < row_pages[r] ? (p * 7 + 3) % row_pages[r] : 0;
                table[static_cast<std::size_t>(r) * pages_per_row + p] = first_page[r] + local;
            }
        }
        CUDA_CHECK(cudaMemcpy(cache.block_tables.data, table.data(), cache.block_tables.bytes(),
                              cudaMemcpyHostToDevice));
        Tensor row_table = arena.alloc(DType::I32, {rows});
        CUDA_CHECK(cudaMemcpy(row_table.data, row_ids.data(), row_table.bytes(), cudaMemcpyHostToDevice));
        for (int r = 0; r < rows; ++r) {
            if (chunks[r].history == 0) { continue; }
            DeviceArena history(std::size_t(chunks[r].history) * dim * kv_heads * 4 + (1U << 20));
            Tensor hk = history.alloc(DType::BF16, {dim, kv_heads, chunks[r].history});
            Tensor hv = history.alloc(DType::BF16, {dim, kv_heads, chunks[r].history});
            Tensor hp = history.alloc(DType::I32, {chunks[r].history});
            fill(hk);
            fill(hv);
            std::vector<int> history_pos(chunks[r].history);
            for (int i = 0; i < chunks[r].history; ++i) { history_pos[i] = i; }
            CUDA_CHECK(cudaMemcpy(hp.data, history_pos.data(), hp.bytes(), cudaMemcpyHostToDevice));
            PagedKVLayerView row_view;
            row_view.head_dim = dim;
            row_view.num_kv_heads = kv_heads;
            row_view.dtype = dtype;
            row_view.k_pages = cache.k_pages;
            row_view.v_pages = cache.v_pages;
            row_view.block_table = cache.block_tables.slice(1, r, 1).view({pages_per_row});
            ops::gqa_kv_append(hk, hv, hp, row_view, nullptr);
        }
        Tensor k_before = arena.alloc(dtype, {dim, kPagedKVPageSize, kv_heads, pool});
        Tensor v_before = arena.alloc(dtype, {dim, kPagedKVPageSize, kv_heads, pool});
        CUDA_CHECK(cudaMemcpy(k_before.data, cache.k_pages.data, k_before.bytes(), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(v_before.data, cache.v_pages.data, v_before.bytes(), cudaMemcpyDeviceToDevice));
        const auto restore_cache = [&] {
            CUDA_CHECK(cudaMemcpy(cache.k_pages.data, k_before.data, k_before.bytes(), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(cache.v_pages.data, v_before.data, v_before.bytes(), cudaMemcpyDeviceToDevice));
        };
        const auto cache_bytes = [&] {
            auto keys = from_device<std::uint8_t>(cache.k_pages.data, cache.k_pages.bytes());
            const auto values = from_device<std::uint8_t>(cache.v_pages.data, cache.v_pages.bytes());
            keys.insert(keys.end(), values.begin(), values.end());
            return keys;
        };
        const float scale = attention_scale(geometry);
        const auto envelope_of = [&](int r) {
            const auto seen = static_cast<std::uint32_t>(chunks[r].history + chunks[r].length);
            return ops::GqaExecutionEnvelope{seen, seen, 0};
        };
        const auto own_call = [&](int r, Tensor& into, WorkspaceArena& scratch) {
            Tensor out = into.slice(2, column_of[r], chunks[r].length);
            ops::gqa_attention(q.slice(2, column_of[r], chunks[r].length),
                               k.slice(2, column_of[r], chunks[r].length),
                               v.slice(2, column_of[r], chunks[r].length),
                               positions.slice(0, column_of[r], chunks[r].length), Tensor{},
                               row_table.slice(0, r, 1), scale, cache, envelope_of(r), scratch, out, nullptr);
        };
        {
            WorkspaceArena scratch(64U << 20);
            for (int r = 0; r < rows; ++r) { own_call(r, reference, scratch); }
            cuda_synchronize();
        }
        const auto expected = from_device<std::uint16_t>(reference.data, reference.numel());
        const auto expected_cache = cache_bytes();
        for (const std::int32_t max_lanes : {0, 1, 3}) {
            restore_cache();
            CUDA_CHECK(cudaMemset(packed_out.data, 0xFF, packed_out.bytes()));
            WorkspaceArena scratch(512U << 20);
            std::vector<ops::GqaPackedSegment> segments;
            for (int r = 0; r < rows; ++r) {
                if (ops::gqa_attention_packs_prompt(dim, heads, kv_heads, dtype, chunks[r].length, envelope_of(r))) {
                    segments.push_back({column_of[r], chunks[r].length, r, envelope_of(r)});
                } else {
                    own_call(r, packed_out, scratch);
                }
            }
            const std::string label = std::string("fork packed prompts ") + geometry.name + " dtype=" +
                                      std::to_string(int(dtype)) + " max_lanes=" + std::to_string(max_lanes);
            if (enabled ? segments.size() < 4 || segments.size() == chunks.size() : !segments.empty()) {
                std::cerr << label << ": " << segments.size() << " of " << chunks.size()
                          << " chunks pack with the switch " << (enabled ? "on" : "off") << "\n";
                ++failures;
            }
            if (!segments.empty()) {
                ops::gqa_attention_packed_prompts(q, k, v, positions, segments, scale, cache, scratch,
                                                  packed_out, nullptr, max_lanes);
            }
            cuda_synchronize();
            if (from_device<std::uint16_t>(packed_out.data, packed_out.numel()) != expected) {
                std::cerr << label << ": outputs differ from the segments' own calls\n";
                ++failures;
            }
            if (cache_bytes() != expected_cache) {
                std::cerr << label << ": the cache differs from the segments' own calls'\n";
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const char* raw = std::getenv("SUROGATE_SERVE_GQA_FORK_SHAPES");
    std::cout << "SUROGATE_SERVE_GQA_FORK_SHAPES=" << (raw == nullptr ? "(unset)" : raw) << "\n";
    int failures = 0;
    failures += run_fork_shape(Geometry{"qwen3_8b", 32, 8, 128}, 9100U);
    failures += run_fork_shape(Geometry{"qwen3_14b", 40, 8, 128}, 9600U);
    failures += run_fork_shape(Geometry{"mha_128", 32, 32, 128}, 10100U);
    for (const Geometry geometry : {Geometry{"qwen3_8b", 32, 8, 128}, Geometry{"qwen3_14b", 40, 8, 128},
                                    Geometry{"mha_128", 32, 32, 128}}) {
        failures += verify_fork_shape_packed_prompts(geometry);
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " gqa fork shapes\n";
    return failures == 0 ? 0 : 1;
}
