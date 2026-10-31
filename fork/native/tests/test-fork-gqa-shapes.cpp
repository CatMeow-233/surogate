// GPU cases for the GQA shapes the fork registers in the serve engine (ledger T16), run on
// upstream's GQA test harness.
//
// csrc/src/testing/serve/ops/test_gqa_attention.cpp (sinfer_gqa_attention_test) checks Qwen3-8B's
// 32q/8kv at head dim 128 among its cases, but not Qwen3-14B's 40q/8kv or the 128-wide multi-head
// shape. This file compiles that harness unchanged, its main renamed, and runs the same kinds of
// cases on all three fork shapes against its double-precision oracle: decode and verify rows,
// appending (A1) and over a populated cache (A3); 7-63-column prompt segments and wide prompts;
// sliding windows; batched rows with masked columns and permuted table rows; an e4m3 cache
// against attention in double over the cache's own codes (decode, verify, short and wide
// prompts, with and without a window), and appended against cached bits; batched int8 rows,
// which stay on the generic kernel either way; and packed multi-prompt rounds against the
// segments' own calls, bit for bit.
//
// fork/native/cmake/fork_native.cmake builds it as sinfer_gqa_fork_shapes_test and registers it
// with ctest twice: with SUROGATE_SERVE_GQA_FORK_SHAPES=0 (the routes the shapes took before
// they were registered; the multi-head shape, which upstream refused, must be refused) and set
// to 1 (the registered small-T kernels for BF16 and e4m3, the multi-head shape served).
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

// An e4m3 cache against attention in double over the codes the cache holds, as upstream's
// verify_fp8_wide_prompts checks Qwen3-8B's decode and verify rows (its cases 819 and 820): the
// widths the small-T kernel, its chunks and the prompt tiles serve with the switch on, and the
// generic kernel with it off (1, 3, 6, 17), and one past the prompt kernel's first width (66),
// over a history that crosses pages, 128-key partitions and splits, with and without a window,
// on scattered pages. Each width runs at the default switches and batch-invariant; the wide one
// also with the prompt kernel's FP8 query-key product, against an oracle that rounds the queries
// the same way, as upstream does. A misread head, group or page is off by order one; the bounds
// are upstream's.
int verify_fork_shape_fp8_against_reference(const Geometry& geometry, std::uint32_t seed) {
    struct Case {
        int history;
        int tokens;
        int window;
    };
    const Case cases[] = {{2000, 1, 0},   {2000, 3, 0},   {2000, 6, 0},   {2000, 17, 0},
                          {2000, 66, 0},  {1999, 1, 300}, {1999, 3, 300}, {1999, 6, 300},
                          {1999, 17, 300}, {1999, 66, 300}};
    constexpr int pages = 64;
    constexpr float scale = 1.0F / 16.0F;
    int device = 0, major = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    const int dim = geometry.head_dim, heads = geometry.q_heads, kv_heads = geometry.kv_heads;
    const int group = heads / kv_heads;
    int failures = 0;
    for (const Case& c : cases) {
        const int keys = c.history + c.tokens;
        const std::uint32_t case_seed = seed + 7u * static_cast<std::uint32_t>(c.tokens) +
                                        static_cast<std::uint32_t>(c.window);
        DeviceArena arena(256U << 20);
        Tensor q = arena.alloc(DType::BF16, {dim, heads, c.tokens});
        Tensor k = arena.alloc(DType::BF16, {dim, kv_heads, keys});
        Tensor v = arena.alloc(DType::BF16, {dim, kv_heads, keys});
        Tensor positions = arena.alloc(DType::I32, {keys});
        const std::vector<float> q_values = make_bf16_values(q.numel(), case_seed, -2.0F, 2.0F);
        const std::vector<float> k_values = make_bf16_values(k.numel(), case_seed + 1u, -2.0F, 2.0F);
        const std::vector<float> v_values = make_bf16_values(v.numel(), case_seed + 2u, -1.0F, 1.0F);
        const auto upload = [](Tensor tensor, const std::vector<float>& values) {
            const std::vector<std::uint16_t> bits = to_bf16_bits(values);
            CUDA_CHECK(cudaMemcpy(tensor.data, bits.data(), tensor.bytes(), cudaMemcpyHostToDevice));
        };
        upload(q, q_values);
        upload(k, k_values);
        upload(v, v_values);
        std::vector<int> pos(keys), table(pages);
        for (int t = 0; t < keys; ++t) { pos[t] = t; }
        for (int i = 0; i < pages; ++i) { table[i] = (i * 17 + 3) % pages; }
        CUDA_CHECK(cudaMemcpy(positions.data, pos.data(), positions.bytes(), cudaMemcpyHostToDevice));
        PagedKVLayerView cache;
        cache.head_dim     = dim;
        cache.num_kv_heads = kv_heads;
        cache.dtype        = DType::FP8_E4M3FN;
        cache.k_pages      = arena.alloc(cache.dtype, {dim, kPagedKVPageSize, kv_heads, pages});
        cache.v_pages      = arena.alloc(cache.dtype, {dim, kPagedKVPageSize, kv_heads, pages});
        cache.block_table  = arena.alloc(DType::I32, {pages});
        CUDA_CHECK(cudaMemcpy(cache.block_table.data, table.data(), cache.block_table.bytes(),
                              cudaMemcpyHostToDevice));
        ops::gqa_kv_append(k, v, positions, cache, nullptr);
        cuda_synchronize();

        // The oracle reads the codes the cache holds: [page][kv head][slot][dim].
        const auto k_codes = from_device<std::uint8_t>(cache.k_pages.data, cache.k_pages.numel());
        const auto v_codes = from_device<std::uint8_t>(cache.v_pages.data, cache.v_pages.numel());
        const auto cached = [&](const std::vector<std::uint8_t>& codes, int t, int g, int d) {
            const std::size_t page = static_cast<std::size_t>(table[t / kPagedKVPageSize]);
            return static_cast<double>(fp8_reference_value(
                codes[((page * kv_heads + g) * kPagedKVPageSize + t % kPagedKVPageSize) * dim + d]));
        };
        std::vector<double> kd(static_cast<std::size_t>(keys) * kv_heads * dim), vd(kd.size());
        for (int t = 0; t < keys; ++t) {
            for (int g = 0; g < kv_heads; ++g) {
                for (int d = 0; d < dim; ++d) {
                    const std::size_t i = (static_cast<std::size_t>(t) * kv_heads + g) * dim + d;
                    kd[i] = cached(k_codes, t, g, d);
                    vd[i] = cached(v_codes, t, g, d);
                }
            }
        }
        const auto attend = [&](const std::vector<float>& queries) {
            std::vector<double> reference(q.numel());
            std::vector<double> scores(keys);
            for (int col = 0; col < c.tokens; ++col) {
                const int last  = c.history + col;
                const int first = c.window > 0 ? std::max(0, last - c.window + 1) : 0;
                for (int h = 0; h < heads; ++h) {
                    const int g = h / group;
                    const float* query = queries.data() + (static_cast<std::size_t>(col) * heads + h) * dim;
                    double peak = -std::numeric_limits<double>::infinity();
                    for (int t = first; t <= last; ++t) {
                        const double* key = kd.data() + (static_cast<std::size_t>(t) * kv_heads + g) * dim;
                        double dot = 0;
                        for (int d = 0; d < dim; ++d) { dot += query[d] * key[d]; }
                        scores[t] = dot * scale;
                        peak      = std::max(peak, scores[t]);
                    }
                    double total = 0;
                    for (int t = first; t <= last; ++t) { total += scores[t] = std::exp(scores[t] - peak); }
                    double* row = reference.data() + (static_cast<std::size_t>(col) * heads + h) * dim;
                    for (int t = first; t <= last; ++t) {
                        const double* value = vd.data() + (static_cast<std::size_t>(t) * kv_heads + g) * dim;
                        const double weight = scores[t] / total;
                        for (int d = 0; d < dim; ++d) { row[d] += weight * value[d]; }
                    }
                }
            }
            return reference;
        };
        const std::vector<double> reference = attend(q_values);
        // The prompt kernel's FP8 query-key product rounds each query row to e4m3 against its own
        // absolute maximum; its oracle rounds the queries the same way (upstream's construction).
        const bool fp8_query = major != 9 && c.tokens >= 64;
        std::vector<double> reference_q8;
        if (fp8_query) {
            std::vector<float> q8_values(q_values.size());
            for (std::size_t row = 0; row < q_values.size(); row += dim) {
                float amax = 0.0F;
                for (int d = 0; d < dim; ++d) { amax = std::max(amax, std::abs(q_values[row + d])); }
                const float inv = amax > 0.0F ? 448.0F / amax : 0.0F;
                const float row_scale = amax * (1.0F / 448.0F);
                for (int d = 0; d < dim; ++d) {
                    q8_values[row + d] = fp8_reference_value(fp8_reference_code(q_values[row + d] * inv)) * row_scale;
                }
            }
            reference_q8 = attend(q8_values);
        }

        Tensor query_pos = positions.slice(0, c.history, c.tokens);
        const ops::GqaExecutionEnvelope envelope{1, pages * kPagedKVPageSize, c.window};
        struct Run {
            bool invariant;
            bool fp8_query;
        };
        const bool default_fp8_query = ops::prompt_attention_fp8_query();
        for (const Run run : {Run{false, false}, Run{false, true}, Run{true, false}}) {
            if (run.fp8_query && !fp8_query) { continue; }
            ops::set_batch_invariant(run.invariant);
            ops::set_prompt_attention_fp8_query(run.fp8_query);
            const bool pv_fp8 = run.fp8_query && major == 12 && c.window == 0 && ops::prompt_attention_fp8_pv();
            Tensor out = arena.alloc(DType::BF16, {dim, heads, c.tokens});
            CUDA_CHECK(cudaMemset(out.data, 0xFF, out.bytes()));
            // The engine sizes its plan with the history form (layouts_impl.h), which also covers
            // the prompt tiles of every shorter history.
            WorkspaceArena scratch(std::max<std::size_t>(256, ops::gqa_attention_history_workspace_capacity_bytes(
                dim, heads, kv_heads, cache.dtype, envelope, 1, c.tokens, c.tokens)));
            ops::gqa_attention_cached(q, query_pos, scale, cache, envelope, scratch, out, nullptr);
            cuda_synchronize();
            ops::set_batch_invariant(false);
            ops::set_prompt_attention_fp8_query(default_fp8_query);
            const bool flash = !run.invariant && major == 9 && c.tokens >= 32;
            const bool kernel = !run.invariant && !flash && c.tokens >= 64;
            const bool q8 = kernel && run.fp8_query;
            const bool p8 = q8 && pv_fp8;
            const std::string label = std::string("fork fp8 reference ") + geometry.name +
                                      " T=" + std::to_string(c.tokens) + " keys=" + std::to_string(keys) +
                                      " window=" + std::to_string(c.window) +
                                      " invariant=" + std::to_string(int(run.invariant)) +
                                      (flash ? " flash-attention-3" : p8 ? " prompt kernel fp8 query and probabilities" :
                                       q8 ? " prompt kernel fp8 query" : kernel ? " prompt kernel" : " small-T, tiles or generic");
            failures += verify_attention(label, bf16_bits_to_double(from_device<std::uint16_t>(out.data, out.numel())),
                                         q8 ? reference_q8 : reference,
                                         flash || p8 ? kAttentionFp8QueryCriterion : kAttentionBf16Criterion);
        }
    }
    return failures;
}

// With the switch off the multi-head shape is refused as it was before the fork registered it
// (surogate_fork::gqa_fork_kv_pair_withheld): the workspace plans, the KV append and the attention
// calls throw std::invalid_argument naming the geometry (the messages upstream gave it), and
// nothing packs. The switch-on run serves the same shape through run_fork_shape.
int verify_fork_kv_pair_refused(const Geometry& geometry) {
    const int dim = geometry.head_dim, heads = geometry.q_heads, kv_heads = geometry.kv_heads;
    int failures = 0;
    const auto refused = [&](const std::string& what, const auto& call) {
        try {
            call();
        } catch (const std::invalid_argument& error) {
            if (std::string(error.what()).find("geometry") != std::string::npos) { return; }
            std::cerr << geometry.name << ": " << what << " refused for another reason: " << error.what() << "\n";
            ++failures;
            return;
        }
        std::cerr << geometry.name << ": " << what << " accepted with the switch off\n";
        ++failures;
    };
    const ops::GqaExecutionEnvelope envelope{1, 128};
    for (const auto dtype : {DType::BF16, DType::FP8_E4M3FN, DType::I8}) {
        const std::string name = std::string(" dtype=") + std::to_string(int(dtype));
        refused("the workspace plan" + name, [&] {
            (void)ops::gqa_attention_workspace_capacity_bytes(dim, heads, kv_heads, dtype, envelope, 1, 1, 8);
        });
        refused("the history workspace plan" + name, [&] {
            (void)ops::gqa_attention_history_workspace_capacity_bytes(dim, heads, kv_heads, dtype, envelope, 1, 1, 66);
        });
        if (ops::gqa_attention_packs_prompt(dim, heads, kv_heads, dtype, 40, envelope)) {
            std::cerr << geometry.name << ": a prompt packs with the switch off" << name << "\n";
            ++failures;
        }
    }
    DeviceArena arena(16U << 20);
    Tensor q = arena.alloc(DType::BF16, {dim, heads, 4});
    Tensor k = arena.alloc(DType::BF16, {dim, kv_heads, 4});
    Tensor v = arena.alloc(DType::BF16, {dim, kv_heads, 4});
    Tensor out = arena.alloc(DType::BF16, {dim, heads, 4});
    Tensor positions = arena.alloc(DType::I32, {4});
    const std::vector<int> pos = {0, 1, 2, 3}, table = {1, 0};
    CUDA_CHECK(cudaMemcpy(positions.data, pos.data(), positions.bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(q.data, 0, q.bytes()));
    CUDA_CHECK(cudaMemset(k.data, 0, k.bytes()));
    CUDA_CHECK(cudaMemset(v.data, 0, v.bytes()));
    PagedKVLayerView cache;
    cache.head_dim     = dim;
    cache.num_kv_heads = kv_heads;
    cache.dtype        = DType::BF16;
    cache.k_pages      = arena.alloc(cache.dtype, {dim, kPagedKVPageSize, kv_heads, 2});
    cache.v_pages      = arena.alloc(cache.dtype, {dim, kPagedKVPageSize, kv_heads, 2});
    cache.block_table  = arena.alloc(DType::I32, {2});
    CUDA_CHECK(cudaMemcpy(cache.block_table.data, table.data(), cache.block_table.bytes(), cudaMemcpyHostToDevice));
    WorkspaceArena scratch(1U << 20);
    refused("gqa_kv_append", [&] { ops::gqa_kv_append(k, v, positions, cache, nullptr); });
    refused("gqa_attention_cached", [&] {
        ops::gqa_attention_cached(q, positions, attention_scale(geometry), cache, envelope, scratch, out, nullptr);
    });
    cuda_synchronize();
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
    const bool enabled = surogate_fork::gqa_fork_shapes_enabled();
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
    // The switch withholds exactly the multi-head shape's KV pair, and only while it is off;
    // 128 over 8 KV heads is upstream's Gqa128_16q8 pair.
    if (surogate_fork::gqa_fork_kv_pair_withheld(128, 32) == surogate_fork::gqa_fork_shapes_enabled() ||
        surogate_fork::gqa_fork_kv_pair_withheld(128, 8)) {
        std::cerr << "the fork withholds the wrong KV pairs\n";
        ++failures;
    }
    struct Shape {
        Geometry geometry;
        std::uint32_t seed;
    };
    for (const Shape& shape : {Shape{Geometry{"qwen3_8b", 32, 8, 128}, 9100U},
                               Shape{Geometry{"qwen3_14b", 40, 8, 128}, 9600U},
                               Shape{Geometry{"mha_128", 32, 32, 128}, 10100U}}) {
        const Geometry& geometry = shape.geometry;
        if (surogate_fork::gqa_fork_kv_pair_withheld(geometry.head_dim, geometry.kv_heads)) {
            failures += verify_fork_kv_pair_refused(geometry);
            continue;
        }
        failures += run_fork_shape(geometry, shape.seed);
        failures += verify_fork_shape_fp8_against_reference(geometry, shape.seed + 500U);
        failures += verify_fork_shape_packed_prompts(geometry);
    }
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " gqa fork shapes\n";
    return failures == 0 ? 0 : 1;
}
