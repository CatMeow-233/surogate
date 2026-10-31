// Fork-owned (ledger T20). Whether train_step_graphed runs its full-step eager warmup.
//
// Upstream (282f9901) runs one eager forward+backward over every micro-step before it captures the
// full-step graph, then discards it (zero_grads, RNG state restored), and shrinks the DSL stack to
// the high-water mark that pass measured. A model with capture-unsafe ops (MoE routing and grouped
// GEMMs, EP dispatch/combine) never captures: the step then runs eagerly, gs.captured stays false,
// and the warmup ran again on every step, an extra fwd+bwd whose results are thrown away.
// has_capture_unsafe_ops() is known before the first warmup (the graphs compile in
// allocate_run_state), so after one warmup (which keeps the stack shrink) such a model skips it.
//
// Identical results only where the warmup leaves no state behind. With expert parallelism it
// advances the sticky EP plan ages (ep_plan_refresh_interval, 16 by default), so a skipped warmup
// would refresh the plans on a different cadence: EP keeps the warmup unless the upstream switch
// SUROGATE_DSL_GRAPH_SKIP_WARMUP=1 skips it everywhere.
#pragma once

namespace surogate_fork::full_step_warmup {

struct Inputs {
    bool captured = false;          ///< the full-step graph exists (upstream: no warmup)
    bool skip_env = false;          ///< SUROGATE_DSL_GRAPH_SKIP_WARMUP=1 (upstream: no warmup, no stack shrink)
    bool force_env = false;         ///< SUROGATE_FORK_FORCE_GRAPH_WARMUP=1: upstream behaviour
    bool capture_unsafe = false;    ///< dsl_model->has_capture_unsafe_ops(): the step will run eagerly
    bool expert_parallel = false;   ///< EPSize > 1
    bool warmed_up = false;         ///< this trainer context already ran the warmup once
};

inline bool run_warmup(const Inputs& in) {
    if (in.captured || in.skip_env) return false;
    if (in.force_env) return true;
    return !(in.capture_unsafe && !in.expert_parallel && in.warmed_up);
}

}  // namespace surogate_fork::full_step_warmup
