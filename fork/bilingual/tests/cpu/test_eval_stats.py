"""I6-E5: the confirmatory statistics of MS §9.2–9.6 (SCI-07, SCI-09). Numbers here are checks of the statistical
machinery on constructed or simulated data, not experimental results."""

import math
import time

import numpy as np
import pytest

from surogate_bilingual.eval import stats


@pytest.mark.parametrize(
    ("t", "df", "tail"),
    [(2.919986, 2, 0.05), (4.302653, 2, 0.025), (2.015048, 5, 0.05), (6.205, 2, 0.0125), (3.163, 5, 0.0125)],
)
def test_i6_e5_t_cdf_matches_known_quantiles(t, df, tail):
    assert abs(stats.t_sf(t, df) - tail) <= 1e-5
    assert abs(stats.t_cdf(t, df) - (1 - tail)) <= 1e-5
    assert abs(stats.t_cdf(-t, df) - tail) <= 1e-5


def test_i6_e5_t_cdf_closed_forms_and_ppf():
    # df = 1 is Cauchy, df = 2 and df = 3 have closed forms
    for t in (-3.0, -0.4, 0.0, 0.7, 5.0):
        assert abs(stats.t_cdf(t, 1) - (0.5 + math.atan(t) / math.pi)) <= 1e-12
        assert abs(stats.t_cdf(t, 2) - (0.5 + t / (2 * math.sqrt(2 + t * t)))) <= 1e-12
        s = t / math.sqrt(3)
        assert abs(stats.t_cdf(t, 3) - (0.5 + (s / (1 + s * s) + math.atan(s)) / math.pi)) <= 1e-12
    for p in (0.0125, 0.5, 0.95, 0.9875):
        for df in (2, 5, 30):
            assert abs(stats.t_cdf(stats.t_ppf(p, df), df) - p) <= 1e-10


def test_i6_e5_paired_t_one_sided_equals_a_hand_computation():
    r = stats.paired_t_one_sided([1.0, 2.0, 3.0, 4.0])
    # mean 2.5, sd sqrt(5/3), se sqrt(5/12), t = 2.5 / sqrt(5/12) = sqrt(15); df 3: closed form of the t CDF
    assert (r.n, r.df) == (4, 3)
    assert abs(r.mean - 2.5) <= 1e-15 and abs(r.sd - math.sqrt(5 / 3)) <= 1e-14
    assert abs(r.t - math.sqrt(15)) <= 1e-12
    s = math.sqrt(15) / math.sqrt(3)
    assert abs(r.p - (0.5 - (s / (1 + s * s) + math.atan(s)) / math.pi)) <= 1e-12
    # non-inferiority shifts by the margin; zero spread is decided by the sign
    nr = stats.paired_t_one_sided([-0.01, 0.0, 0.01], margin=0.02)
    assert abs(nr.t - 0.02 / (0.01 / math.sqrt(3))) <= 1e-12
    assert stats.paired_t_one_sided([0.1, 0.1, 0.1]).p == 0.0
    assert stats.paired_t_one_sided([-0.1, -0.1, -0.1]).p == 1.0
    with pytest.raises(ValueError):
        stats.paired_t_one_sided([0.1])


def test_i6_e5_clustered_bootstrap_is_deterministic():
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(40), 10)
    diff = rng.normal(0.05, 0.3, 400) + np.repeat(rng.normal(0, 0.2, 40), 10)
    a = stats.clustered_bootstrap(diff, clusters, n=2000, seed=7)
    b = stats.clustered_bootstrap(diff, clusters, n=2000, seed=7)
    c = stats.clustered_bootstrap(diff, clusters, n=2000, seed=8)
    assert a == b and (a.lo, a.hi) != (c.lo, c.hi)
    assert a.n_clusters == 40 and a.n_items == 400 and a.lo < a.estimate < a.hi
    # string labels give the same clusters as integers
    assert stats.clustered_bootstrap(diff, np.array([f"t{k}" for k in clusters]), n=2000, seed=7) == a


def test_i6_e5_clustered_bootstrap_coverage_on_simulated_clustered_data():
    began = time.perf_counter()
    rng = np.random.default_rng(20261102)
    n_sims, n_clusters, per, mu = 400, 60, 8, 0.1
    clusters = np.repeat(np.arange(n_clusters), per)
    hits = 0
    for sim in range(n_sims):
        effects = np.repeat(rng.normal(0.0, 0.3, n_clusters), per)
        diff = mu + effects + rng.normal(0.0, 0.5, n_clusters * per)
        ci = stats.clustered_bootstrap(diff, clusters, n=2000, seed=sim)
        hits += ci.lo <= mu <= ci.hi
    elapsed = time.perf_counter() - began
    assert 0.92 <= hits / n_sims <= 0.98, hits / n_sims
    assert elapsed <= 20.0


def test_i6_e5_tost_decides_constructed_cases():
    assert stats.tost([0.001, -0.002, 0.0, 0.002, -0.001], margin=0.03, alpha=0.0125)
    assert not stats.tost([0.05, 0.06, 0.04, 0.055], margin=0.03, alpha=0.0125)  # clearly outside
    assert not stats.tost([0.1, -0.1, 0.05], margin=0.03, alpha=0.0125)  # too wide to tell
    # exactly the (1 - 2 alpha) interval: equivalent iff -m < lo and hi < m
    d = [0.01, 0.02, 0.015, 0.012]
    lo, hi = stats.paired_t_one_sided(d).ci(0.05)
    assert stats.tost(d, margin=hi + 1e-9, alpha=0.05) and not stats.tost(d, margin=hi - 1e-9, alpha=0.05)
    with pytest.raises(ValueError):
        stats.tost(d, margin=0.0, alpha=0.05)


def test_i6_e5_three_way_decision_on_constructed_cases():
    assert stats.decide([0.08, 0.09, 0.10, 0.085, 0.095], 0.0125, 0.05)["decision"] == "superior"
    assert stats.decide([0.001, -0.002, 0.0, 0.002, -0.001], 0.0125, 0.03)["decision"] == "equivalent"
    assert stats.decide([0.1, -0.1, 0.05], 0.0125, 0.05)["decision"] == "inconclusive"
    # significant but below the SESOI: never "superior"; equivalent when inside +-SESOI, else inconclusive
    out = stats.decide([0.03, 0.031, 0.029, 0.03, 0.032], 0.0125, 0.05)
    assert out["p"] < 0.0125 and out["decision"] == "equivalent"
    out = stats.decide([0.04, 0.06, 0.03, 0.05, 0.045], 0.0125, 0.05)
    assert out["p"] < 0.0125 and out["estimate"] < 0.05 and out["decision"] == "inconclusive"


# MS §9.3: H_A -> H_B (branch 1, alpha 0.0125), H_C (branch 2, alpha 0.0125); a fully rejected branch passes its
# alpha to the other branch. (p_A, p_B, p_C) -> rejected set, and the alpha at which H_A was tested when it matters.
GATE_TABLE = [
    ((0.010, 0.010, 0.010), {"H_A", "H_B", "H_C"}, None),  # everything; H_C ends with 0.025
    ((0.020, 0.001, 0.001), {"H_A", "H_B", "H_C"}, 0.025),  # H_C first; its alpha rescues H_A, then H_B
    ((0.020, 0.001, 0.020), set(), None),  # nothing: H_B never receives alpha while H_A stands
    ((0.010, 0.020, 0.020), {"H_A"}, 0.0125),  # H_B gets 0.0125 only, H_C keeps 0.0125
    ((0.010, 0.010, 0.020), {"H_A", "H_B", "H_C"}, 0.0125),  # branch 1 done: H_C tested at 0.025
    ((0.010, 0.020, 0.001), {"H_A", "H_B", "H_C"}, 0.0125),  # H_C's alpha goes to H_B (H_A already rejected)
    ((0.500, 0.0001, 0.010), {"H_C"}, None),  # H_B's tiny p is never tested without H_A
    ((0.013, 0.500, 0.012), {"H_A", "H_C"}, 0.025),  # H_C at 0.0125, then H_A at 0.025
]


@pytest.mark.parametrize(("pvalues", "rejected", "alpha_a"), GATE_TABLE)
def test_i6_e5_graph_gatekeeper_reproduces_the_hand_table(pvalues, rejected, alpha_a):
    out = stats.graph_gatekeeper(dict(zip(("H_A", "H_B", "H_C"), pvalues, strict=True)))
    assert {h for h, d in out.items() if d.rejected} == rejected
    if alpha_a is not None:
        assert abs(out["H_A"].alpha - alpha_a) <= 1e-15
    # the rejected set does not depend on the order in which rejectable hypotheses are taken
    rev = stats.graph_gatekeeper(
        dict(zip(("H_C", "H_B", "H_A"), pvalues[::-1], strict=True)),
        stats.Graph(alpha={"H_C": 0.0125, "H_B": 0.0, "H_A": 0.0125}, edges=stats.MS_GRAPH.edges),
    )
    assert {h for h, d in rev.items() if d.rejected} == rejected


def test_i6_e5_alpha_transfer_amounts():
    out = stats.graph_gatekeeper({"H_A": 0.01, "H_B": 0.02, "H_C": 0.02})
    assert out["H_B"].alpha == 0.0125 and out["H_C"].alpha == 0.0125  # final alphas, nothing more rejected
    out = stats.graph_gatekeeper({"H_A": 0.01, "H_B": 0.01, "H_C": 0.02})
    assert out["H_C"].rejected and abs(out["H_C"].alpha - 0.025) <= 1e-15
    with pytest.raises(ValueError):
        stats.Graph(alpha={"a": 0.1}, edges={"a": {"a": 1.0}})


def test_i6_e5_run_level_t_test_holds_its_type_one_error_with_between_run_variance():
    """Under the null with replicate-level random effects (SD 0.05 on accuracy) and Bernoulli items, the run-level
    one-sided t test at alpha 0.05 rejects 0.05 +- 0.02 of 4000 simulated studies (R = 3)."""
    rng = np.random.default_rng(12)
    sims, R, items = 4000, 3, 300
    difficulty = rng.uniform(0.2, 0.8, items)
    rejections = 0
    for _ in range(sims):
        run = rng.normal(0.0, 0.05, (2, R, 1))  # each arm's replicate effect, same distribution under the null
        p = np.clip(difficulty[None, None, :] + run, 0.0, 1.0)
        correct = rng.random((2, R, items)) < p
        diffs = correct[0].mean(1) - correct[1].mean(1)
        rejections += stats.paired_t_one_sided(diffs).p <= 0.05
    assert abs(rejections / sims - 0.05) <= 0.02, rejections / sims


def test_i6_e5_power_and_mde_are_monotone():
    sd, alpha = 0.03, 0.0125
    by_r = [stats.power_paired_t(sd, 0.05, alpha, r) for r in range(3, 11)]
    by_delta = [stats.power_paired_t(sd, d, alpha, 4) for d in (0.0, 0.01, 0.02, 0.04, 0.08)]
    assert all(a < b for a, b in zip(by_r, by_r[1:], strict=False))
    assert all(a < b for a, b in zip(by_delta, by_delta[1:], strict=False))
    assert abs(by_delta[0] - alpha) <= 1e-6  # at delta = 0 the power is the size
    mdes = [stats.mde(sd, alpha, r) for r in range(3, 11)]
    assert all(a > b for a, b in zip(mdes, mdes[1:], strict=False))
    assert abs(stats.power_paired_t(sd, mdes[0], alpha, 3) - 0.8) <= 1e-6
    # the exact (quadrature) power agrees with simulation
    sim = stats.power_paired_t(sd, 0.05, alpha, 5, sims=40_000, seed=1)
    assert abs(sim - by_r[2]) <= 0.01


def test_i6_e5_replicate_rule_and_the_df2_critical_value():
    assert abs(stats.t_ppf(1 - 0.0125, 2) - 6.205) <= 1e-3 and abs(stats.t_ppf(1 - 0.0125, 5) - 3.163) <= 1e-3
    small = stats.replicate_rule(sd=0.01, sesoi=0.05)
    assert small["R"] == 3 and small["label"] == "powered"
    large = stats.replicate_rule(sd=0.05, sesoi=0.05)
    assert large["R"] is None and large["label"] == "underpowered, exploratory" and large["mde_at_max"] > 0.05


def test_g1p_gate_and_futility_rule():
    gate = stats.pilot_gate({"B0": [0.30, 0.34, 0.32], "B2": [0.40, 0.31, 0.32], "B12a": [0.20, 0.25, 0.30]})
    assert gate["best"] == "B2" and abs(gate["threshold"] - 0.325) <= 1e-12 and gate["passed"]
    assert not stats.pilot_gate({"B0": [0.3, 0.3, 0.3]})["passed"]
    assert stats.futility(0.02, 0.01)["stop_branch_1"]
    assert not stats.futility(0.03, 0.01)["stop_branch_1"]
    assert stats.discordance({"a": True, "b": False, "c": True}, {"a": True, "b": True, "d": False}) == (0.5, 2)
