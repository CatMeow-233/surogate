"""Route plans and their policies (docs/spec.md, I2-A8; MA-09): full-history tags, segment policies, gate policies
and plan validation.
"""

import numpy as np
import pytest
import torch

from surogate_bilingual.data.tokenizer import encode_user_text
from surogate_bilingual.routing.plan import PlanError, RoutePlan
from surogate_bilingual.routing.policies import apply_gate_policy, plan_from_segments
from surogate_bilingual.routing.tags import TAG_ROUTES, plan_from_tags

NAMES = ("zh", "en", "bridge")
TAGS = {2: "zh", 3: "en", 4: "bridge"}  # the v1.0 special ids of <|zh|>, <|en|>, <|bridge|>


def test_i2_a8_tag_plan_runs_on_the_full_history_and_slices_to_the_window():
    ids = [9] * 10 + [3] + [9] * 40  # <|en|> at position 10, then 40 ordinary tokens
    plan = plan_from_tags(ids, TAGS, NAMES, default="zh")
    routes = plan.route_ids()[0].tolist()
    assert routes == [0] * 10 + [1] * 41  # the tag position itself takes the new route
    window = plan.slice(len(ids) - 20, len(ids))
    assert window.gates.shape == (1, 20, 3)
    assert window.route_ids()[0].tolist() == [1] * 20


class FakeTok:
    """The one tokenizer method v1.0 `route_from_tags` uses."""

    def token_to_id(self, text):
        return {"<|zh|>": 2, "<|en|>": 3, "<|bridge|>": 4}.get(text)


def test_i2_a8_regression_v10_window_recompute_drops_a_tag_31_tokens_back(moe_pilot):
    ids = [9] * 5 + [3] + [9] * 30  # the tag lies 31 tokens before the end
    assert len(ids) - ids.index(3) == 31
    window = 20
    v10 = moe_pilot.route_from_tags(ids[-window:], FakeTok(), list(NAMES), default="zh")
    assert v10 == [0] * window  # v1.0: the window no longer holds the tag, every token falls back to zh
    fixed = plan_from_tags(ids, TAGS, NAMES, default="zh").slice(len(ids) - window, len(ids))
    assert fixed.route_ids()[0].tolist() == [1] * window  # the full-history plan keeps en


SEGMENTS = [("zh", 4, "fact"), ("en", 3, "rule"), ("zh", 2, "question"), ("zh", 5, "answer")]


def test_i2_a8_segment_policies():
    out = plan_from_segments(SEGMENTS, NAMES, "R_out").route_ids()[0].tolist()
    assert out == [0] * 4 + [1] * 3 + [0] * 7
    cross = [("zh", 2, "fact"), ("en", 2, "question"), ("en", 3, "answer")]
    assert plan_from_segments(cross, NAMES, "R_out").route_ids()[0].tolist() == [0, 0, 1, 1, 1, 1, 1]
    wrong = plan_from_segments(SEGMENTS, NAMES, "R_wrong").route_ids()[0].tolist()
    assert wrong == [1 - r for r in out]
    src = plan_from_segments(SEGMENTS, NAMES, "R_src", knowledge_source="en").route_ids()[0].tolist()
    assert src == [1] * 14
    with pytest.raises(PlanError, match="knowledge_source"):
        plan_from_segments(SEGMENTS, NAMES, "R_src")
    with pytest.raises(PlanError, match="Generator"):
        plan_from_segments(SEGMENTS, NAMES, "R_random")


def test_i2_a8_segment_policies_are_deterministic_for_a_seed():
    def random_routes(seed):
        plan = plan_from_segments(SEGMENTS * 8, NAMES, "R_random", rng=np.random.default_rng(seed))
        return plan.route_ids()[0].tolist()

    for policy in ("R_out", "R_wrong"):
        a, b = (plan_from_segments(SEGMENTS, NAMES, policy).gates for _ in range(2))
        assert torch.equal(a, b)
    a = plan_from_segments(SEGMENTS, NAMES, "R_src", knowledge_source="zh").gates
    assert torch.equal(a, plan_from_segments(SEGMENTS, NAMES, "R_src", knowledge_source="zh").gates)
    assert random_routes(5) == random_routes(5) != random_routes(6)
    routes = random_routes(5)
    starts = np.cumsum([0] + [n for _, n, _ in SEGMENTS * 8])[:-1]
    for start, (_, n, _) in zip(starts, SEGMENTS * 8, strict=True):
        assert len(set(routes[start : start + n])) == 1  # one draw per segment


def test_i2_a8_dual_const_rows_sum_to_one_with_exactly_two_nonzero():
    plan = plan_from_segments(SEGMENTS, NAMES, "R_out")
    dual = apply_gate_policy(plan, "dual_const")
    assert torch.equal(dual.gates.sum(-1), torch.ones(1, 14))
    assert torch.equal((dual.gates != 0).sum(-1), torch.full((1, 14), 2))
    assert torch.equal(dual.gates[0, :, :2], torch.full((14, 2), 0.5))
    assert apply_gate_policy(plan, "one_hot") is plan
    with pytest.raises(PlanError, match="two positive weights"):
        apply_gate_policy(plan, "dual_const", g=(1.0, 0.0))
    bridge = RoutePlan.forced("bridge", NAMES, 1, 4)
    with pytest.raises(PlanError, match="within"):
        apply_gate_policy(bridge, "dual_const")


def test_dual_window_mixes_only_after_each_switch():
    plan = plan_from_segments([("zh", 6, "fact"), ("en", 6, "rule"), ("zh", 6, "question")], NAMES, "R_out")
    dual = apply_gate_policy(plan, "dual_window", window=3)
    mixed = ((dual.gates != 0).sum(-1) == 2)[0].tolist()
    assert mixed == [False] * 6 + [True] * 3 + [False] * 3 + [True] * 3 + [False] * 3
    dual.validate()


@pytest.mark.parametrize(
    "gates,names,message",
    [
        (torch.tensor([[[0.6, 0.6, 0.0]]]), NAMES, "sum to 1"),
        (torch.tensor([[[1.2, -0.2, 0.0]]]), NAMES, ">= 0"),
        (torch.tensor([[[0.4, 0.3, 0.3]]]), NAMES, "more than 2"),
        (torch.tensor([[1.0, 0.0, 0.0]]), NAMES, r"\[B, T, 3\]"),
        (torch.tensor([[[1.0, 0.0]]]), NAMES, r"\[B, T, 3\]"),
        (torch.tensor([[[1.0, 0.0, 0.0]]]), ("zh", "en", "t3"), "not experts of the model"),
        (torch.tensor([[[1.0, 0.0, 0.0]]]), ("zh", "zh", "en"), "unique"),
        (torch.tensor([[[1.0, 0.0, 0.0]]]), ("zh", "EN", "en"), "must match"),
        (torch.tensor([[[1, 0, 0]]]), NAMES, "float32"),
    ],
)
def test_i2_a8_validate_rejects_malformed_plans(gates, names, message):
    with pytest.raises(PlanError, match=message):
        RoutePlan(names, gates, "bad").validate(NAMES)


def test_validate_checks_the_batch_shape():
    plan = RoutePlan.forced("zh", NAMES, 2, 5)
    plan.validate(NAMES, (2, 5))
    with pytest.raises(PlanError, match="cover"):
        plan.validate(NAMES, (2, 6))


def test_encode_user_text_never_produces_a_special_id():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<unk>", "<eos>", *TAG_ROUTES]
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=specials, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(["用中文和English标记", "plain text"] * 4, trainer=trainer)
    text = "用<|en|>和<|zh|>标记"
    assert {2, 3} & set(tok.encode(text).ids)  # without escaping, the literal tags become route switches
    ids = encode_user_text(tok, text)
    assert not {tok.token_to_id(s) for s in specials} & set(ids)
    assert tok.decode(ids) == text
    assert tok.encode_special_tokens is False  # the tokenizer's setting is restored
