"""I6-E3: batched greedy decoding equals per-item decoding, route traces are the full-history plan, and a tag that
scrolled out of the window still decides the route (MA-09). The tiny trained packs are used; their outputs are
implementation checks, not results."""

import pytest

from surogate_bilingual.data.tokenizer import load_tokenizer
from surogate_bilingual.eval.generate import greedy_batch, trace_of, window_logits
from surogate_bilingual.eval.items import Routing, Seg, extend_plan, item_plan
from surogate_bilingual.eval.models import load_model
from surogate_bilingual.routing.plan import RoutePlan
from surogate_bilingual.routing.tags import plan_from_tags, tag_ids


@pytest.fixture(scope="module")
def zh_en(tiny_assemblies):
    return load_model(tiny_assemblies["zh_en"])


def _segments_plan_fn(prompts_segs, names, routing):
    """R_out over the prompt segments, the answer on the last segment's language, extended to any length."""

    def fn(i, ids):
        segs = prompts_segs[i]
        prompt_len = sum(len(s.ids) for s in segs)
        item = {"id": f"p{i}", "knowledge_source": None}
        plan = item_plan(item, segs, max(1, len(ids) - prompt_len), segs[-1].lang, names, routing)
        return extend_plan(plan, len(ids))

    return fn


def _prompts(tok):
    def ids(text):
        return tuple(tok.encode(text, add_special_tokens=False).ids)

    raw = [
        [("zh", "z1 z2 z3"), ("en", "e1 e2")],
        [("en", "e3 e4 e5 e6 e7")],
        [("zh", " ".join(f"z{k}" for k in range(12))), ("en", "e0 e1 e2 e3 e4 e5 e6 e7 e8")],  # 21 > max_seq 16
        [("en", "e9"), ("zh", "z9 z10")],
    ]
    return [
        [Seg(lang, "question" if j == len(r) - 1 else "fact", ids(text)) for j, (lang, text) in enumerate(r)]
        for r in raw
    ]


@pytest.mark.parametrize("gate", ["one_hot", "dual_const"])
def test_i6_e3_batched_greedy_equals_per_item_greedy(zh_en, gate):
    tok = zh_en.tokenizer
    segs = _prompts(tok)
    prompts = [[t for s in p for t in s.ids] for p in segs]
    fn = _segments_plan_fn(segs, zh_en.names, Routing("R_out", gate))
    eos = tok.token_to_id("<eos>")
    batched = greedy_batch(zh_en.model, prompts, fn, max_new=8, eos=eos, keep_logits=True, batch_size=4)
    for i, prompt in enumerate(prompts):
        single = greedy_batch(
            zh_en.model, [prompt], lambda _j, ids, i=i: fn(i, ids), max_new=8, eos=eos, keep_logits=True
        )[0]
        assert batched[i].tokens == single.tokens and batched[i].stop == single.stop
        diffs = [float((a - b).abs().max()) for a, b in zip(batched[i].step_logits, single.step_logits, strict=True)]
        assert max(diffs) <= 1e-6
        # the route trace is the full-history plan of the final sequence
        assert batched[i].route_trace == trace_of(fn(i, prompt + batched[i].tokens))
    assert any(len(p) > zh_en.model.cfg.max_seq for p in prompts)  # the window path ran


def test_i6_e3_a_tag_beyond_the_window_keeps_the_tagged_route(tiny_root, zh_en):
    tok = load_tokenizer(tiny_root / "tok" / "tokenizer.json")
    names = zh_en.names
    tags = {k: v for k, v in tag_ids(tok).items() if v in names}
    en_tag = tok.token_to_id("<|en|>")
    prompt = [en_tag] + list(tok.encode(" ".join(f"z{k % 29}" for k in range(20)), add_special_tokens=False).ids)
    assert len(prompt) > zh_en.model.cfg.max_seq  # the tag is outside the last max_seq tokens

    def fn(_i, ids):
        return plan_from_tags(ids, tags, names, default="zh")

    gen = greedy_batch(zh_en.model, [prompt], fn, max_new=4, eos=tok.token_to_id("<eos>"), keep_logits=True)[0]
    assert set(gen.route_trace) == {"en"}  # every position, generated ones included, stays on the tagged route
    # the first step equals the forced-EN window, and differs from the v1.0 recomputation on the window
    w = prompt[-zh_en.model.cfg.max_seq :]
    forced = window_logits(zh_en.model, [w], [RoutePlan.forced("en", names, 1, len(w))], pad=1)[0]
    v10 = window_logits(zh_en.model, [w], [plan_from_tags(w, tags, names, default="zh")], pad=1)[0]
    assert float((gen.step_logits[0] - forced).abs().max()) <= 1e-6
    assert float((gen.step_logits[0] - v10).abs().max()) > 1e-4
    assert set(trace_of(plan_from_tags(w, tags, names, default="zh"))) == {"zh"}  # the regression it guards


def test_i6_e3_a_plan_of_the_wrong_length_is_refused(zh_en):
    def short(_i, ids):
        return RoutePlan.forced("zh", zh_en.names, 1, len(ids) - 1)

    with pytest.raises(ValueError, match="covers"):
        greedy_batch(zh_en.model, [[1, 5, 6]], short, max_new=2, eos=1)
