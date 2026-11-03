"""Models on the one backend (spec steps S11-S12): the A0 twin against TLC's reference (gate TW-1), the Mamba-2
initialisation, the H-Net pieces (router, boundaries, chunk / upsample, STE, ratio loss, width adapters), causality
and batching invariance for every model kind of both tables, the bilingual trunk against RoutedLM, the arm builder,
and a tiny H-Net that learns. fp64 on the CPU unless a test says otherwise; tiny geometries.
"""

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch
from surogate_tokstudy import kernels
from surogate_tokstudy.arms import ArmSpec
from surogate_tokstudy.codecs import BYTE_VOCAB, PROJECT_SPECIALS
from surogate_tokstudy.models import build, hnet, mamba2, twin_a0
from surogate_tokstudy.models.hnet import HNetConfig, HNetLM
from surogate_tokstudy.models.lm import ByteLM, SubwordLM
from surogate_tokstudy.models.trunk_bilingual import BilingualTrunk
from surogate_tokstudy.models.twin_a0 import A0Twin, twin_config

F64 = torch.float64
REF_FILE = Path(__file__).resolve().parents[3] / "tlc" / "prototypes" / "ref_mamba2.py"
# A two-layer A0 with a short scan chunk, so that 17 positions cross chunk boundaries.
SMALL_TLC = {"num_hidden_layers": 2, "tlc_layer_pattern": "MM", "chunk_size": 8, "vocab_size": 320}
TINY_HNET = {
    "d0": 16,
    "encoder_layers": 1,
    "decoder_layers": 1,
    "ratio": 4.0,
    "head_dim": 8,
    "state": 4,
    "ssd_chunk": 8,
}


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def ref():
    spec = importlib.util.spec_from_file_location("ref_mamba2", REF_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tlc_dirs(tlc_model_dir) -> dict[str, Path]:
    """Two-layer A0 and A0-split model directories, initialised by TLC's init from the same seed, then every norm
    weight and D scaled by its own random factors, so that no two tensors of a layer are alike (TLC's init sets all
    of them to 1, under which a swapped name would go unseen)."""
    from safetensors.torch import load_file, save_file

    dirs = {}
    for row in ("A0", "A0-split"):
        path = tlc_model_dir({**SMALL_TLC, "row": row}, seed=7)
        weights = load_file(path / "model.safetensors")
        g = torch.Generator().manual_seed(11)
        for name in sorted(weights):
            if name.endswith(("ln1_weight", "ln2_weight", "gated_norm_weight", "D_param", "norm.weight")):
                factor = 1 + 0.3 * torch.randn(weights[name].shape, generator=g)
                weights[name] = (weights[name].float() * factor).to(weights[name].dtype)
        save_file(weights, path / "model.safetensors")
        dirs[row] = path
    return dirs


def ids(batch, length, vocab, seed, low=0):
    return torch.randint(low, vocab, (batch, length), generator=torch.Generator().manual_seed(seed))


# --- S11: the A0 twin (TW-1) -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("row", ["A0", "A0-split"])
def test_tw1_twin_logits_equal_the_tlc_reference(ref, tlc_dirs, row):
    """from_tlc_model_dir on a TLC-initialised directory: logits equal ref_mamba2.Reference (scan order) within 1e-9
    relative, fp64, batch 2 x length 17; the twin reads exactly the names TLC writes, and writes them back."""
    path = tlc_dirs[row]
    twin = A0Twin.from_tlc_model_dir(path, dtype=F64)
    reference = ref.Reference(path, dtype=F64)
    x = ids(2, 17, SMALL_TLC["vocab_size"], seed=0)
    with torch.no_grad():
        got, expected = twin(x), reference.logits(x)
    assert float((got - expected).abs().max() / expected.abs().max()) <= 1e-9
    split = row == "A0-split"
    assert set(twin.tlc_names(split=split)) == ref.weight_names(reference.cfg)
    from safetensors.torch import load_file

    stored = load_file(path / "model.safetensors")
    back = twin.tlc_weights(split=split)
    assert all(torch.equal(back[name].to(stored[name].dtype), stored[name]) for name in stored)


def test_twin_geometry_is_tlc_a0(tlc_default_config):
    """Profile "a0" is TLC's A0 geometry; at it the twin has 12 x 2,124,196 + 384 = 25,490,736 non-embedding
    parameters (spec S11), counted on the meta device."""
    assert twin_a0.config_from_tlc(tlc_default_config) == twin_config("a0")
    with torch.device("meta"):
        twin = A0Twin(twin_config("a0"), vocab_size=16384, init_seed=None)
    assert twin.non_embedding_parameters() == 12 * 2_124_196 + 384 == 25_490_736
    assert sum(p.numel() for p in twin.parameters()) == 25_490_736 + 16384 * 384  # tied: one embedding


def test_twin_refuses_rows_it_does_not_model(tlc_default_config):
    with pytest.raises(ValueError, match="all-Mamba"):
        twin_a0.config_from_tlc({**tlc_default_config, "tlc_layer_pattern": "MMMMMAMMMMMA"})
    with pytest.raises(ValueError, match="tied"):
        twin_a0.config_from_tlc({**tlc_default_config, "tie_word_embeddings": False})
    with pytest.raises(ValueError, match="unknown twin keys"):
        twin_config("a0", vocab_size=7)
    with pytest.raises(ValueError, match="without an embedding"):
        A0Twin(twin_config("tiny"), vocab_size=None).embed(torch.zeros(1, 2, dtype=torch.long))


def test_mamba_initialisation_follows_the_tlc_scheme():
    twin = A0Twin(twin_config("tiny"), vocab_size=300, init_seed=3)
    again = A0Twin(twin_config("tiny"), vocab_size=300, init_seed=3)
    other = A0Twin(twin_config("tiny"), vocab_size=300, init_seed=4)
    state, same, different = twin.state_dict(), again.state_dict(), other.state_dict()
    assert all(torch.equal(state[k], same[k]) for k in state)
    assert not torch.equal(
        state["stack.layers.0.mixer.in_proj.weight"], different["stack.layers.0.mixer.in_proj.weight"]
    )
    d, ffn, layers = 32, 48, 2
    twin.requires_grad_(False)
    for layer in twin.stack.layers:
        mixer = layer.mixer
        dt = torch.nn.functional.softplus(mixer.dt_bias)
        assert float(dt.min()) >= mamba2.DT_INIT_FLOOR - 1e-9 and float(dt.max()) <= mamba2.DT_MAX + 1e-9
        assert float(mixer.A_log.min()) >= 0 and float(mixer.A_log.max()) <= math.log(16)
        assert torch.equal(mixer.D.detach(), torch.ones_like(mixer.D))
        assert float(mixer.in_proj.weight.abs().max()) <= d**-0.5
        assert float(mixer.out_proj.weight.abs().max()) <= mixer.geometry.inner**-0.5 / math.sqrt(2 * layers)
        assert float(layer.mlp.down.weight.abs().max()) <= ffn**-0.5 / math.sqrt(2 * layers)
        assert float(mixer.conv_weight.abs().max()) <= 4**-0.5
    assert abs(float(twin.embedding.weight.std()) - 0.02) < 0.003


# --- S12: H-Net pieces -----------------------------------------------------------------------------------------------


def test_ratio_loss_is_one_at_the_target_and_above_it_on_the_diagonal():
    for N in (2.0, 4.0, 5.0):  # L / N whole, so F can be 1/N exactly
        L = 600
        k = round(L / N)
        b = torch.zeros(1, L, dtype=torch.bool)
        b[0, :k] = True
        p = torch.full((1, L), k / L, dtype=F64)
        assert float(hnet.ratio_loss(b, p, N)) == pytest.approx(1.0, abs=1e-12)
        for frac in (1 / N - 0.1, 1 / N + 0.1, 0.95):  # F = G away from 1/N
            b2 = torch.zeros(1, L, dtype=torch.bool)
            b2[0, : int(frac * L)] = True
            assert float(hnet.ratio_loss(b2, torch.full((1, L), int(frac * L) / L, dtype=F64), N)) > 1.0
    third = torch.tensor(1 / 3.8, dtype=F64)  # N need not be whole
    assert float(hnet.ratio_from_stats(third, third, 3.8)) == pytest.approx(1.0, abs=1e-15)
    valid = torch.tensor([[True, True, False, False]])
    b = torch.tensor([[True, False, True, True]])
    F_, G = hnet.ratio_stats(b, torch.tensor([[1.0, 0.25, 0.9, 0.9]], dtype=F64), valid)
    assert float(F_) == 0.5 and float(G) == 0.625


def test_ste_is_one_forward_and_the_identity_backward():
    c = torch.tensor([0.3, 0.5, 0.999, 1e-7], dtype=torch.float32, requires_grad=True)
    out = hnet.ste(c)
    assert torch.equal(out, torch.ones_like(c))
    (out * torch.tensor([1.0, 2.0, 3.0, 4.0])).sum().backward()
    assert torch.equal(c.grad, torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_router_boundaries_forced_positions_and_padding():
    router = hnet.Router(4).to(F64).requires_grad_(False)
    x = torch.randn(2, 6, 4, generator=torch.Generator().manual_seed(0), dtype=F64)
    p = router(x)
    q, k = x[:, 1:], x[:, :-1]  # identity projections at the start
    cos = (q * k).sum(-1) / (q.norm(dim=-1) * k.norm(dim=-1))
    assert torch.allclose(p[:, 1:], (1 - cos) / 2, atol=1e-14) and torch.equal(p[:, 0], torch.ones(2, dtype=F64))
    forced = torch.zeros(2, 6, dtype=torch.bool)
    forced[0, 3] = forced[1, 5] = True
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[1, 4:] = False
    b, p2 = hnet.boundaries(p, forced, valid)
    assert bool(b[:, 0].all()) and float(p2[0, 3]) == 1.0 and bool(b[0, 3])
    assert not bool(b[1, 4:].any())  # padding is never a boundary, forced or not
    assert torch.equal(b[valid & ~forced], (p >= 0.5)[valid & ~forced])
    low = torch.full((1, 4), 0.2, dtype=F64)  # p_1 = 1 by definition, whatever the caller passes
    b, p3 = hnet.boundaries(low)
    assert b.tolist() == [[True, False, False, False]] and p3[0].tolist() == [1.0, 0.2, 0.2, 0.2]
    for bad in (torch.zeros(2, 6, dtype=torch.bool), torch.tensor([[True] * 6, [True, False, True, True, True, True]])):
        with pytest.raises(ValueError, match="non-empty prefix"):
            hnet.check_valid(bad, torch.Size([2, 6]))


def test_chunk_then_upsample_restores_each_positions_most_recent_chunk():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(3, 10, 5, generator=g, dtype=F64)
    b = torch.rand(3, 10, generator=g) < 0.4
    b[:, 0] = True
    p = torch.rand(3, 10, generator=g, dtype=F64)
    x_c, P_c, lengths = hnet.chunk(x, b, p, multiple=4)
    assert torch.equal(lengths, b.sum(-1)) and x_c.shape[1] % 4 == 0 and x_c.shape[1] >= int(lengths.max())
    for row in range(3):
        n = int(lengths[row])
        assert torch.equal(x_c[row, :n], x[row][b[row]]) and torch.equal(P_c[row, :n], p[row][b[row]])
        assert not bool(x_c[row, n:].any()) and bool((P_c[row, n:] == 1).all())
    up = hnet.upsample(x_c, b)
    for row in range(3):
        last = 0
        for t in range(10):
            last = t if b[row, t] else last
            assert torch.equal(up[row, t], x[row, last]), (row, t)


def test_dechunk_with_confident_boundaries_is_the_upsampled_main_output():
    """P = p = 1 at every boundary (clamped to 1 - 1e-4) and p = 0 elsewhere: c = 1, so the output is the repeat of
    each chunk's own vector up to the clamp."""
    g = torch.Generator().manual_seed(2)
    b = torch.tensor([[True, False, False, True, True, False]])
    p = b.to(F64)
    z_c = torch.randn(1, 3, 4, generator=g, dtype=F64)
    out = hnet.dechunk(z_c, torch.ones(1, 3, dtype=F64), b, p)
    assert float((out - hnet.upsample(z_c, b)).abs().max()) <= 2e-4 * float(z_c.abs().max())


def test_the_router_learns_through_the_confidence_and_the_ema():
    """d out_t / d p_t through STE(c_t) is +z_t at a boundary (c = p) and -z_t elsewhere (c = 1 - p), z_t the
    upsampled EMA (eqs. 6-9); P_c, the chunks' p, is the EMA's other path."""
    g = torch.Generator().manual_seed(3)
    b = torch.tensor([[True, False, True, False, False]])
    p = torch.tensor([[1.0, 0.3, 0.8, 0.1, 0.45]], dtype=F64, requires_grad=True)
    z_c = torch.randn(1, 2, 3, generator=g, dtype=F64)
    P_c = torch.tensor([[0.9, 0.7]], dtype=F64)  # given separately, so only the STE path reaches p
    weights = torch.randn(1, 5, 3, generator=g, dtype=F64)
    (hnet.dechunk(z_c, P_c, b, p) * weights).sum().backward()
    up = hnet.upsample(kernels.ema_scan(z_c, P_c), b)
    sign = torch.where(b, 1.0, -1.0).to(F64)
    assert torch.allclose(p.grad, sign * (up * weights).sum(-1), atol=1e-14)
    P_c.requires_grad_()
    hnet.dechunk(z_c, P_c, b, p.detach()).sum().backward()
    assert bool((P_c.grad != 0).all())


def test_width_adapters_round_trip_and_refuse_a_narrower_main():
    up, down = hnet.WidthUp(16, 24), hnet.WidthDown(16)
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        up.extra.copy_(torch.arange(8.0))
    wide = up(x)
    assert wide.shape == (2, 5, 24) and torch.equal(down(wide), x) and torch.equal(wide[0, 3, 16:], torch.arange(8.0))
    with pytest.raises(ValueError, match="monotone"):
        hnet.WidthUp(16, 8)


# --- every model kind: causality and batching ------------------------------------------------------------------------


def tiny_main(table: str, vocab_size):
    if table == "tlc-N":
        return A0Twin(twin_config("tiny"), vocab_size=vocab_size, init_seed=1)
    return BilingualTrunk("tiny", {"max_seq": 64}, vocab_size=vocab_size, init_seed=1)


def tiny_model(table: str, kind: str):
    if kind == "subword":
        return SubwordLM(tiny_main(table, 300)).to(F64)
    if kind == "bytes":
        return ByteLM(tiny_main(table, BYTE_VOCAB)).to(F64)
    return HNetLM(tiny_main(table, None), HNetConfig(**TINY_HNET), init_seed=2).to(F64)


def run(model, x, forced=None, valid=None):
    if isinstance(model, HNetLM):
        out = model(x, forced, valid)
        return out.logits, out.b
    return model(x), None


KINDS = [(table, kind) for table in ("tlc-N", "bilingual-B") for kind in ("subword", "bytes", "hnet")]


@pytest.mark.parametrize(
    "table,kind,with_forced", [(t, k, False) for t, k in KINDS] + [(t, "hnet", True) for t in ("tlc-N", "bilingual-B")]
)
def test_suffix_edits_leave_earlier_outputs_unchanged(table, kind, with_forced):
    """Changing the ids after position t leaves the logits (and the H-Net's boundaries) at positions <= t unchanged
    (spec §6.5), with forced boundaries on and off; exact for boundaries, 1e-12 for logits (an H-Net's main network
    then sees another chunk count, so its matmuls have other shapes)."""
    model = tiny_model(table, kind).eval()
    vocab = 300 if kind == "subword" else BYTE_VOCAB
    L = 40
    x = ids(2, L, vocab, seed=3, low=5)
    forced = None
    if with_forced:
        forced = torch.zeros(2, L, dtype=torch.bool)
        forced[:, ::7] = True
    with torch.no_grad():
        logits, b = run(model, x, forced)
        for t in (0, 9, 23, 38):
            changed = x.clone()
            changed[:, t + 1 :] = ids(2, L - t - 1, vocab, seed=10 + t, low=5)
            new_logits, new_b = run(model, changed, forced)
            assert float((new_logits[:, : t + 1] - logits[:, : t + 1]).abs().max()) <= 1e-12, t
            if b is not None:
                assert torch.equal(new_b[:, : t + 1], b[:, : t + 1]), t


@pytest.mark.parametrize("table,kind", KINDS)
def test_a_row_gives_the_same_outputs_alone_and_in_a_batch(table, kind):
    """Right-padded rows of other lengths (hence other chunk counts) in the batch change nothing in a row."""
    model = tiny_model(table, kind).eval()
    vocab = 300 if kind == "subword" else BYTE_VOCAB
    lengths = (40, 23, 31)
    x = ids(3, 40, vocab, seed=4, low=5)
    valid = torch.arange(40)[None, :] < torch.tensor(lengths)[:, None]
    with torch.no_grad():
        if isinstance(model, HNetLM):
            batch = model(x, valid=valid)
            assert len(set(batch.lengths.tolist())) > 1  # the rows really have other chunk counts
        else:
            batch_logits = model(x)
        for row, n in enumerate(lengths):
            if isinstance(model, HNetLM):
                alone = model(x[row : row + 1, :n])
                assert torch.equal(alone.b[0], batch.b[row, :n])
                assert int(alone.lengths[0]) == int(batch.lengths[row])
                got, expected = batch.logits[row, :n], alone.logits[0]
            else:
                got, expected = batch_logits[row, :n], model(x[row : row + 1, :n])[0]
            assert float((got - expected).abs().max()) <= 1e-12, row


@pytest.mark.parametrize("table,kind", KINDS)
def test_every_kind_runs_forward_and_backward_under_bf16_autocast(table, kind):
    """The training precision (fp32 weights, bf16 autocast): finite loss and gradients; the router, its statistics
    and the residual projection stay in fp32."""
    model = tiny_model(table, kind).float()
    x = ids(2, 30, 300 if kind == "subword" else BYTE_VOCAB, seed=8, low=5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = model(x)
        logits = out.logits if isinstance(model, HNetLM) else out
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].float().flatten(0, 1), x[:, 1:].flatten())
        if isinstance(model, HNetLM):
            assert out.p.dtype == out.ratio_loss.dtype == torch.float32
            loss = loss + model.cfg.alpha * out.ratio_loss
    assert logits.dtype == torch.bfloat16
    loss.backward()
    assert math.isfinite(loss.item())
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)


def test_hnet_out_statistics_ignore_padding_and_count_chunks():
    model = tiny_model("tlc-N", "hnet").eval()
    x = ids(2, 30, BYTE_VOCAB, seed=5, low=5)
    valid = torch.ones(2, 30, dtype=torch.bool)
    valid[1, 12:] = False
    with torch.no_grad():
        out = model(x, valid=valid)
    assert out.logits.shape == (2, 30, BYTE_VOCAB)
    assert torch.equal(out.lengths, out.b.sum(-1)) and not bool(out.b[1, 12:].any())
    expected_F = float(out.b[valid].double().mean())
    assert float(out.F) == pytest.approx(expected_F, abs=1e-15)
    assert float(out.G) == pytest.approx(float(out.p[valid].mean()), abs=1e-15)
    assert float(out.ratio_loss) == pytest.approx(float(hnet.ratio_loss(out.b, out.p, 4.0, valid)), abs=1e-15)


def test_bilingual_trunk_equals_the_production_routed_lm():
    """A subword arm's model on the bilingual table is RoutedLM itself: same initialisation (seed profile, one expert
    "seed", plan None) and the same logits, bit for bit, fp64."""
    from surogate_bilingual.config import profile
    from surogate_bilingual.model.lm import RoutedLM

    trunk = BilingualTrunk("tiny", vocab_size=64, init_seed=5).to(F64)
    production = RoutedLM(profile("tiny", "swiglu", vocab_size=64), ("seed",), init_seed=5).to(F64)
    x = ids(2, 16, 64, seed=6, low=5)
    with torch.no_grad():
        assert torch.equal(SubwordLM(trunk)(x), production(x))
    as_main = BilingualTrunk("tiny", vocab_size=None)
    assert not as_main.lm.backbone.emb.weight.requires_grad and as_main.lm.cfg.vocab_size == 6
    with pytest.raises(ValueError, match="may not set"):
        BilingualTrunk("tiny", {"vocab_size": 9}, vocab_size=64)


def test_parameter_stages_partition_the_trainable_parameters():
    model = tiny_model("bilingual-B", "hnet")
    stages = model.parameter_stages()
    ids_ = {name: {id(p) for p in params} for name, params in stages.items()}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert ids_["outer"] | ids_["main"] == trainable and not ids_["outer"] & ids_["main"]
    assert id(model.width_up.extra) in ids_["main"] and id(model.router.W_q.weight) in ids_["outer"]
    assert id(model.main.lm.backbone.emb.weight) not in trainable  # the trunk's unused, frozen embedding


# --- the arm builder -------------------------------------------------------------------------------------------------


def arm(table, kind, model):
    project, main = {"tlc-N": ("tlc", "twin_a0"), "bilingual-B": ("bilingual", "bilingual_trunk")}[table]
    return ArmSpec(f"{kind}-arm", table, kind, None, PROJECT_SPECIALS[project], model, main, (0,), ("screen",))


def test_build_model_builds_every_kind_of_both_tables():
    tiny = {"tlc-N": "tiny", "bilingual-B": "tiny"}
    for table, kind in KINDS:
        model_table = {"profile": tiny[table]}
        if table == "bilingual-B":
            model_table["overrides"] = {"max_seq": 64}
        if kind == "hnet":
            model_table["hnet"] = dict(TINY_HNET)
        model = build.build_model(
            arm(table, kind, model_table), vocab_size=300 if kind == "subword" else None, init_seed=0
        )
        expected = {"subword": SubwordLM, "bytes": ByteLM, "hnet": HNetLM}[kind]
        assert type(model) is expected, (table, kind)
        assert (model.cfg.vocab_size if kind == "hnet" else model.vocab_size) == (300 if kind == "subword" else 261)
        out = model(ids(1, 12, 261, seed=7, low=5))
        assert (out.logits if kind == "hnet" else out).shape[-1] >= 261
    # the outer stage and the main network never share a random stream
    assert build.derived_seed(0, "hnet-outer") != 0 and build.derived_seed(0, "a") != build.derived_seed(1, "a")


@pytest.mark.parametrize(
    "table,kind,model,vocab,message",
    [
        ("tlc-N", "hnet", {"profile": "tiny"}, None, "needs model.hnet"),
        ("tlc-N", "bytes", {"profile": "tiny", "hnet": TINY_HNET}, None, "has no model.hnet"),
        ("tlc-N", "subword", {"profile": "tiny"}, None, "needs the codec's vocab_size"),
        ("tlc-N", "subword", {"profile": "tiny", "seed": 1}, 300, "unknown model keys"),
        ("tlc-N", "hnet", {"profile": "tiny", "hnet": {**TINY_HNET, "depth": 2}}, None, "model.hnet"),
        ("tlc-N", "hnet", {"profile": "tiny", "hnet": {**TINY_HNET, "ratio": 1.0}}, None, "must exceed 1"),
        ("tlc-N", "bytes", {"profile": "no-such"}, None, "unknown twin profile"),
    ],
)
def test_build_model_refusals(table, kind, model, vocab, message):
    with pytest.raises(ValueError, match=message):
        build.build_model(arm(table, kind, model), vocab_size=vocab, init_seed=0)


# --- a tiny H-Net learns ---------------------------------------------------------------------------------------------

WORDS = ("def", "add", "x", "return", "total", "self", "value", "for", "in", "range")


def word_rows(generator, rows=4, length=65):
    out = []
    for _ in range(rows):
        text = ""
        while len(text) < length:
            text += WORDS[int(torch.randint(len(WORDS), (1,), generator=generator))] + " "
        out.append(list(text.encode()[:length]))
    return torch.tensor(out) + 5


def test_a_tiny_hnet_trains_on_the_cpu():
    """50 AdamW steps on SYNTHETIC word sequences (fp32, CPU, alpha 0.03, N = 4): the byte loss falls and F, the
    boundary fraction, ends nearer 1/N than it starts."""
    torch.manual_seed(0)
    model = HNetLM(A0Twin(twin_config("tiny"), vocab_size=None, init_seed=1), HNetConfig(**TINY_HNET), init_seed=2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    generator = torch.Generator().manual_seed(0)
    history = []
    for _ in range(50):
        x = word_rows(generator)
        out = model(x[:, :-1])
        ce = torch.nn.functional.cross_entropy(out.logits.reshape(-1, BYTE_VOCAB), x[:, 1:].reshape(-1))
        loss = ce + model.cfg.alpha * out.ratio_loss
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        history.append((ce.item(), out.F.item()))
    first_ce = sum(c for c, _ in history[:5]) / 5
    last_ce = sum(c for c, _ in history[-5:]) / 5
    assert last_ce < first_ce - 1.0, (first_ce, last_ce)
    target = 1 / model.cfg.ratio
    last_F = sum(f for _, f in history[-10:]) / 10
    assert abs(last_F - target) < abs(history[0][1] - target), (history[0][1], last_F)
    json.dumps(history)  # plain floats, as a run log would record them
