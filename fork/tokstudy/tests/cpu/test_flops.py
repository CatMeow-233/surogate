"""Analytic FLOPs and H-Net matching (implementation step S13): the A0 forward equals surogate_tlc.ir.flops_per_token
term for term; the bilingual seed trunk equals 6 x surogate_bilingual.audit.flops.macs_per_token in the matmul
convention and 652,889,088 FLOPs per token at T = 1024 in the parameter convention; a flat-byte A0 costs
153,545,760 FLOPs per byte with scan=False and exactly the SSD and conv terms more with scan=True; the parameter
counts of the specs equal the models'; match_hnet_config lands within 3 % or refuses.
"""

import pytest
from surogate_tokstudy import flops
from surogate_tokstudy.models.build import flop_spec
from surogate_tokstudy.models.hnet import HNetConfig, HNetLM
from surogate_tokstudy.models.lm import ByteLM, SubwordLM
from surogate_tokstudy.models.trunk_bilingual import BilingualTrunk
from surogate_tokstudy.models.twin_a0 import A0Twin, twin_config


def a0(vocab):
    return A0Twin(twin_config("a0"), vocab_size=vocab, init_seed=None)


def test_a0_forward_equals_the_tlc_ir_formula(tlc_default_config):
    from surogate_tlc import ir

    cfg = tlc_default_config
    spec = flop_spec(SubwordLM(a0(cfg["vocab_size"])))
    per_token = flops.stack_fwd(spec["main"], 512) + flops.head_fwd(384, cfg["vocab_size"])
    assert per_token == ir.flops_per_token(cfg, 512)
    m = spec["main"]
    mixer = flops.mamba2_mixer_fwd(m["d"], m["heads"], m["head_dim"], m["state"], m["groups"], m["chunk"], m["kernel"])
    assert 12 * (mixer + flops.swiglu_fwd(384, 1024)) + flops.head_fwd(384, cfg["vocab_size"]) == per_token
    assert flops.row_train_flops(spec, 7) == 3 * 7 * per_token
    # F_target of tlc-N: A0 at 0.8B tokens [verified arithmetic]
    assert flops.row_train_flops(spec, 1) * 800_000_000 == 164_207_001_600_000_000


def test_bilingual_seed_trunk_flops_in_both_conventions():
    from surogate_bilingual.audit.flops import macs_per_token

    trunk = BilingualTrunk("pilot", vocab_size=32768, init_seed=0)
    spec = flop_spec(SubwordLM(trunk))
    assert flops.row_train_flops(spec, 1024, convention="params") == 652_889_088 * 1024
    assert flops.row_train_flops(spec, 1024) == 6 * macs_per_token(trunk.cfg, "single", T=1024) * 1024
    assert flops.row_train_flops(spec, 1024) == 652_738_560 * 1024  # norms are no matmuls
    assert flops.stack_params(spec["main"]) == trunk.non_embedding_parameters() == 79_454_720


def test_flat_byte_a0_flops_with_and_without_the_scan():
    spec = flop_spec(ByteLM(a0(261)))
    no_scan = flops.row_train_flops(spec, 1, convention="params", scan=False)
    assert no_scan == 153_545_760
    with_scan = flops.row_train_flops(spec, 1, convention="params", scan=True)
    assert with_scan - no_scan == 3 * 12 * flops.mamba2_scan_fwd(spec["main"])
    assert flops.stack_params(spec["main"]) == a0(None).non_embedding_parameters() == 25_490_736
    # the matmul convention without the scan leaves out the SSD and conv terms too
    assert flops.row_train_flops(spec, 1, scan=False) < flops.row_train_flops(spec, 1)


def test_hnet_row_flops_split_into_outer_and_main_on_the_realized_chunks():
    cfg = HNetConfig(d0=16, encoder_layers=2, decoder_layers=1, ratio=4.0, head_dim=8, state=4, ssd_chunk=8)
    model = HNetLM(A0Twin(twin_config("tiny"), vocab_size=None, init_seed=1), cfg, init_seed=2)
    spec = flop_spec(model)
    assert spec["encoder"]["layers"] == 2 and spec["decoder"]["layers"] == 1 and spec["encoder"]["ffn"] is None
    outer = flops.outer_fwd_per_byte(spec)
    main = flops.stack_fwd(spec["main"], 0)
    assert flops.row_train_flops(spec, 40, 9) == 3 * (40 * outer + 9 * main)
    assert outer == sum(flops.stack_fwd(spec[k], 0) for k in ("encoder", "decoder")) + 6 * 16 * 16 + 2 * 16 * 261
    with pytest.raises(ValueError, match="realized chunk count"):
        flops.row_train_flops(spec, 40)
    # parameter counts of the outer stacks equal the modules'
    assert flops.stack_params(spec["encoder"]) == sum(p.numel() for p in model.encoder.parameters())
    assert flops.stack_params(spec["decoder"]) == sum(p.numel() for p in model.decoder.parameters())
    # on a trunk, the main network's attention runs over the row's chunks
    trunk = HNetLM(BilingualTrunk("tiny", {"max_seq": 64}, vocab_size=None, init_seed=1), cfg, init_seed=2)
    tspec = flop_spec(trunk)
    q = tspec["main"]["q_width"]
    assert (flops.row_train_flops(tspec, 40, 10) - flops.row_train_flops(tspec, 40, 9)
            == 3 * (flops.stack_fwd(tspec["main"], 10) * 10 - flops.stack_fwd(tspec["main"], 9) * 9))  # fmt: skip
    assert flops.stack_fwd(tspec["main"], 10) - flops.stack_fwd(tspec["main"], 9) == tspec["main"]["layers"] * 2 * q


def test_match_hnet_config_lands_within_three_percent_or_refuses():
    spec = flop_spec(SubwordLM(a0(16384)))
    c = 3.8
    target = flops.flops_per_byte(spec, row_bytes=2048, bytes_per_position=c)
    assert target == pytest.approx(3 * (flops.stack_fwd(spec["main"], 539) + flops.head_fwd(384, 16384)) / c)
    found = flops.match_hnet_config(spec["main"], target, c, row_bytes=2048)
    assert abs(found["error"]) <= 0.03 and found["candidates"][0]["d0"] == found["d0"]
    stack = flops.outer_stack(found["d0"], found["encoder_layers"])
    hspec = {"kind": "hnet", "main": spec["main"], "vocab": 261, "d0": found["d0"], "encoder": stack, "decoder": stack}
    assert flops.flops_per_byte(hspec, row_bytes=2048, ratio=c) == pytest.approx(found["flops_per_byte"])
    assert all(abs(f["error"]) <= 0.03 for f in found["candidates"])
    # STUDY §10's estimate (2 + 2 layers at d0 = 256) is not within 3 % at c = 3.8 in this convention
    assert (2, 256) not in {(f["encoder_layers"], f["d0"]) for f in found["candidates"]}
    with pytest.raises(ValueError, match="no grid point within"):
        flops.match_hnet_config(spec["main"], target / 10, c, [(4, 384)], row_bytes=2048)
    with pytest.raises(ValueError, match="width"):
        flops.match_hnet_config(spec["main"], target, c, [(2, 512)], row_bytes=2048)


def test_flops_command_line(tmp_path, capsys):
    registry = tmp_path / "arms.toml"
    registry.write_text(
        'format = "tokstudy-arms/1"\ntable = "tlc-N"\n'
        '[[arm]]\nid = "N2"\nkind = "bytes"\nmodel = {profile = "tiny"}\nseeds = [0]\nstages = ["decision"]\n'
        '[[arm]]\nid = "N1"\nkind = "hnet"\nseeds = [0]\nstages = ["decision"]\n'
        '[arm.model]\nprofile = "tiny"\nhnet = {d0 = 16, encoder_layers = 1, decoder_layers = 1, ratio = 4.0, '
        "head_dim = 8, state = 4, ssd_chunk = 8}\n",
        encoding="utf-8",
    )
    assert flops.main([str(registry), "--k-train", "64"]) == 0
    out = capsys.readouterr().out
    assert '"N1"' in out and '"flops_per_byte_at_target_ratio"' in out
    assert flops.main([str(registry), "--k-train", "64", "--match", "N1"]) == 1
    assert "--reference" in capsys.readouterr().err
