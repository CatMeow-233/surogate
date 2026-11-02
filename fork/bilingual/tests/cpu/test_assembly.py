"""Packs, assembly manifests and forced-route fidelity (docs/spec.md, I2-A10; IMPL-04, MA-14).

The fixture writes a tiny family-I seed pack and expert packs forked from it (each expert nudged away from the
seed, as training would), plus packs that must be refused: one from another seed, one LP-k expert, one of family II.
"""

import json
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from surogate_bilingual.__main__ import main as cli
from surogate_bilingual.assembly import assemble as asm
from surogate_bilingual.assembly.hashing import content_hash
from surogate_bilingual.assembly.packs import (
    BackboneMismatchError,
    ConfigMismatchError,
    ExpertPack,
    FormatError,
    HashMismatchError,
    MissingKeyError,
    SeedPack,
    UnexpectedKeyError,
    check_extends,
    probe_positions,
    read_pack,
    write_expert_pack,
    write_seed_pack,
)
from surogate_bilingual.canonical import sha256_file
from surogate_bilingual.config import profile
from surogate_bilingual.model.lm import RoutedLM


def seed_pack(root, family="swiglu", init_seed=1):
    tokenizer = root / f"tokenizer-{family}-{init_seed}.json"
    tokenizer.write_text('{"stand-in": "the real tokenizer lands in step I4"}', "utf-8")
    model = RoutedLM(profile("tiny", family), ["seed"], init_seed=init_seed)
    return write_seed_pack(root / f"seed-{family}-{init_seed}", model, tokenizer, {"run_name": "seed-test"})


def expert_pack(root, seed, name, nudge, cfg=None, out=None):
    model = asm.model_from_seed(seed, [name], cfg)
    g = torch.Generator().manual_seed(nudge)
    with torch.no_grad():
        for p in model.bank.experts[name].parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
    return write_expert_pack(out or root / f"{name}-{nudge}", model, name, seed, {"run_name": f"{name}-test"})


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    root = tmp_path_factory.mktemp("packs")
    seed = seed_pack(root)
    lp = profile("tiny", "swiglu", expert_layers=(0,))
    other_seed = seed_pack(root, init_seed=2)
    slot_seed = seed_pack(root, family="slot")
    return {
        "root": root,
        "seed": seed,
        "zh": expert_pack(root, seed, "zh", 11),
        "en": expert_pack(root, seed, "en", 12),
        "t3": expert_pack(root, seed, "t3", 13),
        "en_lp": expert_pack(root, seed, "en", 14, cfg=lp, out=root / "en-lp"),
        "zh_lp": expert_pack(root, seed, "zh", 15, cfg=lp, out=root / "zh-lp"),
        "en_other_seed": expert_pack(root, other_seed, "en", 16, out=root / "en-other-seed"),
        "en_slot": expert_pack(root, slot_seed, "en", 17, out=root / "en-slot"),
    }


def copy_pack(src, dst, edit_meta=None, edit_weights=None):
    """A copy of a pack, optionally with the sidecar or the weights changed and every hash re-recorded."""
    shutil.copytree(src, dst)
    meta = json.loads((dst / "pack.json").read_text("utf-8"))
    if edit_weights is not None:
        weights = edit_weights(load_file(str(dst / "weights.safetensors")))
        save_file(weights, str(dst / "weights.safetensors"))
        meta["hashes"]["weights_file"] = sha256_file(dst / "weights.safetensors")
        meta["hashes"]["expert"] = content_hash(weights)
    if edit_meta is not None:
        edit_meta(meta)
    (dst / "pack.json").write_text(json.dumps(meta), "utf-8")
    return dst


# --- packs -------------------------------------------------------------------------------------------------------


def test_i2_a10_packs_round_trip_every_hash(packs):
    seed = read_pack(packs["seed"])
    assert isinstance(seed, SeedPack) and seed.meta["format"] == "bilingual-seed/1"
    assert content_hash(seed.backbone_state()) == seed.meta["hashes"]["backbone"]
    assert content_hash(seed.seed_ffn_state()) == seed.meta["hashes"]["seed_ffn"]
    assert sha256_file(packs["seed"] / "tokenizer.json") == seed.meta["tokenizer_sha256"]
    zh = read_pack(packs["zh"])
    assert isinstance(zh, ExpertPack) and zh.name == "zh" and zh.meta["format"] == "bilingual-expert/1"
    assert zh.seed_fingerprint == seed.seed_fingerprint and zh.meta["depends_on"] == []
    assert content_hash(zh.weights) == zh.expert_hash
    assert zh.meta["probe"]["positions"] == probe_positions(16) == [0, 1, 8, 15]
    assert zh.probe.shape == (2, 4, 64) and zh.probe.dtype == torch.float32
    assert not list(packs["root"].glob("*.tmp-*"))
    with pytest.raises(FileExistsError):
        expert_pack(packs["root"], packs["seed"], "zh", 11)


def test_expert_pack_records_its_dependencies(packs, tmp_path):
    path = write_expert_pack(
        tmp_path / "bridge", asm.model_from_seed(packs["seed"], ["bridge"]), "bridge", packs["seed"],
        {"recipe": "bridge", "bridge_variant": "c"}, depends_on=[packs["zh"], packs["en"]],
    )  # fmt: skip
    deps = read_pack(path).meta["depends_on"]
    assert [(d["name"], d["expert_hash"]) for d in deps] == [
        ("zh", read_pack(packs["zh"]).expert_hash),
        ("en", read_pack(packs["en"]).expert_hash),
    ]


def test_an_expert_pack_refuses_a_changed_backbone(packs, tmp_path):
    model = asm.model_from_seed(packs["seed"], ["zh"])
    with torch.no_grad():
        model.backbone.layers[0].attn.q.weight[0, 0] += 1e-3
    with pytest.raises(BackboneMismatchError):
        write_expert_pack(tmp_path / "zh", model, "zh", packs["seed"], {})
    seed = read_pack(packs["seed"])
    with pytest.raises(ConfigMismatchError):
        check_extends(seed, profile("tiny", "swiglu", rope_theta=5000.0))
    # only a family-I expert on every layer may be wider than the seed FFN (B12a, step I5); LP-k keeps the seed FFN
    # inside the backbone, and family II's readout width k is shared, so neither may change ffn_width
    check_extends(seed, profile("tiny", "swiglu", ffn_width=96))
    with pytest.raises(ConfigMismatchError):
        check_extends(seed, profile("tiny", "swiglu", ffn_width=96, expert_layers=(0,)))
    with pytest.raises(ConfigMismatchError):
        check_extends(read_pack(packs["root"] / "seed-slot-1"), profile("tiny", "slot", ffn_width=96))


# --- assembly ----------------------------------------------------------------------------------------------------


def test_i2_a10_assembly_loads_and_reproduces_every_probe(packs, tmp_path):
    out = asm.assemble(
        packs["seed"], {"zh": packs["zh"], "en": packs["en"], "t3": packs["t3"]}, None, tmp_path / "a.json"
    )
    doc = json.loads(out.read_text("utf-8"))
    assert doc["format"] == "bilingual-assembly/1" and doc["fusion"] == "output"
    assert [s["name"] for s in doc["slots"]] == ["zh", "en", "t3"]
    assert [s["expert_hash"] for s in doc["slots"]] == [read_pack(packs[n]).expert_hash for n in ("zh", "en", "t3")]
    assert not any(s["path"].startswith("/") for s in doc["slots"])  # relative to the manifest
    model = asm.load_assembly(out)
    assert model.expert_names == ("zh", "en", "t3")
    report = asm.verify_fidelity(out)
    assert report.passed and [s.name for s in report.slots] == ["zh", "en", "t3"]
    for slot in report.slots:
        assert slot.max_abs <= 1e-5
        assert slot.max_abs == 0.0  # expected on CPU fp32 (MS §9.4)
        assert slot.argmax_agreement == 1.0


def test_i2_a10_lp_assembly_keeps_the_seed_ffn_on_shared_layers(packs, tmp_path):
    out = asm.assemble(packs["seed"], {"zh": packs["zh_lp"], "en": packs["en_lp"]}, None, tmp_path / "lp.json")
    model = asm.load_assembly(out)
    seed = read_pack(packs["seed"])
    assert torch.equal(model.backbone.layers[1].ffn.gate.weight, seed.weights["ffn.1.gate.weight"])
    assert model.backbone.layers[0].ffn is None
    assert asm.verify_fidelity(out).passed


def test_fidelity_fails_when_a_probe_does_not_match(packs, tmp_path):
    def nudge_probe(meta):
        probe = load_file(str(tmp_path / "zh" / "probe.safetensors"))["logits"]
        save_file({"logits": probe + 1e-3}, str(tmp_path / "zh" / "probe.safetensors"))
        meta["hashes"]["probe_file"] = sha256_file(tmp_path / "zh" / "probe.safetensors")

    copy_pack(packs["zh"], tmp_path / "zh", edit_meta=nudge_probe)
    out = asm.assemble(packs["seed"], {"zh": tmp_path / "zh", "en": packs["en"]}, None, tmp_path / "a.json")
    report = asm.verify_fidelity(out)
    assert not report.passed and [s.passed for s in report.slots] == [False, True]


def failing_assemblies(packs, tmp):
    renamed = copy_pack(packs["zh"], tmp / "zh-as-en", edit_meta=lambda m: m.update(name="en"))
    missing = copy_pack(
        packs["zh"], tmp / "missing", edit_weights=lambda w: {k: v for k, v in w.items() if k != "ffn.0.up.weight"}
    )
    extra = copy_pack(
        packs["zh"], tmp / "extra", edit_weights=lambda w: {**w, "ffn.9.up.weight": w["ffn.0.up.weight"].clone()}
    )
    tampered = copy_pack(packs["zh"], tmp / "tampered")
    data = bytearray((tampered / "weights.safetensors").read_bytes())
    data[-1] ^= 0x01
    (tampered / "weights.safetensors").write_bytes(bytes(data))
    future = copy_pack(packs["zh"], tmp / "future", edit_meta=lambda m: m.update(format="bilingual-expert/2"))
    seed, zh, en = packs["seed"], packs["zh"], packs["en"]
    return {
        asm.SlotNameMismatchError: (seed, {"zh": en, "en": zh}, None),
        asm.PackReusedError: (seed, {"zh": zh, "en": zh}, None),
        "relabelled copy": (seed, {"zh": zh, "en": renamed}, None),
        asm.SeedFingerprintMismatchError: (seed, {"zh": zh, "en": packs["en_other_seed"]}, None),
        asm.ConfigHashMismatchError: (seed, {"zh": zh, "en": packs["en_lp"]}, None),
        asm.FamilyMismatchError: (seed, {"zh": zh, "en": packs["en_slot"]}, None),
        MissingKeyError: (seed, {"zh": missing, "en": en}, None),
        UnexpectedKeyError: (seed, {"zh": extra, "en": en}, None),
        asm.DuplicateSlotError: (seed, [("zh", zh), ("zh", en)], None),
        HashMismatchError: (seed, {"zh": tampered, "en": en}, None),
        asm.InvalidSlotNameError: (seed, {"zh": zh, "EN": en}, None),
        asm.FusionError: (seed, {"zh": zh, "en": en}, "pre"),
        FormatError: (zh, {"zh": zh, "en": en}, None),
        "unknown major": (seed, {"zh": future, "en": en}, None),
    }


EXPECTED = {"relabelled copy": asm.PackReusedError, "unknown major": FormatError}


def test_i2_a10_each_assembly_failure_raises_its_own_error(packs, tmp_path):
    cases = failing_assemblies(packs, tmp_path)
    raised = {}
    for case, (seed, slots, fusion) in cases.items():
        with pytest.raises(asm.AssemblyError) as info:
            asm.assemble(seed, slots, fusion, tmp_path / "never.json")
        expected = EXPECTED.get(case, case)
        assert type(info.value) is expected, (case, type(info.value), str(info.value))
        raised[case] = type(info.value)
    distinct = [t for c, t in raised.items() if c not in EXPECTED]
    assert len(set(distinct)) == len(distinct) == 12
    assert not (tmp_path / "never.json").exists()


def test_load_assembly_reverifies_the_packs(packs, tmp_path):
    zh = copy_pack(packs["zh"], tmp_path / "zh")
    out = asm.assemble(packs["seed"], {"zh": zh, "en": packs["en"]}, None, tmp_path / "a.json")
    weights = load_file(str(zh / "weights.safetensors"))
    weights["ffn.0.up.weight"] = weights["ffn.0.up.weight"] + 1
    save_file(weights, str(zh / "weights.safetensors"))
    with pytest.raises(HashMismatchError):
        asm.load_assembly(out)


# --- command line ------------------------------------------------------------------------------------------------


def test_assemble_and_verify_commands(packs, tmp_path, capsys):
    args = ["--seed", str(packs["seed"]), "--slot", f"zh={packs['zh']}", "--slot", f"en={packs['en']}"]
    assert cli(["assemble", *args, "--out", str(tmp_path / "dry.json"), "--dry-run"]) == 0
    assert '"bilingual-assembly/1"' in capsys.readouterr().out and not (tmp_path / "dry.json").exists()
    assert cli(["assemble", *args, "--out", str(tmp_path / "a.json")]) == 0
    assert cli(["verify", str(tmp_path / "a.json"), "--dry-run"]) == 0
    assert cli(["verify", str(tmp_path / "a.json"), "--out", str(tmp_path / "fidelity.json")]) == 0
    report = json.loads((tmp_path / "fidelity.json").read_text("utf-8"))
    assert report["format"] == "bilingual-fidelity/1" and report["passed"] is True
    capsys.readouterr()
    swapped = ["--seed", str(packs["seed"]), "--slot", f"zh={packs['en']}", "--slot", f"en={packs['zh']}"]
    assert cli(["assemble", *swapped, "--out", str(tmp_path / "b.json")]) == 1
    assert "SlotNameMismatchError" in capsys.readouterr().err and not (tmp_path / "b.json").exists()
