"""Non-streaming completion text of SharedModelServer (ledger topic T11).

The modes of SUROGATE_GRPO_DETOK must return what upstream's per-token loop
returns (`per_token`), token for token and text for text, for any completion:
multibyte characters split across byte tokens, invalid UTF-8, literal U+FFFD,
special tokens, stop strings that straddle tokens or occur before `min_tokens`,
model and request stop tokens, tool calls and Harmony's terminal token.
"""

import json
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
import torch

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from surogate.grpo.shared_model import SharedModelServer  # noqa: E402

SPECIAL = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|call|>",
    "<|end|>",
    "<|return|>",
]
SAMPLES = [
    "The quick brown fox jumps over the lazy dog. ",
    "def f(x):\n    return x ** 2  # square\n",
    "混合文本：训练、推理和评测。",
    "数学 1+1=2，然后",
    "Ünïcödé façade — naïve café. ",
    "emoji 🚀🔥👍🏽 and flags 🇨🇳🇺🇸 ",
    "日本語のテキストです。",
    "한국어 문장입니다. ",
    "<think>Let me think.</think>The answer is 42.",
    '{"name": "add", "arguments": {"a": 1, "b": 2}}',
    "Tabs\tand\nnewlines\n\n\nhere. ",
    "a . b ? c ! d , e ' f n't g 'm h 's",
    "\ufffd literal replacement ",
]
TOOLS = [
    dict(
        type="function",
        function=dict(
            name="add", parameters=dict(type="object", properties=dict(a=dict(type="integer"), b=dict(type="integer")))
        ),
    )
]
TEMPLATES = {"json_xml": None, "harmony": "{{ '<|start|>assistant' }}", "qwen_coder": "<function=x><tool_call>"}


def _corpus():
    rng = random.Random(0)
    return ["".join(rng.choice(SAMPLES) for _ in range(6)) for _ in range(300)]


def byte_level_bpe(**kwargs):
    split = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(split), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        _corpus(),
        trainers.BpeTrainer(
            vocab_size=700,
            special_tokens=SPECIAL,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", **kwargs)


def sentencepiece_bpe():
    """Llama-2 style: Metaspace, byte fallback, and a decoder that strips the first space."""
    tok = Tokenizer(models.BPE(byte_fallback=True, unk_token="<unk>", fuse_unk=True))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme="first", split=False)
    tok.decoder = decoders.Sequence(
        [decoders.Replace("\u2581", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)]
    )
    bytes_ = [f"<0x{b:02X}>" for b in range(256)]
    tok.train_from_iterator(
        _corpus(), trainers.BpeTrainer(vocab_size=900, special_tokens=["<unk>"] + bytes_ + SPECIAL, show_progress=False)
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", clean_up_tokenization_spaces=False)


def wordpiece():
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece()
    tok.train_from_iterator(
        _corpus(), trainers.WordPieceTrainer(vocab_size=600, special_tokens=["[UNK]"] + SPECIAL, show_progress=False)
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", unk_token="[UNK]", clean_up_tokenization_spaces=True
    )


@pytest.fixture(scope="module")
def tokenizers_():
    return dict(
        byte_level=byte_level_bpe(clean_up_tokenization_spaces=False),
        # transformers skips cleanup for BPE models unless forced.
        byte_level_bpe_cleanup=byte_level_bpe(clean_up_tokenization_spaces=True),
        byte_level_forced_cleanup=byte_level_bpe(
            clean_up_tokenization_spaces=True,
            clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True,
        ),
        sentencepiece=sentencepiece_bpe(),
        wordpiece=wordpiece(),
    )


def make_server(tokenizer, extra_eos=()):
    trainer = SimpleNamespace(
        batch_size=1,
        seq_length=4096,
        decode_batch_sample=None,
        get_shared_base_weights=lambda: dict(w=torch.zeros(8, dtype=torch.bfloat16)),
    )
    eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), *extra_eos]
    return SharedModelServer(
        trainer,
        tokenizer,
        dict(vocab_size=len(tokenizer)),
        dict(host="127.0.0.1", port=0, model="base", max_context=4096, max_concurrency=2, eos_token_id=eos),
    )


def generate(server, body, completion, mode, monkeypatch, callback=None):
    monkeypatch.setenv("SUROGATE_GRPO_DETOK", mode)
    request = server.prepare(body)
    rng = np.random.default_rng(len(completion))
    script = iter(completion)
    k = body.get("top_logprobs", 0)

    def decode(tokens, reset, sampling):
        token = next(script)
        return dict(
            status=0,
            token=token,
            logprob=float(-rng.random()),
            top_ids=rng.integers(0, len(server.tokenizer), k).tolist(),
            top_logprobs=(-rng.random(k)).tolist(),
        )

    result = server._generate(request, callback, decode)
    return {key: value for key, value in result.items() if key not in ("id", "created")}


def byte_ids(tokenizer, data: bytes):
    """One ByteLevel token per byte, so characters can be split and bytes can be invalid."""
    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("\u00a1"), ord("\u00ac") + 1),
        *range(ord("\u00ae"), ord("\u00ff") + 1),
    ]
    others = [b for b in range(256) if b not in printable]
    char = {b: chr(b) for b in printable} | {b: chr(256 + i) for i, b in enumerate(others)}  # GPT-2 bytes_to_unicode
    vocab = tokenizer.get_vocab()
    return [vocab[char[b]] for b in data]


def random_completion(rng, tokenizer, byte_level):
    ids = []
    specials = [tokenizer.convert_tokens_to_ids(s) for s in SPECIAL]
    for _ in range(rng.randint(1, 10)):
        kind = rng.random()
        if kind < 0.45:
            ids += tokenizer.encode(rng.choice(SAMPLES), add_special_tokens=False)
        elif kind < 0.6:
            ids.append(rng.choice(specials))
        elif kind < 0.65:
            ids += tokenizer.encode(
                '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>', add_special_tokens=False
            )
        elif byte_level and kind < 0.85:
            ids += byte_ids(tokenizer, rng.choice(SAMPLES).encode()[: rng.randint(1, 12)])
        elif byte_level:
            ids += byte_ids(
                tokenizer,
                bytes(
                    rng.choice([0x80, 0xBF, 0xC3, 0xE4, 0xF0, 0x9F, 0xFF, 0x41, 0x20]) for _ in range(rng.randint(1, 5))
                ),
            )
        else:
            ids.append(rng.randrange(len(tokenizer)))
    return ids or [tokenizer.convert_tokens_to_ids("<think>")]


def random_body(rng, tokenizer, completion):
    text = tokenizer.decode(completion, skip_special_tokens=False)
    stops = []
    for _ in range(rng.choice([0, 1, 1, 2, 3])):
        roll = rng.random()
        if roll < 0.7 and text:
            start = rng.randrange(len(text))
            stops.append(text[start : start + rng.randint(1, 6)])
        elif roll < 0.8:
            stops.append("\ufffd")
        else:
            stops.append("never-occurs-ZQX")
    # Half the prompts end inside a think block, as a thinking chat template's do.
    prompt = tokenizer.encode("Question: what is 6 * 7?", add_special_tokens=False)
    if rng.random() < 0.5:
        prompt.append(tokenizer.convert_tokens_to_ids("<think>"))
    body = dict(
        model="policy",
        tokens=prompt,
        temperature=1.0,
        logprobs=rng.random() < 0.8,
        max_tokens=rng.randint(1, len(completion)),
        min_tokens=0,
        stop=stops,
    )
    body["min_tokens"] = rng.randint(0, body["max_tokens"]) if rng.random() < 0.3 else 0
    if rng.random() < 0.2:
        body["top_logprobs"] = rng.randint(1, 3)
    if rng.random() < 0.2:
        body["ignore_eos"] = True
    if rng.random() < 0.2:
        body["stop_token_ids"] = [tokenizer.convert_tokens_to_ids(rng.choice(["</think>", "<|end|>", "</tool_call>"]))]
    if rng.random() < 0.4:
        body["tools"] = TOOLS
        body["parallel_tool_calls"] = rng.random() < 0.5
    template = TEMPLATES[rng.choice(list(TEMPLATES))]
    if template:
        body["chat_template_kwargs"] = dict(chat_template=template)
    return body


@pytest.mark.parametrize(
    "name", ["byte_level", "byte_level_bpe_cleanup", "byte_level_forced_cleanup", "sentencepiece", "wordpiece"]
)
def test_every_mode_returns_the_per_token_result(tokenizers_, name, monkeypatch):
    tokenizer = tokenizers_[name]
    byte_level = name.startswith("byte_level")
    server = make_server(tokenizer, extra_eos=[tokenizer.convert_tokens_to_ids("</tool_call>")])
    rng = random.Random(name)
    stopped_by_string = 0
    try:
        server.publish("policy", [], 0)
        for _ in range(160 if byte_level else 60):
            completion = random_completion(rng, tokenizer, byte_level)
            body = random_body(rng, tokenizer, completion)
            expected = generate(server, body, completion, "per_token", monkeypatch)
            for mode in ("final", "incremental"):
                assert generate(server, body, completion, mode, monkeypatch) == expected, (mode, body, completion)
            choice = expected["choices"][0]
            stop_tokens = (set() if body.get("ignore_eos") else server.eos) | set(body.get("stop_token_ids", []))
            stopped_by_string += choice["finish_reason"] != "length" and choice["token_ids"][-1] not in stop_tokens
    finally:
        server.close()
    assert stopped_by_string >= 5  # the cases exercise stop strings, not only EOS and length


def test_streaming_keeps_the_per_token_loop(tokenizers_, monkeypatch):
    tokenizer = tokenizers_["byte_level"]
    server = make_server(tokenizer)
    completion = tokenizer.encode("<think>plan</think>答案是 42。", add_special_tokens=False)
    body = dict(model="policy", tokens=[1, 2], max_tokens=len(completion), stop=["42"])
    try:
        server.publish("policy", [], 0)
        streams = {}
        for mode in ("per_token", "final", "incremental"):
            chunks = []
            result = generate(server, body, completion, mode, monkeypatch, callback=chunks.append)
            streams[mode] = ([{k: v for k, v in c.items() if k not in ("id", "created")} for c in chunks], result)
        assert streams["final"] == streams["incremental"] == streams["per_token"]
        assert len(streams["final"][0]) > 1
    finally:
        server.close()


class CountingTokenizer:
    """Delegates to a real tokenizer and counts the ids it is asked to decode."""

    def __init__(self, inner):
        self.inner, self.decoded, self.calls = inner, 0, 0
        self.eos_token_id, self.chat_template = inner.eos_token_id, None

    def decode(self, ids, **kwargs):
        self.calls += 1
        self.decoded += len(ids)
        return self.inner.decode(ids, **kwargs)

    def __len__(self):
        return len(self.inner)


class CountingProtocol:
    def __init__(self, inner):
        self.inner, self.name, self.calls = inner, inner.name, 0

    def parse(self, *args, **kwargs):
        self.calls += 1
        return self.inner.parse(*args, **kwargs)


@pytest.mark.parametrize("stop", [[], ["never-occurs-ZQX"]])
def test_a_non_streaming_request_decodes_and_parses_its_text_once(tokenizers_, monkeypatch, stop):
    """Fails on upstream's loop: it decodes k ids and parses on every step k (O(n^2) ids, n parses)."""
    inner = tokenizers_["byte_level"]
    tokenizer = CountingTokenizer(inner)
    trainer = SimpleNamespace(
        batch_size=1,
        seq_length=4096,
        decode_batch_sample=None,
        get_shared_base_weights=lambda: dict(w=torch.zeros(8, dtype=torch.bfloat16)),
    )
    server = SharedModelServer(
        trainer,
        tokenizer,
        dict(vocab_size=len(inner)),
        dict(host="127.0.0.1", port=0, model="base", max_context=4096, max_concurrency=2),
    )
    plain = "".join(sample for sample in SAMPLES if "think" not in sample)
    completion = inner.encode(plain * 2, add_special_tokens=False)[:300]
    n = len(completion)
    try:
        monkeypatch.delenv("SUROGATE_GRPO_DETOK", raising=False)  # the default
        if stop:
            monkeypatch.setenv("SUROGATE_GRPO_DETOK", "incremental")
        prompt = inner.encode("Question:", add_special_tokens=False)
        request = server.prepare(dict(model="base", tokens=prompt, max_tokens=n, logprobs=True, stop=stop))
        request["protocol"] = protocol = CountingProtocol(request["protocol"])
        tokenizer.decoded = tokenizer.calls = 0
        script = iter(completion)
        result = server._generate(
            request, None, lambda *args: dict(status=0, token=next(script), logprob=-1.0, top_ids=[], top_logprobs=[])
        )
    finally:
        server.close()
    assert result["choices"][0]["token_ids"] == completion
    assert result["choices"][0]["message"]["content"] == inner.decode(completion)
    assert protocol.calls == 1
    # One id per token for the logprob piece, then the whole completion once.
    # A CountingTokenizer is not a stock fast tokenizer, so a stop string
    # would be checked on full decodes; this request has none or none eligible
    # for the window, hence the separate stop-string case below.
    if not stop:
        assert tokenizer.calls == n + 1 and tokenizer.decoded == 2 * n


def test_the_byte_level_window_decodes_o1_ids_per_step(tokenizers_):
    from surogate._fork.grpo_detok import _ByteLevelTail, byte_level_backend

    inner = tokenizers_["byte_level"]
    backend = byte_level_backend(inner)
    assert backend is not None

    class Counting:
        decoded = 0

        def decode(self, ids, skip):
            Counting.decoded += len(ids)
            return backend.decode(ids, skip)

    completion = inner.encode("".join(SAMPLES) * 4, add_special_tokens=False)
    tail = _ByteLevelTail(Counting(), ("never-occurs-ZQX",))
    for k in range(1, len(completion) + 1):
        assert tail.advance(completion[:k], True) is False
    assert Counting.decoded <= 2 * len(completion)


@pytest.mark.parametrize("seed", range(4))
def test_the_byte_level_window_answers_as_a_full_decode(tokenizers_, seed):
    """Contract of _ByteLevelTail: same answer as `stop in decode(ids)` on every step up to the first hit."""
    from surogate._fork.grpo_detok import _ByteLevelTail, byte_level_backend

    tokenizer = tokenizers_["byte_level"]
    backend = byte_level_backend(tokenizer)
    rng = random.Random(seed)
    checked = hits = 0
    for _ in range(300):
        ids = random_completion(rng, tokenizer, True)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        stops = tuple(
            rng.choice([text[i : i + rng.randint(1, 5)] for i in range(len(text))] + ["\ufffd", "ZQX"])
            for _ in range(rng.randint(1, 3))
            if text
        )
        if not stops:
            continue
        minimum = rng.randint(0, len(ids))
        tail = _ByteLevelTail(backend, stops)
        for step in range(len(ids)):
            eligible = step + 1 >= minimum
            found = tail.advance(ids[: step + 1], eligible)
            if found is None:  # window limit: the caller switches to full decodes
                break
            full = tokenizer.decode(ids[: step + 1], skip_special_tokens=False)
            assert found == (eligible and any(s in full for s in stops)), (ids[: step + 1], stops, minimum)
            checked += 1
            if found:
                hits += 1
                break
    assert checked > 1000 and hits > 50


def test_long_invalid_byte_runs_fall_back_to_full_decodes(tokenizers_, monkeypatch):
    tokenizer = tokenizers_["byte_level"]
    completion = byte_ids(tokenizer, b"ok " + b"\xe4" * 40 + "中文 end".encode())
    server = make_server(tokenizer)
    body = dict(model="policy", tokens=[1], max_tokens=len(completion), stop=["文 e"])
    try:
        server.publish("policy", [], 0)
        expected = generate(server, body, completion, "per_token", monkeypatch)
        assert expected["choices"][0]["finish_reason"] == "stop"
        assert generate(server, body, completion, "incremental", monkeypatch) == expected
    finally:
        server.close()


def test_the_window_is_used_only_where_it_is_exact(tokenizers_):
    from surogate._fork.grpo_detok import byte_level_backend

    assert byte_level_backend(tokenizers_["byte_level"]) is not None
    assert byte_level_backend(tokenizers_["byte_level_bpe_cleanup"]) is not None
    for name in ("byte_level_forced_cleanup", "sentencepiece", "wordpiece"):
        assert byte_level_backend(tokenizers_[name]) is None, name
    assert byte_level_backend(SimpleNamespace(decode=lambda ids, **kw: "")) is None

    class Custom(PreTrainedTokenizerFast):
        def decode(self, *args, **kwargs):
            return super().decode(*args, **kwargs).upper()

    custom = Custom(tokenizer_object=tokenizers_["byte_level"].backend_tokenizer, eos_token="<|im_end|>")
    assert byte_level_backend(custom) is None
    patched = byte_level_bpe(clean_up_tokenization_spaces=False)
    patched.decode = lambda *args, **kwargs: ""
    assert byte_level_backend(patched) is None

    # Multimodal checkpoints load an AutoProcessor, whose decode forwards to its tokenizer.
    from transformers.processing_utils import ProcessorMixin

    class Processor(ProcessorMixin):
        attributes = ["tokenizer"]
        tokenizer_class = "AutoTokenizer"

        def __init__(self, tokenizer):
            super().__init__(tokenizer)

    class Rewriting(Processor):
        def decode(self, *args, **kwargs):
            return ""

    assert byte_level_backend(Processor(tokenizers_["byte_level"])) is not None
    assert byte_level_backend(Processor(tokenizers_["sentencepiece"])) is None
    assert byte_level_backend(Rewriting(tokenizers_["byte_level"])) is None


def test_the_window_check_imports_nothing_heavy_inside_a_request():
    """Importing transformers.processing_utils on the first stop-string request held the GIL for ~1.5 s."""
    script = """
import sys
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from surogate._fork.grpo_detok import byte_level_backend
tok = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}, merges=[]))
tok.pre_tokenizer, tok.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
assert "transformers.processing_utils" not in sys.modules
assert byte_level_backend(PreTrainedTokenizerFast(tokenizer_object=tok)) is not None
assert "transformers.processing_utils" not in sys.modules, "imported by the check"
"""
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run([sys.executable, "-c", script], cwd=root, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]


def test_an_unknown_mode_is_rejected(monkeypatch):
    from surogate._fork.grpo_detok import completion_view

    monkeypatch.setenv("SUROGATE_GRPO_DETOK", "fast")
    with pytest.raises(ValueError, match="SUROGATE_GRPO_DETOK"):
        completion_view(None, dict(stop=[], minimum=0, maximum=1), None)
    monkeypatch.setenv("SUROGATE_GRPO_DETOK", "per_token")
    assert completion_view(None, dict(stop=[], minimum=0, maximum=1), None) is None


def test_http_rollouts_match_per_token_text(tokenizers_, monkeypatch):
    """End to end through do_POST and generate(), on the CPU sampler path."""
    tokenizer = tokenizers_["byte_level"]
    completion = tokenizer.encode("<think>先想一想。</think>Answer: 🚀 done\n\nSTOP here", add_special_tokens=False)
    position = dict(prompt=0)

    def next_token_logits(inputs, positions):
        step = int(positions[0]) - position["prompt"] + 1
        logits = np.zeros((1, len(tokenizer)), dtype=np.float32)
        logits[0, completion[step]] = 30.0
        return logits

    trainer = SimpleNamespace(
        batch_size=1,
        seq_length=4096,
        next_token_logits=next_token_logits,
        get_shared_base_weights=lambda: dict(w=torch.zeros(8, dtype=torch.bfloat16)),
    )
    server = SharedModelServer(
        trainer,
        tokenizer,
        dict(vocab_size=len(tokenizer)),
        dict(host="127.0.0.1", port=0, model="base", max_context=4096, max_concurrency=2),
    )
    url = f"http://127.0.0.1:{server.http.server_port}/v1/chat/completions/tokens"
    try:
        server.publish("policy", [], 0)
        results = {}
        for mode in ("per_token", "final", "incremental"):
            monkeypatch.setenv("SUROGATE_GRPO_DETOK", mode)
            for stop in ([], ["STOP"], ["\n\n"]):
                position["prompt"] = 2
                body = dict(
                    model="policy", tokens=[5, 6], temperature=0, max_tokens=len(completion), logprobs=True, stop=stop
                )
                response = requests.post(url, json=body, timeout=30)
                assert response.ok, response.text
                payload = response.json()
                results[mode, tuple(stop)] = json.dumps(
                    {k: v for k, v in payload.items() if k not in ("id", "created")}
                )
        for stop in ([], ["STOP"], ["\n\n"]):
            assert (
                results["final", tuple(stop)]
                == results["incremental", tuple(stop)]
                == results["per_token", tuple(stop)]
            )
        assert json.loads(results["final", ("STOP",)])["choices"][0]["finish_reason"] == "stop"
    finally:
        server.close()
