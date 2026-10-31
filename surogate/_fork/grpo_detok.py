"""Non-streaming completion text of the training-path rollout server (ledger topic T11).

`SharedModelServer._generate` builds the response text inside its token loop:
on every token it decodes the whole completion so far, scans it for stop
strings, and runs `protocol.parse` over it. That is O(n) work per token and
O(n^2) per request, all of it under the GIL that the handler threads, the
decode scheduler and the in-process orchestrator share. A non-streaming request
(no callback, which is every GRPO rollout: the orchestrator does not stream)
uses only what the last iteration of the loop computes: `message` is rebuilt on
every step and only the final value is returned, `delta` and `emitted_fields`
only feed the callback, and on a step that does not end the completion
`protocol.parse` runs with `final=False`, which only slices strings and cannot
raise for the validated tools of `request_tools`.

So the hook in `_generate` asks `CompletionEnd.ends` whether a step ends the
completion, and skips the text block on the steps that do not. The step that
does end it runs upstream's own block unchanged: one full decode, the stop
truncation, one `protocol.parse(..., final=True)`, the message. `ends` answers
exactly as upstream's `final` would: the step's token is a stop token, or it is
the last allowed step, or (from step `minimum - 1` on, as upstream checks) a
stop string occurs in `tokenizer.decode(ids)`. Only that last condition needs
text, and only when the request has stop strings.

Modes, from `SUROGATE_GRPO_DETOK` (read per request):

- `final` (default): stop strings are checked on a full decode, as upstream,
  on each step from `minimum - 1` on; without stop strings a step costs nothing
  here. For a given token stream, the same text, message, logprobs and finish
  reason as upstream for every request upstream completes (see below for the
  token stream itself). One difference in failure cases only:
  a tokenizer whose `decode` raises on some prefix of a completion but not on
  the whole of it fails upstream at that step and here not at all (HF fast
  tokenizers never raise there: unknown ids are dropped, invalid UTF-8 is
  replaced).
- `incremental`: as `final`, but for a stock transformers fast tokenizer with a
  ByteLevel decoder (Qwen, Llama 3, GLM, DeepSeek, GPT-OSS) the stop check
  decodes only the tokens since the last complete character and scans a
  window as long as the longest stop string. See `_ByteLevelTail` for why the
  answer equals upstream's. Other tokenizers use the `final` check.
- `per_token`: upstream's loop, unchanged.

Streaming requests (a callback) always take upstream's loop. No state outlives
a request; nothing here is saved in checkpoints.

The equivalence is per request, for the tokens the sampler returns; it does not
mean the GPU receives upstream's calls. Without a decode scheduler requests run
one at a time under `compute_lock` and it does. With one (continuous batching,
whenever the trainer has `decode_batch_logits`), each step joins a round with
whatever other sessions have submitted, and skipping the text work shortens the
host time between a step's result and the next submission, as any host speedup
does. Steps then reach the scheduler at different times, so the number, size
and membership of rounds differ, and batched decode need not be bitwise equal
across batches: above concurrency 1 the sampled tokens match upstream only up
to upstream's own timing nondeterminism, which already decides how rounds form.
"""

from __future__ import annotations

import os
import sys
import weakref

MODE_ENV = "SUROGATE_GRPO_DETOK"
MODES = ("final", "incremental", "per_token")
# Tokens a ByteLevel window may hold while its text still ends with U+FFFD
# (an incomplete character needs at most 4 byte tokens); past this the request
# falls back to the full-decode check, which keeps garbage byte runs O(n).
WINDOW_LIMIT = 16


def detok_mode() -> str:
    mode = os.environ.get(MODE_ENV, "final").strip() or "final"
    if mode not in MODES:
        raise ValueError(f"{MODE_ENV} must be one of {', '.join(MODES)}, not {mode!r}")
    return mode


def completion_view(tokenizer, request: dict, callback) -> CompletionEnd | None:
    """The view `_generate` consults on each step, or None for upstream's per-token text."""
    mode = detok_mode()
    if callback is not None or mode == "per_token":
        return None
    stops = tuple(request["stop"])
    tail = None
    if stops and mode == "incremental":
        backend = byte_level_backend(tokenizer)
        if backend is not None:
            tail = _ByteLevelTail(backend, stops)
    return CompletionEnd(tokenizer, request["minimum"], request["maximum"], stops, tail)


class CompletionEnd:
    """Whether upstream's iteration for a step would be its final one."""

    __slots__ = ("tokenizer", "minimum", "maximum", "stops", "tail")

    def __init__(self, tokenizer, minimum: int, maximum: int, stops: tuple[str, ...], tail=None):
        self.tokenizer, self.minimum, self.maximum, self.stops, self.tail = tokenizer, minimum, maximum, stops, tail

    def ends(self, ids: list[int], step: int, stopped: bool) -> bool:
        # `stopped` is upstream's `token in eos`; `ids` already holds this step's token.
        if stopped or step + 1 == self.maximum:
            return True
        if not self.stops:
            return False
        eligible = step + 1 >= self.minimum
        if self.tail is not None:
            found = self.tail.advance(ids, eligible)
            if found is not None:
                return found
            self.tail = None
        if not eligible:
            return False
        # Upstream's text on a step that is not stopped is decode(ids).
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        return any(stop in text for stop in self.stops)


class _ByteLevelTail:
    """Stop-string check over decode(ids) that decodes only the newest tokens.

    Why this equals a scan of `tokenizer.decode(ids)` (tokenizers 0.23.3 and
    transformers 5.21, checked by `byte_level_backend`):

    1. `TokenizerImpl::decode` maps each id to its token string (unknown ids
       are dropped) and calls the decoder; transformers' `TokenizersBackend._decode`
       returns that string unchanged when it applies no cleanup.
    2. The ByteLevel decoder turns every token into bytes on its own (each
       character through the byte table, or the token's UTF-8 if a character
       is not in it) and returns `String::from_utf8_lossy` of the concatenation.
    3. If `from_utf8_lossy(X)` does not end with U+FFFD, X ends on a complete
       character (an incomplete trailing sequence always yields a final U+FFFD),
       so the UTF-8 decoder is in its initial state after X and
       `lossy(X + Y) == lossy(X) + lossy(Y)` for every Y.

    The committed text, decode(ids[:start]), never ends with U+FFFD, so
    decode(ids) == committed + decode(ids[start:]). It is kept whole in `parts`
    until the first eligible step and afterwards only as its last `keep`
    characters in `tail`. Tokens are committed as soon as their window's text
    does not end with U+FFFD. The committed text is a prefix of
    the text of the previous step and of this one, and the previous step had no
    stop string in its text (or it would have ended the completion), so every
    occurrence in this step's text starts within the last `len(stop) - 1`
    characters of `committed` or later. The first eligible step scans the whole
    text, as upstream does.
    """

    __slots__ = ("decode", "stops", "keep", "start", "parts", "tail")

    def __init__(self, backend, stops: tuple[str, ...]):
        self.decode, self.stops = backend.decode, stops
        self.keep = max(len(stop) for stop in stops) - 1
        self.start, self.parts, self.tail = 0, [], ""

    def advance(self, ids: list[int], eligible: bool) -> bool | None:
        """Whether a stop string occurs in decode(ids); None when the window is too long."""
        window = self.decode(ids[self.start :], False)
        found = False
        if eligible:
            if self.parts is not None:
                region = "".join(self.parts) + window
                self.parts = None
            else:
                region = self.tail + window
            found = any(stop in region for stop in self.stops)
        if not window.endswith("\ufffd"):
            self.start = len(ids)
            if self.parts is not None:
                self.parts.append(window)
            self.tail = (self.tail + window)[-self.keep :] if self.keep else ""
        elif len(ids) - self.start > WINDOW_LIMIT:
            return None
        return found


_BACKENDS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
# Every pattern transformers' clean_up_tokenization rewrites.
_CLEANUP_PROBE = "a . b ? c ! d , e ' f n't g 'm h 's i 've j 're"


def byte_level_backend(tokenizer):
    """The `tokenizers.Tokenizer` behind `tokenizer.decode` when `_ByteLevelTail` is exact for it, else None."""
    try:
        return _BACKENDS[tokenizer]
    except (KeyError, TypeError):
        pass
    backend = _byte_level_backend(tokenizer)
    try:
        _BACKENDS[tokenizer] = backend
    except TypeError:
        pass
    return backend


def _byte_level_backend(tokenizer):
    # Look the classes up instead of importing them: an instance of one means its
    # module is loaded, and importing transformers.processing_utils for the check
    # would hold the GIL for over a second inside a request.
    processors = sys.modules.get("transformers.processing_utils")
    ProcessorMixin = getattr(processors, "ProcessorMixin", None)
    if ProcessorMixin is not None and isinstance(tokenizer, ProcessorMixin):
        # ProcessorMixin.decode forwards to its tokenizer.
        if type(tokenizer).decode is not ProcessorMixin.decode or "decode" in vars(tokenizer):
            return None
        tokenizer = getattr(tokenizer, "tokenizer", None)
    fast = sys.modules.get("transformers.tokenization_utils_tokenizers")
    TokenizersBackend = getattr(fast, "TokenizersBackend", None)
    if TokenizersBackend is None or not isinstance(tokenizer, TokenizersBackend):
        return None
    from tokenizers import decoders
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if (
        type(tokenizer).decode is not PreTrainedTokenizerBase.decode
        or type(tokenizer)._decode is not TokenizersBackend._decode
        or {"decode", "_decode"} & set(vars(tokenizer))
    ):
        return None
    backend = tokenizer.backend_tokenizer
    if type(backend.decoder) is not decoders.ByteLevel:
        return None
    # TokenizersBackend._decode cleans up the whole text, which is not
    # local, unless the flag is off or it skips cleanup for BPE models.
    if tokenizer.clean_up_tokenization_spaces and not (
        type(backend.model).__name__ == "BPE"
        and not getattr(tokenizer, "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output", False)
    ):
        return None
    probe = backend.encode(_CLEANUP_PROBE, add_special_tokens=False).ids
    if tokenizer.decode(probe, skip_special_tokens=False) != backend.decode(probe, False):
        return None
    return backend
