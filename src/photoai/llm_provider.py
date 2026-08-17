"""One interface for "look at these images and answer", five ways to serve it.

Every call in this project has the same shape: a system prompt that is constant
across a batch, some images, some text between them, a token budget, and a
string back. That shape was written out three times against `responses.create`
-- in the preflight, in the content pass and in the artistic pass -- with the
model, the reasoning effort and the payload assembled slightly differently each
time. Changing provider meant finding all three.

So there is one method. Anything a provider cannot do is stated rather than
emulated. `reasoning_effort` is sent where the endpoint takes it and dropped
where it does not, once, after the endpoint says so by name -- rather than
silently mapped onto a thinking budget with different semantics, which produces
a run that costs what you did not expect and reasons differently than you asked.
Dropping it without asking was its own version of that mistake, and cost 627,000
unbudgeted reasoning tokens before anybody counted them.

**Two of the five have met a live endpoint: `openai` and `grok`.** grok is the
default and has 281 photographs through it across both passes; openai is what
the earlier live runs used and what the tests mock. `anthropic`, `gemini` and
`openai-compatible` are written from each vendor's documented request shape and
have never been run against the real thing. They say so in `verified`, and the
CLI says so when you pick one, which is worth more than quietly shipping
untested paths as though they were equivalent.

`verified` means exactly "this adapter has met the endpoint". It is not a claim
that it is correct.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

PRICING_PATH = Path(__file__).parent / "data" / "pricing.json"


@dataclass
class Image:
    """One image, already encoded. Providers differ only in how they wrap it."""

    base64_jpeg: str
    detail: str = "high"


@dataclass
class VisionRequest:
    """What every call in this project actually is."""

    system: str
    images: list[Image] = field(default_factory=list)
    # Text blocks interleaved with the images, in order. `texts[i]` is emitted
    # before `images[i]` where both exist, which is how the frames get labelled.
    texts: list[str] = field(default_factory=list)
    max_tokens: int = 1000
    # "low", "medium", "high", or None for a provider that has no such control.
    reasoning_effort: str | None = "low"
    # Which pass this is, so the usage log can say where the money went.
    stage: str = "?"


def from_openai_content(
    system: str,
    content: list[dict],
    *,
    max_tokens: int,
    reasoning_effort: str | None = "low",
    stage: str = "?",
) -> VisionRequest:
    """Adapt the prompt builders' output without rewriting them.

    `prompts.stage2_user_content` and `stage3_user_content` emit the Responses
    API's block list, and they are well tested. Converting here keeps them as
    they are and still gives every provider the same neutral request.
    """
    texts: list[str] = []
    images: list[Image] = []
    for block in content:
        kind = block.get("type")
        if kind == "input_text":
            texts.append(block.get("text", ""))
        elif kind == "input_image":
            url = block.get("image_url", "")
            images.append(
                Image(url.split("base64,", 1)[-1], detail=block.get("detail", "high"))
            )
    return VisionRequest(
        system=system,
        texts=texts,
        images=images,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        stage=stage,
    )


class ProviderError(RuntimeError):
    """The call failed. The original exception is the cause."""


class Truncated(ProviderError):
    """The budget ran out mid-answer. Splitting the batch is the fix, not retrying.

    Defined before the providers that raise it. It used to sit below
    `OpenAIProvider`, which referenced it -- legal at runtime, and the kind of
    ordering that reads as an accident because it is one.
    """


class Provider:
    name = "unknown"
    verified = False

    def complete_vision(self, request: VisionRequest) -> str:
        raise NotImplementedError

    def check_not_truncated(self, response, max_tokens: int) -> None:
        """Raise if the reply stopped because the budget ran out.

        On the base class because every provider needs it and only OpenAI had
        it. A truncated reply is a partial JSON document: it comes back as a
        string, parses as "no array in the reply", and gets retried identically
        until the attempts run out. That failure cost two minutes a group and
        three times the tokens before it was diagnosed on the OpenAI path, and
        the other three providers were shipped with the same hole.

        Each vendor reports it differently, so all four shapes are checked --
        an unrecognised shape means no exception, which is the safe direction:
        a false positive here would split a batch that was fine.
        """
        if _looks_truncated(response):
            raise Truncated(f"the reply hit the {max_tokens}-token limit")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={getattr(self, 'model', '?')}>"


def _looks_truncated(response) -> bool:
    """Every way the four providers say "I ran out of room"."""
    # OpenAI Responses: status plus a reason.
    if str(getattr(response, "status", "")) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    if details and "max_output_tokens" in str(getattr(details, "reason", details)):
        return True

    # Anthropic messages.
    if str(getattr(response, "stop_reason", "")) == "max_tokens":
        return True

    # Chat completions, which is what xAI and every self-hosted server speak.
    for choice in getattr(response, "choices", None) or []:
        if str(getattr(choice, "finish_reason", "")) == "length":
            return True

    # Gemini.
    for candidate in getattr(response, "candidates", None) or []:
        if "MAX_TOKENS" in str(getattr(candidate, "finish_reason", "")).upper():
            return True
    return False


# --- OpenAI: the one that runs -------------------------------------------------


class OpenAIProvider(Provider):
    name = "openai"
    verified = True

    def __init__(self, model: str, client=None):
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from photoai import bootstrap

            self._client = bootstrap.make_client()
        return self._client

    def complete_vision(self, request: VisionRequest) -> str:
        content: list[dict] = []
        for index, image in enumerate(request.images):
            if index < len(request.texts):
                content.append({"type": "input_text", "text": request.texts[index]})
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image.base64_jpeg}",
                    "detail": image.detail,
                }
            )
        for text in request.texts[len(request.images):]:
            content.append({"type": "input_text", "text": text})

        kwargs = {
            "model": self.model,
            "instructions": request.system,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": request.max_tokens,
        }
        if request.reasoning_effort:
            kwargs["reasoning"] = {"effort": request.reasoning_effort}

        response = self.client.responses.create(**kwargs)
        record_usage(self.model, request.stage, response)
        self.check_not_truncated(response, request.max_tokens)
        return getattr(response, "output_text", "") or ""


# --- the three that are written but not run ------------------------------------


class AnthropicProvider(Provider):
    """Claude's messages API. Images are base64 `source` blocks, not data URIs.

    No reasoning-effort control: `reasoning_effort` is dropped rather than
    mapped onto extended thinking, which is a different mechanism with a
    different price.
    """

    name = "anthropic"
    verified = False

    def __init__(self, model: str, client=None):
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete_vision(self, request: VisionRequest) -> str:
        content: list[dict] = []
        for index, image in enumerate(request.images):
            if index < len(request.texts):
                content.append({"type": "text", "text": request.texts[index]})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image.base64_jpeg,
                    },
                }
            )
        for text in request.texts[len(request.images):]:
            content.append({"type": "text", "text": text})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=request.max_tokens,
            system=request.system,
            messages=[{"role": "user", "content": content}],
        )
        self.check_not_truncated(response, request.max_tokens)
        blocks = getattr(response, "content", []) or []
        return "".join(getattr(b, "text", "") for b in blocks)


class GeminiProvider(Provider):
    """Google's generate_content. Images are inline_data parts."""

    name = "gemini"
    verified = False

    def __init__(self, model: str, client=None):
        self.model = model
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client()
        return self._client

    def complete_vision(self, request: VisionRequest) -> str:
        import base64

        parts: list[dict] = []
        for index, image in enumerate(request.images):
            if index < len(request.texts):
                parts.append({"text": request.texts[index]})
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": base64.b64decode(image.base64_jpeg),
                    }
                }
            )
        for text in request.texts[len(request.images):]:
            parts.append({"text": text})

        response = self.client.models.generate_content(
            model=self.model,
            contents=[{"role": "user", "parts": parts}],
            config={
                "system_instruction": request.system,
                "max_output_tokens": request.max_tokens,
            },
        )
        self.check_not_truncated(response, request.max_tokens)
        return getattr(response, "text", "") or ""


class OpenAICompatibleProvider(Provider):
    """vLLM, Ollama, LM Studio, anything speaking chat/completions at a base_url.

    Chat completions rather than the Responses API, because that is what the
    self-hosted servers implement.
    """

    name = "openai-compatible"
    verified = False

    def __init__(self, model: str, base_url: str = "", api_key: str = "", client=None):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key or "not-needed"
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import openai

            self._client = openai.OpenAI(base_url=self.base_url or None, api_key=self.api_key)
        return self._client

    def complete_vision(self, request: VisionRequest) -> str:
        content: list[dict] = []
        for index, image in enumerate(request.images):
            if index < len(request.texts):
                content.append({"type": "text", "text": request.texts[index]})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image.base64_jpeg}"},
                }
            )
        for text in request.texts[len(request.images):]:
            content.append({"type": "text", "text": text})

        # Nothing else goes on the wire. No presence_penalty, no
        # frequency_penalty, no stop -- xAI rejects parameters it does not
        # implement, and a 400 on the twelfth group is an expensive way to
        # learn that a default crept in.
        payload = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": content},
            ],
        }
        marker = (self.name, self.model)
        wants_effort = bool(request.reasoning_effort) and marker not in _EFFORT_REJECTED
        if wants_effort:
            payload["reasoning_effort"] = request.reasoning_effort

        try:
            response = self.client.chat.completions.create(**payload)
        except Exception as e:
            # A server that does not implement the parameter says so, by name.
            # Retry once without it rather than failing a paid run over a hint.
            if not wants_effort or "reasoning_effort" not in str(e):
                raise
            logger.info("%s rejected reasoning_effort; retrying without it", self.name)
            _EFFORT_REJECTED.add(marker)
            payload.pop("reasoning_effort")
            response = self.client.chat.completions.create(**payload)

        record_usage(self.model, request.stage, response)
        self.check_not_truncated(response, request.max_tokens)
        choices = getattr(response, "choices", []) or []
        return choices[0].message.content if choices else ""


class GrokProvider(OpenAICompatibleProvider):
    """xAI, which speaks chat/completions at api.x.ai.

    A subclass rather than a copy: the wire format is the same one every
    OpenAI-compatible server implements, and the only differences are the
    endpoint and where the key comes from.

    Two things it does NOT inherit from the OpenAI path:

`reasoning_effort` IS forwarded now, in the chat/completions payload. Dropping
    it was a decision made from caution and it cost money: the model reasoned at
    its own default on every call, 627K reasoning tokens against 109K of
    completion over 362 requests, all of it billed as output. A server that does
    not implement the parameter answers with a 400 naming it, which the base
    class catches once per model and then stops sending.

    Images are jpg or png, 20 MiB each, with no cap on how many per request --
    so a group of twelve frames goes in one call exactly as it does now. The
    previews this project sends are 512px JPEGs, three orders of magnitude
    under the size limit.
    """

    name = "grok"
    # Run against api.x.ai for real: 281 photographs through grok-4.6 across
    # Stage 2 and Stage 3, plus the preflight on every run. `verified` means
    # exactly that and nothing more -- it is not a statement that the adapter is
    # correct, only that it has met the endpoint rather than only the docs.
    verified = True

    DEFAULT_BASE_URL = "https://api.x.ai/v1"
    # 20 MiB per image, per xAI's documented limit. Nothing this project sends
    # comes close; the check exists so that if something ever does, it fails
    # here with a sentence rather than at the endpoint with a 400.
    MAX_IMAGE_BYTES = 20 * 1024 * 1024

    def __init__(self, model: str, base_url: str = "", api_key: str = "", client=None):
        super().__init__(
            model,
            base_url=base_url or self.DEFAULT_BASE_URL,
            api_key=api_key or _xai_key(),
            client=client,
        )

    def complete_vision(self, request: VisionRequest) -> str:
        for image in request.images:
            # base64 is 4 characters per 3 bytes.
            if len(image.base64_jpeg) * 3 // 4 > self.MAX_IMAGE_BYTES:
                raise ProviderError(
                    f"an image exceeds xAI's {self.MAX_IMAGE_BYTES // (1024 * 1024)} MiB limit"
                )
        return super().complete_vision(request)


def _xai_key() -> str:
    """XAI_API_KEY, or the OpenAI one.

    The fallback exists because most people arrive here with OPENAI_API_KEY
    already in their `.env`, and failing on a key that is sitting right there
    would be pedantry. A key that does not work is caught by the preflight in
    one request, which is a better place to find out than a config error.
    """
    import os

    return (
        os.environ.get("XAI_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
        or "not-needed"
    )


PROVIDERS = {
    "openai": OpenAIProvider,
    "grok": GrokProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
    "openai-compatible": OpenAICompatibleProvider,
}


def build(name: str, model: str, *, base_url: str = "", client=None) -> Provider:
    """One provider, by name. Unknown names fail loudly rather than defaulting."""
    factory = PROVIDERS.get((name or "openai").strip().lower())
    if factory is None:
        raise ProviderError(
            f"unknown provider {name!r}; available: {', '.join(sorted(PROVIDERS))}"
        )
    if issubclass(factory, OpenAICompatibleProvider):
        # Grok supplies its own endpoint when none is given; the generic
        # compatible provider has none to supply.
        return factory(model, base_url=base_url, client=client)
    return factory(model, client=client)


# --- what it actually cost -----------------------------------------------------
#
# A module-level list rather than provider state, because `_provider()` builds a
# fresh provider for every call and per-instance counters would be thrown away
# with it. Reset at the start of a pass, drained at the end.

USAGE: list[dict] = []

# xAI, and every OpenAI-compatible server, answer a rejected parameter with a
# 400 naming it. Recorded per (provider, model) so one rejection is enough --
# retrying the same parameter on every subsequent call would double the run.
_EFFORT_REJECTED: set[tuple[str, str]] = set()


def _int(value) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def record_usage(model: str, stage: str, response) -> dict:
    """One row per API call: what was sent, what came back, what was thought.

    Reasoning tokens are billed as output and were invisible here. A run quoted
    at $2 cost $5, and the gap was 627K reasoning tokens against 109K of actual
    completion -- six times more thinking than answering, none of it counted.
    """
    usage = getattr(response, "usage", None)
    completion_details = getattr(usage, "completion_tokens_details", None)
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    row = {
        "model": model,
        "stage": stage,
        "prompt": _int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)),
        "completion": _int(
            getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0)
        ),
        "reasoning": _int(getattr(completion_details, "reasoning_tokens", 0)),
        "cached": _int(getattr(prompt_details, "cached_tokens", 0)),
    }
    USAGE.append(row)
    return row


def usage_total(rows: list[dict] | None = None) -> dict:
    """Sum the rows, and price them with reasoning billed as output."""
    rows = USAGE if rows is None else rows
    total = {"calls": len(rows), "prompt": 0, "completion": 0, "reasoning": 0, "cached": 0}
    for row in rows:
        for field_name in ("prompt", "completion", "reasoning", "cached"):
            total[field_name] += _int(row.get(field_name))
    pricing = load_pricing()
    models = pricing.get("models") or {}
    usd = 0.0
    for row in rows:
        entry = models.get(row.get("model", "")) or {}
        rate_in = float(entry.get("usd_per_1m_input", 0.0) or 0.0)
        rate_out = float(entry.get("usd_per_1m_output", 0.0) or 0.0)
        billed_out = _int(row.get("completion")) + _int(row.get("reasoning"))
        usd += _int(row.get("prompt")) / 1e6 * rate_in + billed_out / 1e6 * rate_out
    total["usd"] = round(usd, 4)
    return total


def write_usage(path, rows: list[dict] | None = None):
    """Append the rows to `usage.jsonl` and return the total."""
    from pathlib import Path as _Path

    rows = USAGE if rows is None else rows
    target = _Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return usage_total(rows)


# --- what it costs -------------------------------------------------------------


def load_pricing() -> dict:
    """Prices per model, with the date somebody last checked them.

    In a data file rather than in code because it goes stale on the vendor's
    schedule, not on this project's, and because a number nobody can see the
    provenance of is a number nobody should be quoting at a user.
    """
    try:
        return json.loads(PRICING_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read %s: %s", PRICING_PATH.name, e)
        return {"models": {}, "checked": "unknown", "default_usd_per_100_photos": 1.0}


def estimate_cost(model: str, photographs: int) -> float:
    """A rough figure in dollars, for warning before a large run."""
    pricing = load_pricing()
    entry = (pricing.get("models") or {}).get(model)
    per_hundred = (
        entry.get("usd_per_100_photos")
        if isinstance(entry, dict)
        else pricing.get("default_usd_per_100_photos", 1.0)
    )
    return round(float(per_hundred) * max(0, photographs) / 100.0, 2)


def pricing_checked_on() -> str:
    return str(load_pricing().get("checked", "unknown"))
