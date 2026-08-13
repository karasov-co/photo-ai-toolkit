"""One interface for "look at these images and answer", four ways to serve it.

Every call in this project has the same shape: a system prompt that is constant
across a batch, some images, some text between them, a token budget, and a
string back. That shape was written out three times against `responses.create`
-- in the preflight, in the content pass and in the artistic pass -- with the
model, the reasoning effort and the payload assembled slightly differently each
time. Changing provider meant finding all three.

So there is one method. Anything a provider cannot do is stated rather than
emulated: `reasoning_effort` is a hint, not a contract, and a provider without
one ignores it instead of pretending. The alternative -- silently mapping it
onto a thinking budget with different semantics -- produces a run that costs
what you did not expect and reasons differently than you asked.

**Only the OpenAI provider is exercised.** It is what the live runs used and
what the tests mock. The other three are written from each vendor's documented
request shape and have never been run against the real endpoint; they say so in
`verified`, and the CLI says so when you pick one. That is worth more than
quietly shipping three untested paths as though they were equivalent.
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


def from_openai_content(
    system: str,
    content: list[dict],
    *,
    max_tokens: int,
    reasoning_effort: str | None = "low",
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
    )


class ProviderError(RuntimeError):
    """The call failed. The original exception is the cause."""


class Provider:
    name = "unknown"
    verified = False

    def complete_vision(self, request: VisionRequest) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} model={getattr(self, 'model', '?')}>"


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
            import bootstrap

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
        if _is_truncated(response):
            raise Truncated(f"the reply hit the {request.max_tokens}-token limit")
        return getattr(response, "output_text", "") or ""


class Truncated(ProviderError):
    """The budget ran out mid-answer. Splitting the batch is the fix, not retrying."""


def _is_truncated(response) -> bool:
    if str(getattr(response, "status", "")) == "incomplete":
        return True
    details = getattr(response, "incomplete_details", None)
    return bool(details and "max_output_tokens" in str(getattr(details, "reason", details)))


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

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=request.max_tokens,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": content},
            ],
        )
        choices = getattr(response, "choices", []) or []
        return choices[0].message.content if choices else ""


PROVIDERS = {
    "openai": OpenAIProvider,
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
    if factory is OpenAICompatibleProvider:
        return factory(model, base_url=base_url, client=client)
    return factory(model, client=client)


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
