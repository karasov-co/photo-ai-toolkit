"""One interface, four implementations, and honesty about which of them runs.

The same call shape was written out three times against `responses.create` --
preflight, content pass, artistic pass -- each assembling the payload slightly
differently. These tests hold the two properties that were the point of
collapsing them: every caller goes through one method, and a provider that
cannot do something says so instead of emulating it.
"""

import json

import pytest

import llm_provider


class FakeResponse:
    def __init__(self, text="[]", status="completed"):
        self.output_text = text
        self.status = status


class FakeResponses:
    def __init__(self, response=None):
        self.response = response or FakeResponse()
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


class FakeOpenAI:
    def __init__(self, response=None):
        self.responses = FakeResponses(response)


def request(**overrides):
    payload = {
        "system": "be brief",
        "texts": ["Frame 1:", "Frame 2:"],
        "images": [llm_provider.Image("AAAA"), llm_provider.Image("BBBB")],
        "max_tokens": 500,
    }
    payload.update(overrides)
    return llm_provider.VisionRequest(**payload)


# --- the interface ------------------------------------------------------------


def test_every_provider_implements_the_one_method():
    for name, factory in llm_provider.PROVIDERS.items():
        assert hasattr(factory, "complete_vision"), name
        assert issubclass(factory, llm_provider.Provider), name


def test_only_the_openai_provider_claims_to_have_been_run():
    """Shipping three untested paths as equivalent would be the dishonest part."""
    assert llm_provider.OpenAIProvider.verified is True
    for name in ("anthropic", "gemini", "openai-compatible"):
        assert llm_provider.PROVIDERS[name].verified is False, name


def test_an_unknown_provider_fails_rather_than_defaulting():
    with pytest.raises(llm_provider.ProviderError, match="unknown provider"):
        llm_provider.build("something-else", "a-model")


def test_the_openai_provider_sends_images_and_text_in_order():
    client = FakeOpenAI()
    llm_provider.OpenAIProvider("m", client=client).complete_vision(request())

    content = client.responses.kwargs["input"][0]["content"]
    kinds = [b["type"] for b in content]
    assert kinds == ["input_text", "input_image", "input_text", "input_image"]
    assert content[1]["image_url"].endswith("AAAA")


def test_trailing_text_with_no_image_after_it_still_goes():
    client = FakeOpenAI()
    llm_provider.OpenAIProvider("m", client=client).complete_vision(
        request(texts=["one", "two", "three"], images=[llm_provider.Image("AAAA")])
    )
    texts = [b["text"] for b in client.responses.kwargs["input"][0]["content"]
             if b["type"] == "input_text"]
    assert texts == ["one", "two", "three"]


def test_reasoning_effort_is_passed_when_asked_for():
    client = FakeOpenAI()
    llm_provider.OpenAIProvider("m", client=client).complete_vision(
        request(reasoning_effort="high")
    )
    assert client.responses.kwargs["reasoning"] == {"effort": "high"}


def test_no_reasoning_key_when_the_caller_does_not_want_one():
    """It was hardcoded to low everywhere. Not every provider has the concept."""
    client = FakeOpenAI()
    llm_provider.OpenAIProvider("m", client=client).complete_vision(
        request(reasoning_effort=None)
    )
    assert "reasoning" not in client.responses.kwargs


def test_a_truncated_reply_is_its_own_error():
    """Retrying an identical request truncates identically; splitting is the fix."""
    client = FakeOpenAI(FakeResponse(text="[{", status="incomplete"))
    with pytest.raises(llm_provider.Truncated):
        llm_provider.OpenAIProvider("m", client=client).complete_vision(request())


# --- the adapter from the prompt builders -------------------------------------


def test_the_prompt_builders_convert_without_being_rewritten():
    import prompts

    frames = [{"key": "a.jpg", "views": [("full frame", "AA"), ("face", "BB")], "encoded": "AA"}]
    converted = llm_provider.from_openai_content(
        prompts.STAGE3_SYSTEM, prompts.stage3_user_content(frames), max_tokens=900
    )
    assert converted.system == prompts.STAGE3_SYSTEM
    assert len(converted.images) == 2
    assert converted.images[0].base64_jpeg == "AA"
    assert converted.max_tokens == 900


def test_conversion_strips_the_data_uri_prefix():
    content = [{"type": "input_image", "image_url": "data:image/jpeg;base64,ZZZZ"}]
    converted = llm_provider.from_openai_content("s", content, max_tokens=10)
    assert converted.images[0].base64_jpeg == "ZZZZ"


# --- the unverified three, shape only -----------------------------------------


def test_anthropic_sends_base64_source_blocks_not_data_uris():
    class Messages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return type("R", (), {"content": [type("B", (), {"text": "ok"})()]})()

    class Client:
        def __init__(self):
            self.messages = Messages()

    client = Client()
    out = llm_provider.AnthropicProvider("claude", client=client).complete_vision(request())

    assert out == "ok"
    blocks = client.messages.kwargs["messages"][0]["content"]
    image = next(b for b in blocks if b["type"] == "image")
    assert image["source"] == {
        "type": "base64", "media_type": "image/jpeg", "data": "AAAA"
    }
    assert client.messages.kwargs["system"] == "be brief"


def test_anthropic_drops_reasoning_effort_rather_than_inventing_one():
    """Mapping it onto extended thinking would change the price and the behaviour."""

    class Messages:
        kwargs = None

        def create(self, **kwargs):
            type(self).kwargs = kwargs
            return type("R", (), {"content": []})()

    client = type("C", (), {"messages": Messages()})()
    llm_provider.AnthropicProvider("claude", client=client).complete_vision(
        request(reasoning_effort="high")
    )
    assert "reasoning" not in Messages.kwargs
    assert "thinking" not in Messages.kwargs


def test_the_compatible_provider_uses_chat_completions():
    class Completions:
        kwargs = None

        def create(self, **kwargs):
            type(self).kwargs = kwargs
            message = type("M", (), {"content": "hello"})()
            return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

    client = type("C", (), {"chat": type("Ch", (), {"completions": Completions()})()})()
    out = llm_provider.OpenAICompatibleProvider(
        "local", base_url="http://localhost:8000/v1", client=client
    ).complete_vision(request())

    assert out == "hello"
    assert Completions.kwargs["messages"][0]["role"] == "system"
    blocks = Completions.kwargs["messages"][1]["content"]
    assert any(b["type"] == "image_url" for b in blocks)


# --- pricing ------------------------------------------------------------------


def test_pricing_lives_in_a_file_with_a_date():
    pricing = llm_provider.load_pricing()
    assert pricing["checked"], "a price with no date is a price nobody can trust"
    assert pricing["models"]


def test_the_default_model_is_priced():
    import bootstrap

    assert llm_provider.estimate_cost(bootstrap.DEFAULT_SEMANTIC_MODEL, 100) > 0


def test_an_unpriced_model_falls_back_rather_than_crashing():
    assert llm_provider.estimate_cost("something-nobody-listed", 100) > 0


def test_the_estimate_scales_with_the_batch():
    cheap = llm_provider.estimate_cost("gpt-5.6-terra", 100)
    assert llm_provider.estimate_cost("gpt-5.6-terra", 300) == pytest.approx(cheap * 3)


def test_the_pricing_file_is_valid_json():
    json.loads(llm_provider.PRICING_PATH.read_text(encoding="utf-8"))
