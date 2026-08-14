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


# --- xAI ----------------------------------------------------------------------


class FakeChoice:
    def __init__(self, content="ok", finish_reason="stop"):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish_reason


class FakeChat:
    def __init__(self, choice=None):
        self.completions = self
        self.kwargs = None
        self.choice = choice or FakeChoice()

    def create(self, **kwargs):
        self.kwargs = kwargs
        return type("R", (), {"choices": [self.choice]})()


class FakeCompatible:
    def __init__(self, choice=None):
        self.chat = FakeChat(choice)


def test_grok_is_registered_and_unverified():
    assert llm_provider.PROVIDERS["grok"] is llm_provider.GrokProvider
    assert llm_provider.GrokProvider.verified is False


def test_grok_points_at_xai_by_default():
    assert llm_provider.build("grok", "grok-4.6").base_url == "https://api.x.ai/v1"


def test_an_explicit_base_url_still_wins():
    """For a proxy, or a gateway in front of xAI."""
    engine = llm_provider.build("grok", "grok-4.6", base_url="http://localhost:9000/v1")
    assert engine.base_url == "http://localhost:9000/v1"


def test_grok_speaks_chat_completions_like_the_compatible_provider():
    client = FakeCompatible()
    out = llm_provider.GrokProvider("grok-4.6", client=client).complete_vision(request())

    assert out == "ok"
    assert client.chat.kwargs["model"] == "grok-4.6"
    blocks = client.chat.kwargs["messages"][1]["content"]
    assert [b["type"] for b in blocks] == ["text", "image_url", "text", "image_url"]
    assert blocks[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_grok_forwards_the_reasoning_effort():
    """It used to be dropped, so every call reasoned at the vendor's default.

    That is where the money went: 627K reasoning tokens against 109K of
    completion, all billed as output, on a run quoted at less than half its
    eventual cost.
    """
    client = FakeCompatible()
    llm_provider.GrokProvider("grok-4.6", client=client).complete_vision(
        request(reasoning_effort="high")
    )
    assert client.chat.kwargs["reasoning_effort"] == "high"


def test_nothing_unsupported_goes_to_xai():
    """xAI rejects parameters it does not implement, with a 400, mid-run."""
    client = FakeCompatible()
    llm_provider.GrokProvider("grok-4.6", client=client).complete_vision(request())
    assert set(client.chat.kwargs) <= {
        "model", "max_tokens", "messages", "reasoning_effort"
    }
    for unsupported in ("presence_penalty", "frequency_penalty", "stop", "temperature",
                        "top_p", "n", "logprobs"):
        assert unsupported not in client.chat.kwargs


def test_a_rejected_effort_is_dropped_once_and_not_retried_forever():
    """A server without the parameter says so by name. Retry once, then stop."""

    client = FakeCompatible()
    attempts = []
    original = client.chat.completions.create

    def create(**kwargs):
        attempts.append(dict(kwargs))
        if "reasoning_effort" in kwargs:
            raise ValueError("400: unsupported parameter: reasoning_effort")
        return original(**kwargs)

    client.chat.completions.create = create
    llm_provider._EFFORT_REJECTED.discard(("grok", "picky-model"))
    provider = llm_provider.GrokProvider("picky-model", client=client)
    provider.complete_vision(request(reasoning_effort="high"))
    assert len(attempts) == 2
    assert "reasoning_effort" not in attempts[1]

    provider.complete_vision(request(reasoning_effort="high"))
    assert len(attempts) == 3  # remembered; not tried a second time
    llm_provider._EFFORT_REJECTED.discard(("grok", "picky-model"))


def test_usage_is_recorded_per_call_with_reasoning_separated():
    llm_provider.USAGE.clear()
    client = FakeCompatible()
    llm_provider.GrokProvider("grok-4.6", client=client).complete_vision(request())
    total = llm_provider.usage_total()
    assert total["calls"] == 1
    assert set(total) >= {"prompt", "completion", "reasoning", "cached", "usd"}
    llm_provider.USAGE.clear()


def test_reasoning_tokens_are_priced_as_output():
    rows = [{"model": "grok-4.6", "stage": "stage2", "prompt": 1_000_000,
             "completion": 0, "reasoning": 1_000_000, "cached": 0}]
    entry = llm_provider.load_pricing()["models"]["grok-4.6"]
    expected = entry["usd_per_1m_input"] + entry["usd_per_1m_output"]
    assert llm_provider.usage_total(rows)["usd"] == round(expected, 4)


def test_the_price_estimate_includes_a_reasoning_term():
    """It did not, and the estimate came in at less than half the bill."""
    formula = llm_provider.load_pricing()["_formula"]
    assert formula["reasoning_tokens_per_photo"] > 0
    entry = llm_provider.load_pricing()["models"]["grok-4.6"]
    expected = (
        formula["input_tokens_per_photo"] * 100 / 1e6 * entry["usd_per_1m_input"]
        + (formula["output_tokens_per_photo"] + formula["reasoning_tokens_per_photo"])
        * 100 / 1e6 * entry["usd_per_1m_output"]
    )
    assert abs(entry["usd_per_100_photos"] - expected) < 0.01


def test_all_twelve_frames_go_in_one_request():
    """xAI caps image size, not image count, so the group is not split."""
    client = FakeCompatible()
    images = [llm_provider.Image(f"IMG{i}") for i in range(12)]
    llm_provider.GrokProvider("grok-4.6", client=client).complete_vision(
        request(images=images, texts=[f"Frame {i}:" for i in range(12)])
    )
    blocks = client.chat.kwargs["messages"][1]["content"]
    assert sum(1 for b in blocks if b["type"] == "image_url") == 12


def test_an_oversized_image_is_refused_before_the_call():
    """A sentence here beats a 400 from the endpoint."""
    client = FakeCompatible()
    huge = llm_provider.Image("A" * (llm_provider.GrokProvider.MAX_IMAGE_BYTES * 4 // 3 + 8))
    with pytest.raises(llm_provider.ProviderError, match="20 MiB"):
        llm_provider.GrokProvider("grok-4.6", client=client).complete_vision(
            request(images=[huge], texts=[])
        )
    assert client.chat.kwargs is None, "the request was sent anyway"


def test_the_key_comes_from_xai_first_then_openai(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-one")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-two")
    assert llm_provider.GrokProvider("grok-4.6").api_key == "xai-one"

    monkeypatch.delenv("XAI_API_KEY")
    assert llm_provider.GrokProvider("grok-4.6").api_key == "sk-two"


# --- truncation, for every provider -------------------------------------------


def test_a_length_finish_reason_is_a_truncated_reply():
    """The hole this closes: three providers returned partial JSON silently."""
    client = FakeCompatible(FakeChoice(content='[{"n": 1', finish_reason="length"))
    with pytest.raises(llm_provider.Truncated):
        llm_provider.GrokProvider("grok-4.6", client=client).complete_vision(request())


def test_anthropics_max_tokens_stop_reason_is_truncation():
    class Messages:
        def create(self, **kwargs):
            return type("R", (), {"stop_reason": "max_tokens", "content": []})()

    client = type("C", (), {"messages": Messages()})()
    with pytest.raises(llm_provider.Truncated):
        llm_provider.AnthropicProvider("claude", client=client).complete_vision(request())


def test_geminis_max_tokens_candidate_is_truncation():
    class Models:
        def generate_content(self, **kwargs):
            candidate = type("C", (), {"finish_reason": "MAX_TOKENS"})()
            return type("R", (), {"candidates": [candidate], "text": "partial"})()

    client = type("C", (), {"models": Models()})()
    with pytest.raises(llm_provider.Truncated):
        llm_provider.GeminiProvider("gemini", client=client).complete_vision(request())


def test_a_normal_reply_is_not_mistaken_for_a_truncated_one():
    for response in (
        type("R", (), {"choices": [FakeChoice(finish_reason="stop")]})(),
        type("R", (), {"stop_reason": "end_turn", "content": []})(),
        type("R", (), {"status": "completed"})(),
        type("R", (), {})(),
    ):
        assert not llm_provider._looks_truncated(response)


def test_truncated_is_defined_before_the_providers_that_raise_it():
    """It sat below OpenAIProvider, which referenced it. Legal, and an accident."""
    source = llm_provider.PRICING_PATH.parent.parent / "llm_provider.py"
    text = source.read_text(encoding="utf-8")
    assert text.index("class Truncated") < text.index("class OpenAIProvider")


# --- pricing for the new default ----------------------------------------------


def test_grok_is_the_default_priced_model():
    pricing = llm_provider.load_pricing()
    assert pricing["default_model"] == "grok-4.6"
    assert "grok-4.6" in pricing["models"]


def test_the_grok_price_is_derived_from_its_token_rates():
    """Not a guess: the arithmetic in the file has to produce the figure in it."""
    pricing = llm_provider.load_pricing()
    formula = pricing["_formula"]
    entry = pricing["models"]["grok-4.6"]

    expected = (
        formula["input_tokens_per_photo"] * 100 / 1e6 * entry["usd_per_1m_input"]
        + (formula["output_tokens_per_photo"] + formula["reasoning_tokens_per_photo"])
        * 100 / 1e6 * entry["usd_per_1m_output"]
    )
    assert entry["usd_per_100_photos"] == pytest.approx(expected, abs=0.01)
    assert entry["derived"] is True


def test_the_undserived_figures_say_so():
    """The gpt-5.6 numbers predate the formula and are order-of-magnitude only."""
    models = llm_provider.load_pricing()["models"]
    for name in ("gpt-5.6-terra", "gpt-5.6-sol"):
        assert models[name]["derived"] is False, name


def test_every_model_carries_the_date_it_was_checked():
    for name, entry in llm_provider.load_pricing()["models"].items():
        assert entry.get("checked"), name
