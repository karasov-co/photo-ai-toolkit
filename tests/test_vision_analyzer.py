import base64
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from conftest import WITH_EXIF

import vision_analyzer
from vision_analyzer import (
    DRY_RUN_STUB,
    MAX_OUTPUT_TOKENS,
    MAX_RETRIES,
    MODEL,
    VISION_PROMPT,
    VisionAnalysisError,
    VisionParseError,
    _parse_vision_response,
    analyze_photo,
)

VALID_PAYLOAD = {
    "description": "A stone relief lit by a narrow shaft of light.",
    "tags": ["temple", "stone", "shadow"],
    "quality_score": 762,
    "quality_reasoning": "Strong atmosphere and confident use of directional light.",
}


# --- test doubles -----------------------------------------------------------


class FakeClient:
    """Mimics the slice of openai.OpenAI that vision_analyzer actually touches."""

    def __init__(self, *, text=None, results=None, status="completed", reason=None):
        if results is None:
            results = [text if text is not None else json.dumps(VALID_PAYLOAD)]
        self._results = list(results)
        self._status = status
        self._reason = reason
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._results.pop(0) if len(self._results) > 1 else self._results[0]
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(
            output_text=result,
            status=self._status,
            incomplete_details=SimpleNamespace(reason=self._reason),
        )


def _api_error(cls, status_code=500):
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    if cls in (openai.APIConnectionError, openai.APITimeoutError):
        return cls(request=request)
    return cls("boom", response=httpx.Response(status_code, request=request), body=None)


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch):
    """Retry tests must not actually wait out the backoff."""
    slept = []
    monkeypatch.setattr(vision_analyzer.time, "sleep", slept.append)
    return slept


# --- dry run ----------------------------------------------------------------


def test_dry_run_returns_the_stub_without_calling_the_api():
    client = FakeClient()
    assert analyze_photo(WITH_EXIF, client, dry_run=True) == DRY_RUN_STUB
    assert client.calls == []


def test_dry_run_stub_matches_the_real_output_schema():
    """It scored 5 on a 1-1000 scale, so every dry run averaged 5/1000."""
    assert set(DRY_RUN_STUB) == {"description", "tags", "quality_score", "quality_reasoning"}
    assert 1 <= DRY_RUN_STUB["quality_score"] <= 1000
    assert isinstance(DRY_RUN_STUB["quality_score"], int)
    assert len(DRY_RUN_STUB["tags"]) <= 10


def test_dry_run_returns_a_copy_the_caller_cannot_corrupt():
    result = analyze_photo(WITH_EXIF, FakeClient(), dry_run=True)
    result["description"] = "mutated"
    assert DRY_RUN_STUB["description"].startswith("DRY RUN")
    assert analyze_photo(WITH_EXIF, FakeClient(), dry_run=True) == DRY_RUN_STUB


# --- request shape ----------------------------------------------------------


def test_request_targets_the_configured_model_and_budget():
    client = FakeClient()
    analyze_photo(WITH_EXIF, client)
    call = client.calls[0]
    assert call["model"] == MODEL
    assert call["max_output_tokens"] == MAX_OUTPUT_TOKENS
    assert call["reasoning"] == {"effort": "low"}


def test_request_uses_the_responses_api_input_content_types():
    client = FakeClient()
    analyze_photo(WITH_EXIF, client)
    content = client.calls[0]["input"][0]["content"]
    assert client.calls[0]["input"][0]["role"] == "user"
    assert [part["type"] for part in content] == ["input_image", "input_text"]


def test_image_is_sent_as_a_low_detail_base64_data_uri():
    client = FakeClient()
    analyze_photo(WITH_EXIF, client)
    image = client.calls[0]["input"][0]["content"][0]
    assert image["detail"] == "low"
    assert image["image_url"].startswith("data:image/jpeg;base64,")
    payload = image["image_url"].split(",", 1)[1]
    assert base64.standard_b64decode(payload) == WITH_EXIF.read_bytes()


def test_the_scoring_prompt_is_sent_verbatim():
    client = FakeClient()
    analyze_photo(WITH_EXIF, client)
    assert client.calls[0]["input"][0]["content"][1]["text"] == VISION_PROMPT


# --- response parsing -------------------------------------------------------


def test_parses_a_clean_json_object():
    assert _parse_vision_response(json.dumps(VALID_PAYLOAD)) == VALID_PAYLOAD


def test_parses_json_wrapped_in_a_markdown_fence():
    raw = f"```json\n{json.dumps(VALID_PAYLOAD)}\n```"
    assert _parse_vision_response(raw)["quality_score"] == 762


def test_parses_json_buried_in_prose():
    raw = f"Sure! Here is my analysis:\n{json.dumps(VALID_PAYLOAD)}\nHope that helps."
    assert _parse_vision_response(raw)["description"] == VALID_PAYLOAD["description"]


def test_returns_exactly_the_four_schema_keys():
    assert set(_parse_vision_response(json.dumps(VALID_PAYLOAD))) == {
        "description",
        "tags",
        "quality_score",
        "quality_reasoning",
    }


def test_analyze_photo_returns_the_parsed_payload():
    assert analyze_photo(WITH_EXIF, FakeClient()) == VALID_PAYLOAD


# --- malformed responses ----------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "I'm sorry, I can't analyze this image.",
        "{ not valid json at all",
        "[1, 2, 3]",
        "{}",
        "null",
    ],
    ids=["empty", "whitespace", "prose", "broken-braces", "array", "empty-object", "null"],
)
def test_unparseable_responses_raise_vision_parse_error(raw):
    with pytest.raises(VisionParseError):
        _parse_vision_response(raw)


@pytest.mark.parametrize("missing", sorted(VALID_PAYLOAD))
def test_a_missing_required_key_raises_vision_parse_error(missing):
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != missing}
    with pytest.raises(VisionParseError, match="missing required keys"):
        _parse_vision_response(json.dumps(payload))


def test_parse_errors_are_a_kind_of_vision_analysis_error():
    assert issubclass(VisionParseError, VisionAnalysisError)


def test_an_empty_model_response_reports_the_incomplete_reason():
    client = FakeClient(text="", status="incomplete", reason="max_output_tokens")
    with pytest.raises(VisionParseError, match="max_output_tokens"):
        analyze_photo(WITH_EXIF, client)


def test_an_empty_response_is_not_retried():
    client = FakeClient(text="", status="incomplete", reason="max_output_tokens")
    with pytest.raises(VisionParseError):
        analyze_photo(WITH_EXIF, client)
    assert len(client.calls) == 1


# --- score validation -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [
        (762, 762),
        (1, 1),
        (1000, 1000),
        (0, 1),
        (-50, 1),
        (1001, 1000),
        (99999, 1000),
        ("850", 850),
        (762.9, 762),
        (None, 1),
        ("high", 1),
        ([700], 1),
        ({}, 1),
    ],
)
def test_scores_are_coerced_into_the_1_to_1000_range(raw_score, expected):
    result = _parse_vision_response(json.dumps({**VALID_PAYLOAD, "quality_score": raw_score}))
    assert result["quality_score"] == expected
    assert isinstance(result["quality_score"], int)


@pytest.mark.parametrize("raw_score", [762, "850", 1001, None, "high"])
def test_scores_always_land_inside_the_documented_range(raw_score):
    score = _parse_vision_response(json.dumps({**VALID_PAYLOAD, "quality_score": raw_score}))[
        "quality_score"
    ]
    assert 1 <= score <= 1000


# --- tags -------------------------------------------------------------------


def test_tags_are_capped_at_ten():
    payload = {**VALID_PAYLOAD, "tags": [f"tag{i}" for i in range(25)]}
    assert len(_parse_vision_response(json.dumps(payload))["tags"]) == 10


@pytest.mark.parametrize("bad_tags", ["not-a-list", 42, None, {"a": 1}])
def test_non_list_tags_become_an_empty_list(bad_tags):
    payload = {**VALID_PAYLOAD, "tags": bad_tags}
    assert _parse_vision_response(json.dumps(payload))["tags"] == []


def test_tag_elements_are_coerced_to_strings():
    payload = {**VALID_PAYLOAD, "tags": ["temple", 42, None, True]}
    assert _parse_vision_response(json.dumps(payload))["tags"] == ["temple", "42", "None", "True"]


def test_text_fields_are_coerced_to_strings():
    payload = {**VALID_PAYLOAD, "description": 123, "quality_reasoning": None}
    result = _parse_vision_response(json.dumps(payload))
    assert result["description"] == "123"
    assert isinstance(result["quality_reasoning"], str)


# --- retry behaviour --------------------------------------------------------


@pytest.mark.parametrize(
    "error_cls",
    [openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError],
)
def test_transient_errors_are_retried_then_succeed(error_cls):
    client = FakeClient(results=[_api_error(error_cls), json.dumps(VALID_PAYLOAD)])
    assert analyze_photo(WITH_EXIF, client)["quality_score"] == 762
    assert len(client.calls) == 2


def test_generic_api_errors_are_also_retried():
    error = openai.APIError("boom", request=httpx.Request("POST", "https://x"), body=None)
    client = FakeClient(results=[error, json.dumps(VALID_PAYLOAD)])
    assert analyze_photo(WITH_EXIF, client)["quality_score"] == 762


def test_bad_requests_fail_immediately_without_retrying():
    client = FakeClient(results=[_api_error(openai.BadRequestError, 400)])
    with pytest.raises(openai.BadRequestError):
        analyze_photo(WITH_EXIF, client)
    assert len(client.calls) == 1


def test_exhausted_retries_raise_vision_analysis_error():
    client = FakeClient(results=[_api_error(openai.RateLimitError)])
    with pytest.raises(VisionAnalysisError, match=f"failed after {MAX_RETRIES} retries"):
        analyze_photo(WITH_EXIF, client)
    assert len(client.calls) == MAX_RETRIES


def test_backoff_grows_exponentially(_no_sleeping):
    client = FakeClient(results=[_api_error(openai.RateLimitError)])
    with pytest.raises(VisionAnalysisError):
        analyze_photo(WITH_EXIF, client)
    assert _no_sleeping == [1, 2, 4]


def test_a_successful_first_call_never_sleeps(_no_sleeping):
    analyze_photo(WITH_EXIF, FakeClient())
    assert _no_sleeping == []
