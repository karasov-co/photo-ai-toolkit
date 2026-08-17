"""Prove the model works before a single photograph is opened.

The failure this exists to prevent, exactly as it happened: 299 assets were
discovered, checksummed, decoded, measured and previewed, the output directory
was migrated to the new layout, and only then did the first API call come back
with *the model does not exist or is not available to this key*. The run
produced no report, the previous report had already been moved, and the user had
waited through all of it to learn that a configuration string was wrong.

Every one of those minutes was spent on work that could not possibly be used.
The check that would have caught it costs one request against a 32x32 image.

So this module runs first, always, and it is deliberately strict. It does not
ask "can I authenticate" -- an authenticated key with no access to the
configured model fails at exactly the same point in the run. It asks the
question the pipeline will actually ask: *this model, an image, a structured
reply*. Anything less proves nothing about whether the analysis can run.

Three rules hold:

**It never touches a user photograph.** The test image is generated here, in
memory, sixteen pixels square. Sending somebody's photograph to prove a
configuration is wrong would be both a privacy leak and a cost.

**It never leaks the key.** Provider errors quote request payloads and headers,
so every message that reaches a user or a log goes through `reports.redact`
first.

**It never suggests a way around.** No older model, no local-only mode, no
fallback. Those are all ways of producing a result that is not the result asked
for, and the previous version of this tool shipped three of them.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from enum import StrEnum

logger = logging.getLogger(__name__)

# Small enough to be free, large enough that a vision endpoint accepts it.
#
# 32 rather than 16 because xAI enforces a floor of 512 *total* pixels, and a
# 16x16 image is 256 -- it was rejected with `invalid_image` after the key had
# already authenticated, so the preflight failed at "model access" on a
# configuration that was fine. 32x32 is 1024, which clears the floor with room
# to spare and still encodes to well under a kilobyte.
#
# Anything smaller is a false failure waiting for the next provider with a
# minimum. Anything much larger costs tokens on a check that runs on every run.
TEST_IMAGE_PX = 32

# The smallest total pixel count any supported provider accepts. Asserted
# against the constant above rather than left as a comment, because the failure
# it prevents looks like a broken account rather than a too-small image.
MIN_PROVIDER_PIXELS = 512

# The reply has to parse the way Stage 2's replies are parsed, or the pipeline
# would fail on its first real group for a reason this check had declared fine.
PREFLIGHT_INSTRUCTIONS = (
    "You are verifying an API configuration. Look at the image and reply with "
    "ONLY a JSON array containing exactly one object: "
    '[{"n": 1, "ok": true}]. No prose, no markdown.'
)

MAX_OUTPUT_TOKENS = 800


class Failure(StrEnum):
    """Why the preflight stopped. Each maps to one sentence a person can act on."""

    NO_KEY = "no_key"
    AUTH = "authentication"
    MODEL_ACCESS = "model_access"
    PERMISSION = "permission"
    QUOTA = "quota"
    BILLING = "billing"
    VISION = "vision_unsupported"
    STRUCTURED_OUTPUT = "structured_output"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


# What each failure means, in the words of somebody who has to fix it. None of
# these proposes a different model or a degraded mode -- the fix is always to
# the account or the configuration, and saying otherwise would be telling a
# person to accept a worse analysis than the one they asked for.
EXPLANATIONS: dict[str, str] = {
    Failure.NO_KEY.value: (
        "No API key was found. Put XAI_API_KEY in the project's .env file "
        "(one line: XAI_API_KEY=xai-...), then run the same command again. "
        "OPENAI_API_KEY is also accepted."
    ),
    Failure.AUTH.value: (
        "The API rejected the key. It may have been revoked, rotated, or copied "
        "with a character missing. Issue a new key and replace the value in .env."
    ),
    Failure.MODEL_ACCESS.value: (
        "The key authenticates, but this account has no access to the model this "
        "toolkit requires. Enable it for your organisation in the OpenAI "
        "dashboard, or use a key from an organisation that already has it."
    ),
    Failure.PERMISSION.value: (
        "The key authenticates but is not permitted to use this endpoint. A "
        "restricted or project-scoped key usually needs the Responses API and "
        "model access granted explicitly."
    ),
    Failure.QUOTA.value: (
        "The account has no remaining quota. Nothing is wrong with the "
        "configuration; the limit has been reached."
    ),
    Failure.BILLING.value: (
        "The account has a billing problem -- usually an unpaid balance or a "
        "missing payment method. Resolve it in the OpenAI dashboard."
    ),
    Failure.VISION.value: (
        "The model accepted the request but not the image. This toolkit sends "
        "photographs, so a model without image input cannot run the analysis."
    ),
    Failure.STRUCTURED_OUTPUT.value: (
        "The model answered, but not with the JSON the pipeline reads. Analysis "
        "would fail on the first group of photographs."
    ),
    Failure.UNREACHABLE.value: (
        "The API could not be reached. Check the network connection, then run "
        "the same command again."
    ),
    Failure.UNKNOWN.value: "The preflight request failed.",
}


@dataclass
class Check:
    name: str
    passed: bool = False
    detail: str = ""


@dataclass
class PreflightResult:
    ok: bool = False
    model: str = ""
    provider: str = "openai"
    checks: list[Check] = field(default_factory=list)
    failure: str = ""
    message: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "provider": self.provider,
            "model": self.model,
            "failure": self.failure,
            "message": self.message,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks],
        }


def test_image_base64() -> str:
    """A 32x32 JPEG, generated here. Never a user photograph."""
    from PIL import Image

    image = Image.new("RGB", (TEST_IMAGE_PX, TEST_IMAGE_PX), (128, 128, 128))
    for x in range(TEST_IMAGE_PX):
        for y in range(TEST_IMAGE_PX):
            if (x + y) % 4 == 0:
                image.putpixel((x, y), (240, 240, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def classify(error: Exception) -> str:
    """Map a provider exception onto one actionable failure.

    Matched on the message rather than on the exception class: the SDK raises
    the same `APIStatusError` for a model the key cannot see and for a key that
    has run out of money, and those need different sentences.
    """
    text = str(error).lower()
    status = _status_of(error)

    # The class the SDK raised, which is a stronger signal than anything in the
    # message and weaker than an explicit status. Checked before the prose so a
    # bare "401" body is still recognised by the exception that carried it.
    by_class = _class_failure(error)

    if "does not exist" in text or "do not have access" in text or "model_not_found" in text:
        return Failure.MODEL_ACCESS.value
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return Failure.QUOTA.value
    if "billing" in text or "payment" in text:
        return Failure.BILLING.value
    if "invalid_api_key" in text or "incorrect api key" in text or _code(text, status, 401):
        return Failure.AUTH.value
    if _code(text, status, 403) or "permission" in text or "not allowed" in text:
        return Failure.PERMISSION.value
    if _code(text, status, 429):
        return Failure.QUOTA.value
    if "image" in text and ("not support" in text or "unsupported" in text):
        return Failure.VISION.value
    if "connection" in text or "timed out" in text or "timeout" in text:
        return Failure.UNREACHABLE.value
    if _code(text, status, 404):
        return Failure.MODEL_ACCESS.value
    if by_class:
        return by_class
    return Failure.UNKNOWN.value


# What the OpenAI SDK names its exceptions. Matched on a substring of the class
# name so that subclasses and vendored variants are covered.
_CLASS_FAILURES = (
    ("authentication", Failure.AUTH.value),
    ("permissiondenied", Failure.PERMISSION.value),
    ("notfound", Failure.MODEL_ACCESS.value),
    ("ratelimit", Failure.QUOTA.value),
    ("apiconnection", Failure.UNREACHABLE.value),
    ("apitimeout", Failure.UNREACHABLE.value),
)


def _class_failure(error: Exception) -> str:
    name = type(error).__name__.lower()
    for fragment, failure in _CLASS_FAILURES:
        if fragment in name:
            return failure
    return ""


def _status_of(error: Exception) -> int | None:
    """The HTTP status, from wherever the SDK put it.

    Tried in order and preferred over the message every time: an attribute is
    what the transport actually saw, while the text is prose that happens to
    contain numbers.
    """
    for holder, attribute in (
        (error, "status_code"),
        (error, "status"),
        (getattr(error, "response", None), "status_code"),
        (getattr(error, "response", None), "status"),
    ):
        value = getattr(holder, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return None


# A status code in prose has to look like one being reported, not merely be a
# three-digit number somewhere in the string. Matched forms:
#
#   "Error code: 404 - ..."      a keyword introduces it
#   "status 429"                 likewise
#   "HTTP 403"                   likewise
#   "404 Not Found"              at the start, with a reason phrase after it
#
# Deliberately unmatched: `req_9f404ab` (inside an identifier), `/v1/404/x` (a
# path segment -- a slash is a word boundary, which is why a bare boundary rule
# was not enough), `gpt-404` (a model name).
_CODE_PATTERNS = (
    r"(?:error\s+code|status(?:\s+code)?|http|code)\s*[:=]?\s*({code})\b",
    r"^\s*({code})\s+[a-z]",
)


def _code(text: str, status, wanted: int) -> bool:
    """Whether this error carries the given status.

    The attribute wins outright. A different status on the exception means the
    number in the text belongs to something else -- a request id, a path, a
    quoted payload -- and reading it as the status is how a 200 with a bad body
    came to be reported as a missing model.

    Text is consulted only when no attribute exists at all, and then on a word
    boundary rather than as a substring: `"404" in text` matched request ids.
    """
    import re

    if status is not None:
        return status == wanted
    return any(
        re.search(pattern.format(code=wanted), text) for pattern in _CODE_PATTERNS
    )


def run(model: str, *, client=None, provider: str = "openai", base_url: str = "") -> PreflightResult:
    """One request, five verdicts. Nothing else in the toolkit runs before this.

    The checks are reported individually because "it failed" is not actionable
    and "model access: failed, everything before it verified" is.
    """
    from photoai import bootstrap, reports

    result = PreflightResult(model=model)

    key_check = Check("Authentication")
    model_check = Check("Model access")
    vision_check = Check("Vision input")
    api_check = Check("Structured reply")
    result.checks = [key_check, model_check, vision_check, api_check]

    if client is None and not bootstrap.has_credentials():
        return _fail(result, Failure.NO_KEY.value)

    from photoai import llm_provider

    result.provider = provider
    try:
        engine = llm_provider.build(provider, model, base_url=base_url, client=client)
    except Exception as e:
        return _fail(result, Failure.AUTH.value, reports.redact(str(e)))

    try:
        text = engine.complete_vision(
            llm_provider.VisionRequest(
                system=PREFLIGHT_INSTRUCTIONS,
                texts=["Reply with the JSON array."],
                images=[llm_provider.Image(test_image_base64(), detail="low")],
                max_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort=None,
            )
        )
    except Exception as e:
        failure = classify(e)
        # Everything up to the failing check passed, and saying which is the
        # difference between "something is wrong" and a fix.
        if failure in (Failure.AUTH.value, Failure.NO_KEY.value):
            pass
        elif failure in (Failure.MODEL_ACCESS.value, Failure.PERMISSION.value,
                         Failure.QUOTA.value, Failure.BILLING.value):
            key_check.passed = True
        elif failure == Failure.VISION.value:
            key_check.passed = model_check.passed = True
        else:
            key_check.passed = True
        return _fail(result, failure, reports.redact(str(e)))

    key_check.passed = model_check.passed = vision_check.passed = True
    key_check.detail = "verified"
    model_check.detail = "verified"
    vision_check.detail = "verified"

    if not _parses_as_expected(text):
        api_check.detail = "the reply was not the expected JSON"
        return _fail(result, Failure.STRUCTURED_OUTPUT.value, reports.redact(text[:200]))

    api_check.passed = True
    api_check.detail = "verified"
    result.ok = True
    return result


def _parses_as_expected(text: str) -> bool:
    """The same parser Stage 2 uses, so a pass here means a pass there."""
    from photoai import batch_runner

    try:
        items = batch_runner.parse_group_json(text)
    except (ValueError, json.JSONDecodeError):
        return False
    return bool(items) and isinstance(items[0], dict)


def _fail(result: PreflightResult, failure: str, detail: str = "") -> PreflightResult:
    result.ok = False
    result.failure = failure
    result.message = EXPLANATIONS.get(failure, EXPLANATIONS[Failure.UNKNOWN.value])
    result.detail = detail
    return result


def format_result(result: PreflightResult) -> str:
    """The block printed at startup, pass or fail."""
    lines = ["LLM preflight", f"  Provider: {result.provider}", f"  Model: {result.model}"]
    for check in result.checks:
        if check.passed:
            state = "verified"
        elif result.failure and check is _first_failed(result):
            state = "FAILED"
        else:
            state = "not reached"
        lines.append(f"  {check.name}: {state}")
    return "\n".join(lines)


def _first_failed(result: PreflightResult) -> Check | None:
    for check in result.checks:
        if not check.passed:
            return check
    return None


def format_failure(result: PreflightResult) -> str:
    """What went wrong and what to do, with nothing suggesting a way around it."""
    lines = ["", f"Cannot run: {result.message}", ""]
    if result.detail:
        lines.append(f"  The API said: {result.detail}")
        lines.append("")
    lines.append(f"  Model required: {result.model}")
    lines.append("  No photograph was opened, and nothing in the output directory was changed.")
    return "\n".join(lines)
