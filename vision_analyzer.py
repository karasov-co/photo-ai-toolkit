import base64
import contextlib
import json
import logging
import re
import time
from pathlib import Path

import openai

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "You are a senior photo editor reviewing a photographer's archive to identify their strongest work. "
    "Evaluate this single photograph on its photographic merit alone.\n\n"

    "Score from 1 to 1000 based on how the image performs across these dimensions, weighted equally:\n"
    "- Composition: framing, balance, use of space, leading lines, subject placement\n"
    "- Light: quality, direction, contrast, exposure, dynamic range handling\n"
    "- Color: harmony, palette, white balance, mood\n"
    "- Subject and storytelling: clarity of intent, emotional weight, narrative\n"
    "- Technical execution: focus, sharpness, noise control, artifacts\n"
    "- Originality: visual interest, scroll-stopping quality, memorability\n\n"

    "Scoring anchors:\n"
    "1-200: technical failure (severe blur, total miss-focus, unrecoverable exposure) or visually empty\n"
    "201-400: weak — multiple significant problems, no redeeming qualities\n"
    "401-600: competent but unremarkable — looks fine, no reason to keep\n"
    "601-750: strong — clearly above average, worth keeping in a portfolio\n"
    "751-900: excellent — would publish, share, or print\n"
    "901-1000: exceptional — career-defining, gallery-grade work\n\n"

    "Critical rules:\n"
    "- Aspect ratio and orientation (portrait, landscape, square) are creative choices. "
    "Do NOT lower the score because of orientation, format, or fitness for any specific platform like Instagram.\n"
    "- Be discriminating. Spread scores across the full range. Most photos in a typical archive land in 400-650. "
    "Reserve 750+ for genuinely strong work and 850+ for outstanding shots only.\n"
    "- Judge each photo on its own, not relative to others.\n\n"

    "Respond with valid JSON only, no preamble or markdown:\n"
    '{"description": "one sentence describing what is in the frame", '
    '"tags": ["5-10 lowercase keywords"], '
    '"quality_score": <integer 1-1000>, '
    '"quality_reasoning": "2-4 sentences explaining the score, naming concrete strengths and weaknesses"}'
)

MODEL = "gpt-5.5"

# GPT-5.x spends part of its output budget on reasoning tokens before it emits
# any text, so this ceiling has to cover reasoning + the JSON payload.
MAX_OUTPUT_TOKENS = 2000

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2

DRY_RUN_STUB = {
    "description": "DRY RUN: A sample scene description for testing purposes.",
    "tags": ["dry-run", "test", "sample"],
    "quality_score": 5,
    "quality_reasoning": "DRY RUN: No actual analysis performed.",
}


class VisionAnalysisError(Exception):
    pass


class VisionParseError(VisionAnalysisError):
    pass


def analyze_photo(preview_path: Path, client: openai.OpenAI, dry_run: bool = False) -> dict:
    if dry_run:
        return dict(DRY_RUN_STUB)

    encoded = _encode_image_base64(preview_path)
    raw_response = _retry_with_backoff(_call_vision_api, encoded, client)
    return _parse_vision_response(raw_response)


def _encode_image_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _call_vision_api(encoded: str, client: openai.OpenAI) -> str:
    response = client.responses.create(
        model=MODEL,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        reasoning={"effort": "low"},
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded}",
                        "detail": "low",
                    },
                    {
                        "type": "input_text",
                        "text": VISION_PROMPT,
                    },
                ],
            }
        ],
    )

    text = response.output_text or ""
    if not text.strip():
        reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
        raise VisionParseError(
            f"Model returned no text (status={getattr(response, 'status', '?')}, reason={reason})"
        )
    return text


def _parse_vision_response(raw: str) -> dict:
    data = None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            with contextlib.suppress(json.JSONDecodeError):
                data = json.loads(match.group(0))

    if not isinstance(data, dict) or not data:
        raise VisionParseError(f"Could not extract JSON from API response: {raw[:200]}")

    required = {"description", "tags", "quality_score", "quality_reasoning"}
    missing = required - set(data.keys())
    if missing:
        raise VisionParseError(f"API response missing required keys: {missing}")

    try:
        score = max(1, min(1000, int(data["quality_score"])))
    except (ValueError, TypeError):
        score = 1

    tags = data["tags"] if isinstance(data["tags"], list) else []
    tags = [str(t) for t in tags[:10]]

    return {
        "description": str(data["description"]),
        "tags": tags,
        "quality_score": score,
        "quality_reasoning": str(data["quality_reasoning"]),
    }


def _retry_with_backoff(func, *args, **kwargs):
    retryable = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
    )
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except openai.BadRequestError:
            raise
        except retryable as e:
            last_exc = e
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("API call failed (attempt %d/%d): %s. Retrying in %ds...",
                           attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)
        except openai.APIError as e:
            last_exc = e
            wait = RETRY_BACKOFF_BASE ** attempt
            logger.warning("API error (attempt %d/%d): %s. Retrying in %ds...",
                           attempt + 1, MAX_RETRIES, e, wait)
            time.sleep(wait)
    raise VisionAnalysisError(f"API call failed after {MAX_RETRIES} retries: {last_exc}") from last_exc
