import base64
import json
import logging
import re
import time
from pathlib import Path

import openai

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    'You are an elite photography curator selecting the best shots for Instagram Stories. '
    'Score each photo on a 1-1000 scale where: 1-200 = technical failure or boring, '
    '201-400 = mediocre, 401-600 = decent but not post-worthy, 601-800 = good, would engage followers, '
    '801-1000 = exceptional, must post immediately. '
    'Evaluate: composition, lighting, color harmony, emotional impact, storytelling, Instagram appeal. '
    'Be very discriminating - spread scores widely, avoid clustering. '
    'Respond ONLY in valid JSON: {"description": "...", "tags": [...], '
    '"quality_score": 750, "quality_reasoning": "..."}'
)

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
    response = client.chat.completions.create(
        model="gpt-5.4",
        max_completion_tokens=500,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                            "detail": "low",
                        },
                    },
                    {
                        "type": "text",
                        "text": VISION_PROMPT,
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def _parse_vision_response(raw: str) -> dict:
    data = None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    if not data:
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
