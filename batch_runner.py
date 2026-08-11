"""Running both stages through the Batch API, and counting what they cost.

The Batch API is half price and its only completion window is 24h, so a run is
something you start and come back to -- not something you watch. That shapes the
design: the batch id is written to disk the moment it exists, and `resume`
picks a run back up from that file. A crashed terminal must not cost a rerun.

Everything that builds a request or parses a result is a pure function, so the
shapes are tested without spending anything. Only `submit`, `wait` and `fetch`
touch the network.

The system prompt goes in `instructions`, which is the front of the cached
prefix -- identical across every request in a batch, which is the whole reason
it is a module-level constant rather than something assembled per frame.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import prompts

logger = logging.getLogger(__name__)

RESPONSES_ENDPOINT = "/v1/responses"
COMPLETION_WINDOW = "24h"
POLL_SECONDS = 30

TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


# --- building requests ------------------------------------------------------


def stage1_request(custom_id: str, encoded_jpeg: str, model: str) -> dict:
    """One garbage-filter call. Low detail, one word out."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": RESPONSES_ENDPOINT,
        "body": {
            "model": model,
            "instructions": prompts.STAGE1_SYSTEM,
            "input": [{"role": "user", "content": prompts.stage1_user_content(encoded_jpeg)}],
            "max_output_tokens": prompts.STAGE1_MAX_OUTPUT_TOKENS,
            "reasoning": {"effort": "low"},
        },
    }


def stage2_request(custom_id: str, frames: list[dict], model: str) -> dict:
    """One group of frames, ranked against each other. High detail."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": RESPONSES_ENDPOINT,
        "body": {
            "model": model,
            "instructions": prompts.STAGE2_SYSTEM,
            "input": [{"role": "user", "content": prompts.stage2_user_content(frames)}],
            "max_output_tokens": _stage2_budget(len(frames)),
            "reasoning": {"effort": "low"},
        },
    }


def _stage2_budget(frame_count: int) -> int:
    """Room for one compact object per frame, plus reasoning headroom.

    Reasoning tokens come out of the same ceiling as the text, so this cannot be
    sized from the JSON alone -- a group that runs out mid-array returns nothing
    usable and the whole group has to be re-run.
    """
    return 900 + 260 * frame_count


class MixedModelBatch(ValueError):
    pass


def assert_single_model(requests: list[dict]) -> str:
    """A batch may only target one model. Checked before upload, not after.

    The API enforces this, but it enforces it by failing the *entire* batch
    after validation with `mismatched_model`. Since Stage 1 runs on the cheap
    model and Stage 2 on the expensive one, putting them in one file is an easy
    mistake, and finding out costs a full round trip through the queue.
    """
    models = {r["body"]["model"] for r in requests}
    if len(models) > 1:
        raise MixedModelBatch(
            f"a batch must target one model, got {sorted(models)}; "
            "submit Stage 1 and Stage 2 as separate batches"
        )
    return models.pop() if models else ""


def write_jsonl(requests: list[dict], path: Path) -> Path:
    assert_single_model(requests)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
    return path


# --- parsing results --------------------------------------------------------


@dataclass
class BatchResult:
    custom_id: str
    text: str
    usage: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def parse_batch_output(lines: list[str]) -> dict[str, BatchResult]:
    """Turn the batch output JSONL into results keyed by custom_id.

    A failed line becomes a BatchResult carrying the error rather than being
    dropped: a frame that silently vanished between stages is worse than one
    that is visibly broken.
    """
    results: dict[str, BatchResult] = {}
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("Unparseable batch output line: %s", e)
            continue

        custom_id = row.get("custom_id", "")
        if row.get("error"):
            results[custom_id] = BatchResult(custom_id, "", error=str(row["error"]))
            continue

        body = (row.get("response") or {}).get("body") or {}
        status = (row.get("response") or {}).get("status_code")
        if status and status >= 400:
            results[custom_id] = BatchResult(custom_id, "", error=f"HTTP {status}")
            continue

        results[custom_id] = BatchResult(
            custom_id=custom_id,
            text=_output_text(body),
            usage=body.get("usage") or {},
        )
    return results


def _output_text(body: dict) -> str:
    """Pull the assistant text out of a Responses body."""
    if isinstance(body.get("output_text"), str):
        return body["output_text"]
    chunks = []
    for item in body.get("output") or []:
        for block in item.get("content") or []:
            if block.get("type") in {"output_text", "text"} and block.get("text"):
                chunks.append(block["text"])
    return "".join(chunks)


def stage1_verdict(text: str) -> bool:
    """True to keep. Anything that is not a clear REJECT keeps the frame.

    Deliberately asymmetric: an unreadable answer must not quietly delete a
    photograph.
    """
    return "reject" not in text.strip().lower()[:20]


def parse_group_json(text: str) -> list[dict]:
    """Extract the JSON array a Stage 2 group should have returned."""
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    if isinstance(data, dict):
        data = data.get("frames") or data.get("results") or [data]
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


AXES = ("axis_a", "axis_b", "axis_c")


def validate_group_ranks(items: list[dict], expected: int) -> list[str]:
    """Check the reply is a strict ranking. Returns the problems found.

    Stage 2 is told each axis must use every rank from 1 to N exactly once. A
    reply that repeats a rank, skips one, or covers only half the frames is not
    a ranking, and feeding it to Bradley-Terry produces a confident ordering out
    of nothing. A group that fails this is re-run or sent to review -- it is
    never silently accepted.
    """
    problems: list[str] = []
    if len(items) != expected:
        problems.append(f"expected {expected} objects, got {len(items)}")

    indices = [item.get("n") for item in items]
    if len(set(map(str, indices))) != len(indices):
        problems.append(f"duplicate or missing 'n': {indices}")

    for axis in AXES:
        values = []
        for item in items:
            try:
                values.append(int(item[axis]))
            except (KeyError, TypeError, ValueError):
                problems.append(f"{axis}: missing or non-numeric on at least one frame")
                break
        else:
            if sorted(values) != list(range(1, len(values) + 1)):
                problems.append(f"{axis}: not a permutation of 1..{len(values)}: {sorted(values)}")
    return problems


def attach_filenames(items: list[dict], group: list[str]) -> list[dict]:
    """Map each returned object back onto a filename via its 1-based `n`.

    Falls back to position when `n` is missing or out of range, and drops
    anything that still cannot be placed -- a mis-indexed object would
    otherwise attach one frame's ranks to another frame's file.
    """
    placed = []
    for position, item in enumerate(items):
        index = item.get("n", position + 1)
        try:
            index = int(index)
        except (TypeError, ValueError):
            index = position + 1
        if not 1 <= index <= len(group):
            index = position + 1
        if not 1 <= index <= len(group):
            continue
        placed.append({**item, "filename": group[index - 1]})
    return placed


# --- token accounting -------------------------------------------------------


class TokenLedger:
    """Actual spend, per stage, gathered from the batch results themselves."""

    def __init__(self) -> None:
        self.stages: dict[str, dict[str, int]] = {}

    def add(self, stage: str, usage: dict) -> None:
        bucket = self.stages.setdefault(
            stage, {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "calls": 0}
        )
        bucket["input_tokens"] += int(usage.get("input_tokens") or 0)
        bucket["output_tokens"] += int(usage.get("output_tokens") or 0)
        details = usage.get("input_tokens_details") or {}
        bucket["cached_tokens"] += int(details.get("cached_tokens") or 0)
        bucket["calls"] += 1

    def totals(self) -> dict[str, int]:
        out = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "calls": 0}
        for bucket in self.stages.values():
            for key in out:
                out[key] += bucket[key]
        return out

    def report(self, frames: int) -> str:
        lines = ["", "TOKEN SPEND (actual, from batch results)", "-" * 62]
        header = f"  {'stage':<18}{'calls':>7}{'input':>12}{'cached':>11}{'output':>10}"
        lines.append(header)
        for stage, b in self.stages.items():
            lines.append(
                f"  {stage:<18}{b['calls']:>7}{b['input_tokens']:>12,}"
                f"{b['cached_tokens']:>11,}{b['output_tokens']:>10,}"
            )
        t = self.totals()
        lines.append("-" * 62)
        lines.append(
            f"  {'total':<18}{t['calls']:>7}{t['input_tokens']:>12,}"
            f"{t['cached_tokens']:>11,}{t['output_tokens']:>10,}"
        )
        if frames:
            lines.append(
                f"  per frame: {t['input_tokens'] / frames:,.0f} in / "
                f"{t['output_tokens'] / frames:,.0f} out  ({frames} frames)"
            )
        if t["input_tokens"]:
            lines.append(f"  cache hit rate: {t['cached_tokens'] / t['input_tokens']:.0%}")
        return "\n".join(lines)


# --- the network-touching part ---------------------------------------------


def submit(client, jsonl_path: Path, state_path: Path | None = None) -> str:
    """Upload and start a batch. Writes the id to disk before returning."""
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=RESPONSES_ENDPOINT,
        completion_window=COMPLETION_WINDOW,
    )
    if state_path:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"batch_id": batch.id, "input_file": uploaded.id}), "utf-8")
    logger.info("Submitted batch %s (%s requests)", batch.id, _count_lines(jsonl_path))
    return batch.id


def wait(client, batch_id: str, poll_seconds: int = POLL_SECONDS, on_poll=None):
    """Block until the batch reaches a terminal state."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if on_poll:
            on_poll(batch)
        if batch.status in TERMINAL_STATES:
            return batch
        time.sleep(poll_seconds)


def fetch(client, batch) -> dict[str, BatchResult]:
    """Download and parse a finished batch's output."""
    if not getattr(batch, "output_file_id", None):
        raise RuntimeError(f"batch {batch.id} finished as {batch.status} with no output")
    content = client.files.content(batch.output_file_id)
    return parse_batch_output(content.text.splitlines())


def _count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
