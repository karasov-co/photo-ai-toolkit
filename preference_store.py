"""What the photographer actually decided, and how much each decision is worth.

Not a `kept/deleted` log. A binary classifier trained on that learns to
reproduce the owner's current blind spots and gets more confident about them
every month -- somebody who discards unusual frames because they have not yet
seen what is in them ends up with a tool that hides unusual frames.

Two design choices follow from that:

- **Comparisons, not ratings.** "A is stronger than B" is a stable judgement;
  "this is a 7/10" is not, and the same photograph gets a different number on a
  different day. Pairwise data also composes: it feeds Bradley-Terry directly.
- **Signals are weighted by what they actually prove.** Rescuing a file from
  quarantine is a loud statement that the tool was wrong. Not opening a folder
  for three months says nothing at all. Treating them alike is how a preference
  model learns noise.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)

STORE_NAME = "preferences.jsonl"
SCHEMA_VERSION = 1


class Signal(StrEnum):
    PAIRWISE_KEEP = "pairwise_keep"
    PAIRWISE_EMOTION = "pairwise_emotion"
    PAIRWISE_SERIES = "pairwise_series"
    PAIRWISE_STOCK = "pairwise_stock"
    BURST_WINNER = "burst_winner"
    RESTORED_FROM_QUARANTINE = "restored_from_quarantine"
    ADDED_TO_PORTFOLIO = "added_to_portfolio"
    PUBLISHED = "published"
    PRINTED = "printed"
    MANUAL_KEEP = "manual_keep"
    MANUAL_REJECT = "manual_reject"
    QUICK_REJECT = "quick_reject"
    INTENT_LABEL = "intent_label"
    VARIANT_CHOICE = "variant_choice"
    CROP_ADJUSTED = "crop_adjusted"
    IMPORTED_EDIT = "imported_edit"
    NOT_OPENED = "not_opened"


# How much each signal is allowed to move the model. Justified individually
# because "the user did something" is not one kind of evidence.
SIGNAL_WEIGHTS: dict[Signal, float] = {
    # The tool was demonstrably wrong and the user went to the trouble of
    # undoing it. Nothing else is this informative.
    Signal.RESTORED_FROM_QUARANTINE: 3.0,
    Signal.ADDED_TO_PORTFOLIO: 2.5,
    Signal.PUBLISHED: 2.0,
    Signal.PRINTED: 2.0,
    # A considered comparison, made while looking at both frames.
    Signal.PAIRWISE_KEEP: 1.5,
    Signal.PAIRWISE_EMOTION: 1.5,
    Signal.PAIRWISE_SERIES: 1.2,
    Signal.PAIRWISE_STOCK: 1.2,
    Signal.BURST_WINNER: 1.4,
    Signal.VARIANT_CHOICE: 1.0,
    Signal.INTENT_LABEL: 1.5,
    Signal.CROP_ADJUSTED: 0.8,
    Signal.IMPORTED_EDIT: 1.0,
    Signal.MANUAL_KEEP: 1.0,
    Signal.MANUAL_REJECT: 1.0,
    # Made in seconds on a small screen, before the frame was really looked at.
    Signal.QUICK_REJECT: 0.4,
    # Absence of action is not a judgement. Kept at zero so it can be recorded
    # for analysis without ever training anything.
    Signal.NOT_OPENED: 0.0,
}


@dataclass
class Decision:
    """One thing the photographer did, with everything needed to learn from it."""

    signal: str
    winner: str = ""          # asset_id
    loser: str = ""           # asset_id, for pairwise signals
    asset_id: str = ""        # for non-pairwise signals
    question: str = ""
    answer: str = ""
    confidence: float = 1.0
    genre: str = ""
    camera: str = ""
    # What the tool believed at the moment of disagreement -- the other half of
    # a training pair, and what makes a wrong call auditable later.
    tool_said: str = ""
    tool_scores: dict = field(default_factory=dict)
    recorded_at: str = ""
    note: str = ""

    @property
    def weight(self) -> float:
        try:
            return SIGNAL_WEIGHTS[Signal(self.signal)] * max(0.0, min(1.0, self.confidence))
        except ValueError:
            return 0.0

    @property
    def is_pairwise(self) -> bool:
        return bool(self.winner and self.loser)

    def to_dict(self) -> dict:
        return asdict(self)


class PreferenceStore:
    """Append-only JSONL. Decisions are history and are never rewritten."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def record(self, decision: Decision) -> Decision:
        decision.recorded_at = decision.recorded_at or datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"schema_version": SCHEMA_VERSION, **decision.to_dict()},
                               ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return decision

    def all(self) -> list[Decision]:
        if not self.path.exists():
            return []
        out: list[Decision] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            payload.pop("schema_version", None)
            known = set(Decision.__dataclass_fields__)
            out.append(Decision(**{k: v for k, v in payload.items() if k in known}))
        return out

    def pairs(self) -> list[Decision]:
        return [d for d in self.all() if d.is_pairwise and d.weight > 0]

    def corrections(self) -> list[Decision]:
        """Where the user overturned the tool. The most valuable subset."""
        return [
            d for d in self.all()
            if d.signal in (Signal.RESTORED_FROM_QUARANTINE.value, Signal.ADDED_TO_PORTFOLIO.value)
            or (d.tool_said and d.answer and d.tool_said != d.answer)
        ]

    def count(self) -> int:
        return len(self.all())

    def genres_seen(self) -> set[str]:
        return {d.genre for d in self.all() if d.genre}

    def cameras_seen(self) -> set[str]:
        return {d.camera for d in self.all() if d.camera}
