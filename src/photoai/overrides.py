"""Decisions the user made by hand, which the tool is not allowed to undo.

A culling tool earns trust slowly and loses it in one incident. The incident is
always the same shape: the user looks at a frame the tool called trash, disagrees,
marks it as a keeper, re-runs the analysis for an unrelated reason, and the mark
is gone. After that they stop trusting any of the output, correctly.

So overrides are stored separately from analysis results, keyed by content
checksum rather than by filename, and applied *after* classification on every
run. Analysis can be repeated, thresholds retuned, the analyzer version bumped
-- none of it touches this file.

Keying on the checksum rather than the path is what makes the guarantee hold
through a rename or a reorganisation. It also means an *edited* file is
deliberately a different asset: the override applied to the frame the user
judged, not to whatever later took its filename.

The store doubles as the training set for future personalised calibration. Each
entry records what the tool said as well as what the user said, which is the
pair a model would need; nothing here learns from it yet, and the schema is
shaped so that adding it later is not a migration.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

OVERRIDES_NAME = "manual_overrides.json"
SCHEMA_VERSION = 1


@dataclass
class Override:
    asset_id: str
    checksum: str = ""
    filename: str = ""
    route_class: str | None = None
    genre: str | None = None
    marketplaces: list[str] = field(default_factory=list)
    excluded: bool = False
    note: str = ""
    # What the tool said at the moment the user disagreed. Kept for calibration.
    tool_said: str = ""
    tool_scores: dict = field(default_factory=dict)
    decided_at: str = ""
    decided_by: str = "user"

    def to_dict(self) -> dict:
        return asdict(self)


class OverrideStore:
    """A small JSON document, written atomically, read on every run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: dict[str, Override] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            # Never start from empty on a parse failure: that would silently
            # discard every manual decision the user has ever made.
            salvage = self.path.with_name(self.path.name + ".corrupt")
            logger.error(
                "Could not read %s (%s); preserving it as %s and refusing to overwrite",
                self.path.name, e, salvage.name,
            )
            with contextlib.suppress(OSError):
                os.replace(self.path, salvage)
            return
        for row in payload.get("overrides") or []:
            try:
                override = Override(**row)
            except TypeError:
                continue
            self._entries[override.asset_id] = override

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "overrides": [o.to_dict() for o in self._entries.values()],
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        return self.path

    def set(self, override: Override) -> Override:
        override.decided_at = override.decided_at or datetime.now(UTC).isoformat(timespec="seconds")
        self._entries[override.asset_id] = override
        return override

    def get(self, asset_id: str) -> Override | None:
        return self._entries.get(asset_id)

    def remove(self, asset_id: str) -> bool:
        return self._entries.pop(asset_id, None) is not None

    def all(self) -> list[Override]:
        return list(self._entries.values())

    def __len__(self) -> int:
        return len(self._entries)


def apply_to(records: list, store: OverrideStore) -> int:
    """Overlay manual decisions onto a fresh run. Returns how many applied.

    The tool's own conclusion is not erased -- it moves to `reasons` so the
    disagreement stays visible, which is what makes the override auditable
    later and usable as calibration data.
    """
    applied = 0
    for record in records:
        override = store.get(record.asset_id)
        if override is None:
            continue

        if override.excluded:
            record.proposed_action = "excluded_by_user"
            record.reasons = ["excluded from analysis by the user", *record.reasons]
            applied += 1
            continue

        if override.route_class and override.route_class != record.route_class:
            record.reasons = [
                f"manual override: user set {override.route_class}; "
                f"the tool had said {record.route_class}"
                + (f" ({override.note})" if override.note else ""),
                *record.reasons,
            ]
            record.route_class = override.route_class
            record.proposed_action = _action_for(override.route_class)
            applied += 1

        if override.genre:
            record.genre = override.genre
        if override.marketplaces:
            record.stock_metadata = {
                **(record.stock_metadata or {}),
                "suggested_marketplaces": override.marketplaces,
            }
        touched = override.route_class or override.genre or override.marketplaces
        if touched and "manual_override" not in record.tags:
            record.tags = [*record.tags, "manual_override"]
    return applied


def resolve_observations(records: list, store: OverrideStore, monitor_path) -> int:
    """Tell the monitor what the photographer actually decided.

    Without this the monitor is a command whose state somebody has to fill in by
    hand, and the false-trash rate it reports is the rate over an empty set --
    which is 0% and means nothing. An override *is* the ground truth: it is the
    photographer contradicting the tool in writing.
    """
    from photoai.model_monitoring import Monitor

    monitor = Monitor(monitor_path)
    resolved = 0
    for record in records:
        override = store.get(record.asset_id)
        if override is None:
            continue
        if override.excluded:
            continue
        actual = "keep" if override.route_class not in ("trash", None, "") else "trash"
        if override.route_class == "flagship":
            actual = "portfolio"
        resolved += monitor.resolve(record.asset_id, actual)
    if resolved:
        monitor.evaluate()
        monitor.save()
    return resolved


def _action_for(route_class: str) -> str:
    if route_class == "trash":
        return "quarantine"
    if route_class == "review":
        return "hold_for_review"
    return "keep_in_place"


def capture(record) -> Override:
    """Seed an override from a record, remembering what the tool concluded."""
    return Override(
        asset_id=record.asset_id,
        checksum=record.checksum,
        filename=record.filename,
        tool_said=record.route_class,
        tool_scores=dict(record.scores or {}),
    )
