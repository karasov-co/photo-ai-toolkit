"""Watching the model for the ways it goes wrong quietly.

Three failures matter, and none of them announces itself:

- **False trash.** The metric the whole design is built around. Counted from
  restorations: every file the photographer pulls back out of quarantine is a
  case the tool got wrong, and it is the only ground truth available without a
  labelled set.
- **Drift.** A new camera, a new lens, a trip to a different country, a shift in
  what the photographer is shooting. The model's inputs move and its confidence
  does not, which is the dangerous combination. Detected as a rise in
  out-of-distribution rate, and it *lowers* automation rather than raising an
  alert nobody reads.
- **Miscalibration.** A model that says 0.95 should be right about 95% of the
  time. Measured in bins; if the high-confidence bin is not accurate, the
  confidence gate in `selective_policy` is meaningless.

The response to all three is the same and is automatic: reduce what the tool is
allowed to do by itself. Automation is a privilege the model keeps only while
the evidence supports it.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MONITOR_NAME = "model_monitoring.json"

# A single false trash in a hundred is already far too many for an archive.
FALSE_TRASH_ALARM = 0.001
# Out-of-distribution above this and the workflow has moved on from the model.
DRIFT_ALARM = 0.25
# A confidence bin that is this far from its claim is not a confidence.
CALIBRATION_ALARM = 0.10


@dataclass
class Observation:
    asset_id: str
    predicted: str
    actual: str = ""
    confidence: float = 0.0
    in_distribution: bool = True
    genre: str = ""
    camera: str = ""
    at: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.actual)

    @property
    def correct(self) -> bool:
        return self.resolved and self.predicted == self.actual

    @property
    def is_false_trash(self) -> bool:
        """Predicted destruction, and the photographer disagreed."""
        return self.predicted in ("trash", "safe_quarantine_candidate") and self.actual in (
            "keep", "restored", "portfolio",
        )


@dataclass
class MonitorState:
    observations: list[Observation] = field(default_factory=list)
    automation_enabled: bool = False
    disabled_reason: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "observations": [asdict(o) for o in self.observations],
            "automation_enabled": self.automation_enabled,
            "disabled_reason": self.disabled_reason,
            "updated_at": self.updated_at,
        }


class Monitor:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.state = self._load()

    def _load(self) -> MonitorState:
        if not self.path.exists():
            return MonitorState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Monitoring state unreadable; starting fresh but keeping automation off")
            return MonitorState(automation_enabled=False, disabled_reason="monitoring state lost")
        known = set(Observation.__dataclass_fields__)
        return MonitorState(
            observations=[
                Observation(**{k: v for k, v in row.items() if k in known})
                for row in payload.get("observations") or []
            ],
            automation_enabled=bool(payload.get("automation_enabled", False)),
            disabled_reason=payload.get("disabled_reason", ""),
            updated_at=payload.get("updated_at", ""),
        )

    def save(self) -> Path:
        self.state.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    def observe(self, observation: Observation) -> None:
        observation.at = observation.at or datetime.now(UTC).isoformat(timespec="seconds")
        self.state.observations.append(observation)

    def resolve(self, asset_id: str, actual: str) -> int:
        """Record what the photographer actually did. Returns how many updated."""
        updated = 0
        for observation in self.state.observations:
            if observation.asset_id == asset_id and not observation.resolved:
                observation.actual = actual
                updated += 1
        return updated

    # --- the three metrics --------------------------------------------------

    def false_trash_rate(self) -> tuple[float, int]:
        resolved = [o for o in self.state.observations if o.resolved]
        if not resolved:
            return 0.0, 0
        return round(sum(o.is_false_trash for o in resolved) / len(resolved), 5), len(resolved)

    def drift(self) -> float:
        recent = self.state.observations[-200:]
        if not recent:
            return 0.0
        return round(sum(not o.in_distribution for o in recent) / len(recent), 4)

    def calibration(self) -> dict[str, dict]:
        """Accuracy per confidence bin. A 0.9 bin should be right ~90% of the time."""
        bins: dict[str, list[Observation]] = {}
        for observation in self.state.observations:
            if not observation.resolved:
                continue
            edge = min(0.9, max(0.0, round(observation.confidence * 10) / 10))
            bins.setdefault(f"{edge:.1f}", []).append(observation)
        return {
            key: {
                "claimed": float(key),
                "actual": round(sum(o.correct for o in group) / len(group), 4),
                "n": len(group),
            }
            for key, group in sorted(bins.items())
        }

    def worst_calibration_gap(self) -> float:
        gaps = [
            abs(row["claimed"] - row["actual"])
            for row in self.calibration().values()
            if row["n"] >= 20
        ]
        return round(max(gaps), 4) if gaps else 0.0

    # --- the response -------------------------------------------------------

    def evaluate(self) -> dict:
        """Check the three metrics and switch automation off if any has slipped."""
        rate, resolved = self.false_trash_rate()
        drift = self.drift()
        gap = self.worst_calibration_gap()

        problems: list[str] = []
        if resolved >= 20 and rate > FALSE_TRASH_ALARM:
            problems.append(f"false-trash rate {rate:.3%} over {resolved} resolved cases")
        if drift > DRIFT_ALARM:
            problems.append(f"{drift:.0%} of recent frames are outside the model's experience")
        if gap > CALIBRATION_ALARM:
            problems.append(f"confidence is off by {gap:.2f} in its worst bin")

        if problems:
            self.state.automation_enabled = False
            self.state.disabled_reason = "; ".join(problems)

        return {
            "false_trash_rate": rate,
            "resolved_cases": resolved,
            "drift": drift,
            "worst_calibration_gap": gap,
            "automation_enabled": self.state.automation_enabled,
            "disabled_reason": self.state.disabled_reason,
            "problems": problems,
        }

    def enable_automation(self, *, holdout_checks: int) -> tuple[bool, str]:
        """Turn automation on, only with the evidence to justify it."""
        from selective_policy import EVIDENCE_FOR_ONE_IN_A_THOUSAND

        if holdout_checks < EVIDENCE_FOR_ONE_IN_A_THOUSAND:
            return False, (
                f"{holdout_checks} holdout checks; {EVIDENCE_FOR_ONE_IN_A_THOUSAND} are needed "
                "before a false-trash rate below 0.1% can be claimed at all"
            )
        report = self.evaluate()
        if report["problems"]:
            return False, "; ".join(report["problems"])
        self.state.automation_enabled = True
        self.state.disabled_reason = ""
        return True, "automation enabled"
