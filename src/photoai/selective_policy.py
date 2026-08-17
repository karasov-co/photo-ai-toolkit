"""When the tool is allowed to act alone -- and the ten gates before it may.

The target is not "throw away 70%". A tool optimised for that reaches the number
by being wrong. The target is "70% of frames do not need a full human decision",
and most of that comes from **auto-keep**, burst consolidation and routing, none
of which destroys anything.

That reframing is the whole point. Confidently keeping a good frame saves
exactly as much of the photographer's attention as confidently discarding a bad
one, and costs nothing when it is wrong.

Automatic quarantine on a *personal aesthetic model* is gated behind all ten
conditions below, simultaneously, and then a random audit slice is held back on
top of that. Automatic purge is not reachable from here at
any confidence: no model output is grounds for permanent deletion.

On sample size, plainly: demonstrating a false-trash rate below 0.1% with zero
observed errors needs on the order of three thousand independent checks before
the upper confidence bound is low enough to mean anything. A hundred correct
predictions demonstrate nothing, so `enough_evidence` counts rather than guesses.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import StrEnum

from photoai.preference_model import MIN_DECISIONS_TO_DECIDE

logger = logging.getLogger(__name__)

# Rule of three: with zero observed failures in n trials, the 95% upper bound on
# the failure rate is about 3/n. For a 0.1% claim that needs n ~= 3000.
EVIDENCE_FOR_ONE_IN_A_THOUSAND = 3000

AUTO_QUARANTINE_CONFIDENCE = 0.93
MIN_QUARANTINE_AGE_DAYS = 90
AUDIT_SAMPLE_FRACTION = 0.05


class Bucket(StrEnum):
    AUTO_KEEP = "auto_keep"
    AUTO_KEEP_AND_EDIT = "auto_keep_and_edit"
    AUTO_STOCK_CANDIDATE = "auto_stock_candidate"
    AUTO_ARCHIVE = "auto_archive"
    BURST_WINNER = "burst_winner"
    SAFE_QUARANTINE_CANDIDATE = "safe_quarantine_candidate"
    CURATORIAL_REVIEW = "curatorial_review"
    MANUAL_REVIEW = "manual_review"


# Buckets that never touch a file. Everything except one.
NON_ACTING = frozenset(Bucket) - {Bucket.SAFE_QUARANTINE_CANDIDATE}


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class PolicyDecision:
    bucket: str
    abstained: bool = False
    gates: list[Gate] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    audit_sampled: bool = False

    @property
    def acts_on_files(self) -> bool:
        return self.bucket == Bucket.SAFE_QUARANTINE_CANDIDATE.value

    @property
    def failed_gates(self) -> list[str]:
        return [f"{g.name}: {g.detail}" for g in self.gates if not g.passed]


def decide(
    *,
    asset_id: str,
    route_class: str,
    technical_evidence: str,
    prediction,
    model,
    artistic_scores,
    genre: str = "",
    camera: str = "",
    is_best_in_cluster: bool = True,
    cluster_size: int = 1,
    stable_across_runs: bool = True,
    holdout_checks: int = 0,
    shadow_mode: bool = True,
    monitor_healthy: bool = False,
) -> PolicyDecision:
    """Route one asset into a decision bucket, applying every gate."""
    reasons: list[str] = []

    # Circuit A: demonstrable technical failure. Not personalised, not
    # overridable by taste, and the only path that needs no model at all.
    if technical_evidence:
        return PolicyDecision(
            bucket=Bucket.SAFE_QUARANTINE_CANDIDATE.value,
            gates=[Gate("technical_evidence", True, technical_evidence)],
            reasons=[f"demonstrable technical failure: {technical_evidence}"],
        )

    if cluster_size > 1 and is_best_in_cluster:
        reasons.append(f"strongest of {cluster_size} near-identical frames")
        return PolicyDecision(bucket=Bucket.BURST_WINNER.value, reasons=reasons)

    # Circuit C: the curatorial prior. Any artistic signal, or any real
    # uncertainty, and a person looks.
    curator_keeps = artistic_scores.has_any_artistic_signal
    if artistic_scores.curatorial_uncertainty >= 60:
        return PolicyDecision(
            bucket=Bucket.CURATORIAL_REVIEW.value,
            abstained=True,
            reasons=[f"curatorial uncertainty {artistic_scores.curatorial_uncertainty}"],
        )

    if route_class in ("stock_strong", "stock_standard"):
        return PolicyDecision(bucket=Bucket.AUTO_STOCK_CANDIDATE.value,
                              reasons=["clears the stock thresholds"])
    if route_class == "flagship":
        return PolicyDecision(bucket=Bucket.AUTO_KEEP_AND_EDIT.value,
                              reasons=["portfolio candidate; edit recipes generated"])

    # Circuit B: the personal model. Every gate must pass together.
    gates = [
        Gate("enough_decisions", model.can_decide,
             f"{model.decisions} decisions recorded; {MIN_DECISIONS_TO_DECIDE} needed before a "
             "personal model may move a file"),
        Gate("in_distribution", prediction.in_distribution, prediction.reason),
        Gate("model_committed", not prediction.abstained, prediction.reason),
        Gate("holdout_validated", holdout_checks >= EVIDENCE_FOR_ONE_IN_A_THOUSAND,
             f"{holdout_checks} holdout checks; {EVIDENCE_FOR_ONE_IN_A_THOUSAND} needed to "
             "bound the false-trash rate below 0.1%"),
        Gate("models_agree", not _disagree(prediction, curator_keeps),
             "personal model and curatorial prior point opposite ways"),
        Gate("no_artistic_rescue", not curator_keeps,
             "an artistic, emotional or documentary signal is present"),
        Gate("stable", stable_across_runs, "the prediction changed between runs"),
        Gate("confident", (1.0 - prediction.probability) >= AUTO_QUARANTINE_CONFIDENCE,
             f"discard confidence {1.0 - prediction.probability:.2f} below "
             f"{AUTO_QUARANTINE_CONFIDENCE}"),
        Gate("not_shadow_mode", not shadow_mode,
             "shadow mode: recording what would happen, moving nothing"),
        # The tenth. Without it the monitor could observe a rising false-trash
        # rate, switch automation off, and the pipeline would carry on
        # quarantining regardless -- two systems that never spoke to each other.
        Gate("monitor_healthy", monitor_healthy,
             "the monitor has not certified this model: automation is off, drifting, "
             "or has recorded a false trash"),
    ]

    if all(gate.passed for gate in gates):
        sampled = _in_audit_sample(asset_id)
        if sampled:
            return PolicyDecision(
                bucket=Bucket.MANUAL_REVIEW.value, gates=gates, audit_sampled=True,
                reasons=["held back as part of the random audit sample"],
            )
        return PolicyDecision(
            bucket=Bucket.SAFE_QUARANTINE_CANDIDATE.value, gates=gates,
            reasons=["every gate passed"],
        )

    failed = [g for g in gates if not g.passed]
    return PolicyDecision(
        bucket=Bucket.AUTO_ARCHIVE.value if not curator_keeps else Bucket.CURATORIAL_REVIEW.value,
        abstained=True,
        gates=gates,
        reasons=[f"held back: {failed[0].name} ({failed[0].detail})"],
    )


def _disagree(prediction, curator_keeps: bool) -> bool:
    if prediction.abstained:
        return False
    return prediction.keeps != curator_keeps


def _in_audit_sample(asset_id: str) -> bool:
    """A deterministic slice held back for checking, stable across runs.

    Deterministic rather than random so that re-running does not reshuffle which
    frames are audited, which would make the audit unreadable.
    """
    if not asset_id:
        return False
    digest = hashlib.sha256(asset_id.encode()).digest()
    return (digest[0] / 255.0) < AUDIT_SAMPLE_FRACTION


def summarise(decisions: list[PolicyDecision]) -> dict:
    """How much attention the run actually saves, and where it goes."""
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.bucket] = counts.get(decision.bucket, 0) + 1
    total = max(len(decisions), 1)
    needs_human = counts.get(Bucket.MANUAL_REVIEW.value, 0) + counts.get(
        Bucket.CURATORIAL_REVIEW.value, 0
    )
    return {
        "buckets": counts,
        "total": len(decisions),
        "needs_full_human_decision": needs_human,
        "automated_fraction": round(1.0 - needs_human / total, 4),
        "acts_on_files": sum(1 for d in decisions if d.acts_on_files),
        "audit_sampled": sum(1 for d in decisions if d.audit_sampled),
    }
