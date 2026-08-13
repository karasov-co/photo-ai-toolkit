"""Which platforms an asset could go to, and what would stop it.

Every rule lives in `data/marketplace_rules.json` rather than in this file. That
separation is the whole point: marketplace requirements change on the
platforms' schedule, not on ours, and a resolution floor or an AI policy should
be a one-line data edit that a non-programmer can make and a reviewer can diff
-- not a constant buried in a scoring function. Each platform entry carries the
URL it came from and the date it was last checked, so a stale rule is visible
rather than merely wrong.

What this module will not do is promise anything. It reports *eligibility* --
the asset meets the published technical and policy requirements -- and a
suitability score for ordering the recommendations. Acceptance is a human
reviewer's decision at every one of these platforms, and sales are a market
outcome. Both are outside what any local analysis can know.

The commercial/editorial split is not advisory here either. An asset with a face
or a readable trademark and no release is editorial-only, full stop, and the
platform recommendation reflects that rather than suggesting the user "consider"
getting a release.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from provenance import ProvenanceRecord, conflicts_for
from scoring import Route, ScoredAsset, Semantic

logger = logging.getLogger(__name__)

RULES_PATH = Path(__file__).parent / "data" / "marketplace_rules.json"


@dataclass
class Ruleset:
    schema_version: int = 1
    ruleset_version: str = "unknown"
    disclaimer: str = ""
    platforms: list[dict] = field(default_factory=list)

    def by_id(self, platform_id: str) -> dict | None:
        return next((p for p in self.platforms if p.get("id") == platform_id), None)

    def accepting(self, kind: str) -> list[dict]:
        return [p for p in self.platforms if kind in (p.get("accepts") or [])]


@lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> Ruleset:
    """Read the rules file. Cached, because it is read once per asset otherwise."""
    rules_path = Path(path) if path else RULES_PATH
    try:
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Could not load marketplace rules from %s: %s", rules_path, e)
        return Ruleset()
    return Ruleset(
        schema_version=int(payload.get("schema_version", 1)),
        ruleset_version=str(payload.get("ruleset_version", "unknown")),
        disclaimer=str(payload.get("disclaimer", "")),
        platforms=list(payload.get("platforms") or []),
    )


@dataclass
class TechnicalFacts:
    """What a platform rule needs to know about the file itself."""

    kind: str = "photo"
    megapixels: float = 0.0
    width: int = 0
    height: int = 0
    duration: float = 0.0
    container: str = ""
    file_format: str = "JPEG"


@dataclass
class Recommendation:
    platform_id: str
    platform_name: str
    eligible: bool
    suitability: int
    route: str
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    missing_releases: list[str] = field(default_factory=list)
    missing_metadata: list[str] = field(default_factory=list)
    policy_conflicts: list[str] = field(default_factory=list)
    manual_submission_required: bool = False
    export_ready: bool = False
    max_keywords: int = 50
    verified_on: str = ""

    def to_dict(self) -> dict:
        return {
            "platform_id": self.platform_id,
            "platform": self.platform_name,
            "eligible": self.eligible,
            "suitability": self.suitability,
            "route": self.route,
            "reasons": self.reasons,
            "blockers": self.blockers,
            "missing_releases": self.missing_releases,
            "missing_metadata": self.missing_metadata,
            "policy_conflicts": self.policy_conflicts,
            "manual_submission_required": self.manual_submission_required,
            "export_ready": self.export_ready,
            "rules_verified_on": self.verified_on,
        }


def check_technical(platform: dict, facts: TechnicalFacts) -> list[str]:
    """Hard technical requirements. Returns the reasons it fails, if any."""
    spec = platform.get(facts.kind) or {}
    blockers: list[str] = []

    if facts.kind not in (platform.get("accepts") or []):
        return [f"{platform.get('name')} does not accept {facts.kind} content"]

    if facts.kind == "photo":
        floor = float(spec.get("min_megapixels", 0) or 0)
        if floor and facts.megapixels < floor:
            blockers.append(f"{facts.megapixels} MP is below the {floor} MP minimum")
        ceiling = float(spec.get("max_megapixels", 0) or 0)
        if ceiling and facts.megapixels > ceiling:
            blockers.append(f"{facts.megapixels} MP exceeds the {ceiling} MP maximum")
        formats = spec.get("formats") or []
        if formats and facts.file_format.upper() not in {f.upper() for f in formats}:
            blockers.append(
                f"{facts.file_format} is not accepted; export to {'/'.join(formats)} first"
            )
    else:
        min_d = float(spec.get("min_duration_s", 0) or 0)
        max_d = float(spec.get("max_duration_s", 0) or 0)
        if min_d and facts.duration < min_d:
            blockers.append(f"{facts.duration:.1f}s is below the {min_d:.0f}s minimum")
        if max_d and facts.duration > max_d:
            blockers.append(f"{facts.duration:.1f}s exceeds the {max_d:.0f}s maximum; trim first")
        min_w = int(spec.get("min_width", 0) or 0)
        min_h = int(spec.get("min_height", 0) or 0)
        if min_w and min_h:
            fits = facts.width >= min_w and facts.height >= min_h
            # Vertical delivery swaps the two, and a platform that publishes a
            # 1920x1080 floor accepts 1080x1920 as well.
            fits_rotated = facts.width >= min_h and facts.height >= min_w
            if not fits and not fits_rotated:
                blockers.append(f"{facts.width}x{facts.height} is below {min_w}x{min_h}")
        containers = spec.get("containers") or []
        if containers and facts.container and not _container_matches(facts.container, containers):
            blockers.append(f"{facts.container} is not an accepted container")
    return blockers


def _container_matches(container: str, accepted: list[str]) -> bool:
    """ffprobe reports comma-joined format names such as 'mov,mp4,m4a,3gp'."""
    reported = {part.strip().lower() for part in container.split(",")}
    return bool(reported & {a.lower() for a in accepted})


def evaluate(
    asset: ScoredAsset,
    facts: TechnicalFacts,
    semantic: Semantic,
    record: ProvenanceRecord,
    *,
    rules: Ruleset | None = None,
    has_model_release: bool = False,
    has_property_release: bool = False,
    metadata_complete: bool = False,
) -> list[Recommendation]:
    """Rank the platforms this asset could realistically go to.

    Sorted by suitability, ineligible platforms last. An ineligible platform is
    kept in the list rather than dropped, because "Shutterstock, blocked because
    provenance is undeclared" is more useful to a contributor than silence.
    """
    rules = rules or load_rules()
    recommendations: list[Recommendation] = []
    conflicts = {c.platform: c for c in conflicts_for(record, rules.platforms)}

    for platform in rules.platforms:
        blockers = check_technical(platform, facts)
        reasons: list[str] = []
        missing_releases: list[str] = []
        missing_metadata: list[str] = []
        policy_notes: list[str] = []

        route = asset.route
        if route is Route.EDITORIAL and not platform.get("editorial_accepted", True):
            blockers.append(f"{platform.get('name')} has no editorial route for this asset")

        if route is Route.COMMERCIAL:
            reasons.append("no release needed: commercial licensing is open")
        else:
            reasons.append(
                "editorial only: a face or trademark is present and commercial stock is blocked"
            )
            if (semantic.faces or semantic.identifiable_people) and not has_model_release:
                missing_releases.append("model release")
            if semantic.recognizable_property and not has_property_release:
                missing_releases.append("property release")
            if semantic.brand_mark:
                policy_notes.append(
                    "readable trademark present: editorial use only, and some buyers will still refuse it"
                )

        conflict = conflicts.get(platform.get("id", ""))
        if conflict:
            policy_notes.append(conflict.message)
            if conflict.severity == "blocking":
                blockers.append(conflict.message)

        if not metadata_complete:
            missing_metadata.append("title, description and ordered keywords")

        suitability = _suitability(asset, platform, facts, route)
        eligible = not blockers

        if platform.get("editorial_strength") == "high" and route is Route.EDITORIAL:
            suitability = min(100, suitability + 12)
            reasons.append("strong editorial market")

        recommendations.append(
            Recommendation(
                platform_id=str(platform.get("id", "")),
                platform_name=str(platform.get("name", "")),
                eligible=eligible,
                suitability=suitability if eligible else 0,
                route=route.value,
                reasons=reasons,
                blockers=blockers,
                missing_releases=missing_releases,
                missing_metadata=missing_metadata,
                policy_conflicts=policy_notes,
                manual_submission_required=bool(platform.get("manual_submission_required", False)),
                export_ready=eligible and not missing_releases and metadata_complete,
                max_keywords=int(platform.get("max_keywords", 50)),
                verified_on=str(platform.get("verified_on", "")),
            )
        )

    return sorted(recommendations, key=lambda r: (-int(r.eligible), -r.suitability, r.platform_name))


def _suitability(asset: ScoredAsset, platform: dict, facts: TechnicalFacts, route: Route) -> int:
    """How well this asset fits this platform, 0-100. Ordering only."""
    base = asset.scores.stock_potential
    if route is Route.EDITORIAL:
        base = int(0.55 * asset.scores.stock_potential + 0.45 * asset.scores.aesthetic_potential)

    spec = platform.get(facts.kind) or {}
    if facts.kind == "photo":
        floor = float(spec.get("min_megapixels", 4.0) or 4.0)
        if facts.megapixels >= floor * 3:
            base += 6
    else:
        recommended = spec.get("recommended_duration_s") or []
        if len(recommended) == 2 and recommended[0] <= facts.duration <= recommended[1]:
            base += 8

    if platform.get("exclusive_by_default"):
        # Exclusivity is a real cost, not a technicality: it removes the asset
        # from every other platform in this list.
        base -= 10
    return max(0, min(100, base))


def summarise(recommendations: list[Recommendation]) -> dict:
    eligible = [r for r in recommendations if r.eligible]
    return {
        "recommended": [r.platform_name for r in eligible[:3]],
        "eligible_count": len(eligible),
        "export_ready_count": sum(1 for r in eligible if r.export_ready),
        "blocked": {r.platform_name: r.blockers for r in recommendations if not r.eligible},
    }
