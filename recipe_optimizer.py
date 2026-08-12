"""Render the candidates, throw out the damaging ones, keep what is left.

The important decision here is **not collapsing the criteria into one score**.
Recovered detail, absence of new clipping, preserved intent and restraint are
not commensurable: a weighted sum lets a candidate that badly violates one
criterion win by excelling at another, which is exactly how an automatic edit
ends up flattening a low-key frame because it "improved contrast a lot".

So candidates are compared by Pareto dominance. A candidate is dropped only when
another is at least as good on *every* criterion and better on one. What
survives is the non-dominated set -- typically one to three genuinely different
readings of the frame, which is the right thing to put in front of a person.

Validation runs before ranking, not after, and its verdict is a veto rather than
a penalty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

import recipe_validator
from renderers import base as renderer_base

logger = logging.getLogger(__name__)

PREVIEW_PX = 900

CRITERIA = ("recovered_detail", "clipping_safety", "intent_preserved", "restraint")


@dataclass
class Candidate:
    recipe: object
    image: Image.Image | None = None
    scores: dict[str, float] = field(default_factory=dict)
    validation: recipe_validator.ValidationResult | None = None
    rejected_for: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.rejected_for

    @property
    def variant(self) -> str:
        return getattr(self.recipe, "variant", "?")


def evaluate(
    path: Path,
    recipes: list,
    *,
    renderer=None,
    max_px: int = PREVIEW_PX,
) -> tuple[list[Candidate], Image.Image | None]:
    """Render every candidate, validate it, and score the survivors."""
    engine = renderer or renderer_base.get()
    from edit_schema import EditRecipe

    try:
        original = engine.render(path, EditRecipe(), max_px=max_px)
    except renderer_base.RenderError as e:
        logger.warning("Could not render the original of %s: %s", Path(path).name, e)
        return [], None

    candidates: list[Candidate] = []
    for recipe in recipes:
        candidate = Candidate(recipe=recipe)
        try:
            candidate.image = engine.render(path, recipe, max_px=max_px)
        except renderer_base.RenderError as e:
            candidate.rejected_for = [f"render failed: {e}"]
            candidates.append(candidate)
            continue

        candidate.validation = recipe_validator.validate(original, candidate.image, recipe)
        if not candidate.validation.ok:
            candidate.rejected_for = candidate.validation.reasons
        else:
            candidate.scores = _score(original, candidate.image, recipe)
        recipe.engine = engine.name
        recipe.engine_version = engine.version()
        candidates.append(candidate)

    return candidates, original


def _score(original: Image.Image, edited: Image.Image, recipe) -> dict[str, float]:
    """Four independent readings. Never summed."""
    before = np.asarray(original.convert("RGB"), dtype=np.float64) / 255.0
    after = np.asarray(
        edited.resize(original.size, Image.LANCZOS) if edited.size != original.size else edited
    ).astype(np.float64) / 255.0

    weights = np.array([0.299, 0.587, 0.114])
    luma_before, luma_after = before @ weights, after @ weights

    # Detail that became visible: mass moved out of the crushed and blown ends.
    hidden_before = float(((luma_before < 0.03) | (luma_before > 0.97)).mean())
    hidden_after = float(((luma_after < 0.03) | (luma_after > 0.97)).mean())
    recovered = max(0.0, hidden_before - hidden_after)

    clipped_after = float(((luma_after <= 0.004) | (luma_after >= 0.996)).mean())
    clipping_safety = max(0.0, 1.0 - clipped_after * 8.0)

    intent = _intent_preserved(luma_before, luma_after, recipe)

    # Restraint: how far the edit travelled. Ties break towards the smaller move,
    # because a smaller move is easier for a person to disagree with later.
    g = recipe.global_adjustments
    travel = (
        abs(g.exposure_ev) / 2.0
        + (abs(g.contrast) + abs(g.highlights) + abs(g.shadows) + abs(g.clarity)) / 400.0
        + recipe.detail.sharpening / 200.0
        + recipe.detail.denoise_luminance / 200.0
    )
    restraint = max(0.0, 1.0 - travel)

    return {
        "recovered_detail": round(recovered, 5),
        "clipping_safety": round(clipping_safety, 5),
        "intent_preserved": round(intent, 5),
        "restraint": round(restraint, 5),
    }


def _intent_preserved(luma_before: np.ndarray, luma_after: np.ndarray, recipe) -> float:
    """How much of what the recipe promised to protect is still there."""
    protected = [p.lower() for p in (recipe.preserve or [])]
    if not protected:
        return 1.0

    kept = []
    if any("low-key" in p or "shadow" in p for p in protected):
        before_mass = float((luma_before < 0.25).mean())
        after_mass = float((luma_after < 0.25).mean())
        kept.append(min(1.0, after_mass / before_mass) if before_mass > 0.01 else 1.0)
    if any("grain" in p for p in protected):
        # Grain surviving is texture surviving: compare high-frequency energy.
        def detail_energy(a: np.ndarray) -> float:
            return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())

        before_energy = detail_energy(luma_before)
        kept.append(
            min(1.0, detail_energy(luma_after) / before_energy) if before_energy > 1e-6 else 1.0
        )
    if any("blur" in p or "motion" in p for p in protected):
        kept.append(0.0 if recipe.detail.sharpening > 0 else 1.0)
    if any("tilt" in p for p in protected):
        kept.append(0.0 if recipe.geometry.rotation_deg else 1.0)

    return round(sum(kept) / len(kept), 5) if kept else 1.0


# --- Pareto -------------------------------------------------------------------


def dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """`a` is at least as good everywhere and better somewhere."""
    at_least = all(a.get(k, 0.0) >= b.get(k, 0.0) - 1e-9 for k in CRITERIA)
    strictly = any(a.get(k, 0.0) > b.get(k, 0.0) + 1e-9 for k in CRITERIA)
    return at_least and strictly


def non_dominated(candidates: list[Candidate]) -> list[Candidate]:
    """The survivors. Usually one to three genuinely different readings."""
    usable = [c for c in candidates if c.ok and c.scores]
    return [
        candidate
        for candidate in usable
        if not any(dominates(other.scores, candidate.scores) for other in usable if other is not candidate)
    ]


def choose(candidates: list[Candidate], *, limit: int = 3) -> list[Candidate]:
    """Non-dominated, capped, faithful first so the safest option leads."""
    order = {"faithful": 0, "expressive": 1, "monochrome": 2}
    survivors = non_dominated(candidates)
    survivors.sort(key=lambda c: order.get(c.variant, 9))
    return survivors[:limit]


def write_variants(
    candidates: list[Candidate],
    original: Image.Image | None,
    out_dir: Path,
    asset_id: str,
) -> dict[str, str]:
    """A/B previews on disk. People choose between pictures, not between numbers."""
    written: dict[str, str] = {}
    folder = Path(out_dir) / "suggestions" / asset_id
    folder.mkdir(parents=True, exist_ok=True)

    if original is not None:
        path = folder / "original.jpg"
        original.save(path, "JPEG", quality=88)
        written["original"] = str(path)

    for candidate in candidates:
        if candidate.image is None:
            continue
        path = folder / f"{candidate.variant}.jpg"
        candidate.image.save(path, "JPEG", quality=88)
        written[candidate.variant] = str(path)
    return written
