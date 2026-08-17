"""Measure whether the quality metric agrees with a person.

`frame_quality` is used twice, and that is the problem this module exists to
expose. It is the objective the preview search hill-climbs on, *and* it is the
ruler used to report the result. So `uplift` -- the headline "editing gains you
12 points" figure -- measures how far the search moved a number it was
optimising, which is not the same claim as "the photograph got better". A metric
scoring its own output will always report progress.

That circularity cannot be argued away from inside. It needs an outside
reference: photographs a person has ranked. This command takes that ranking and
reports the correlation, and until somebody runs it the tool says in the report
that the figure is unvalidated rather than quietly presenting it as measured.

The CSV wants one row per photograph:

    filename,human_score        (higher is better, any scale)
    filename,human_rank         (1 = best)

and optionally, for the uplift half:

    filename,human_score,human_score_edited

Spearman rather than Pearson throughout: the claim being tested is "does this
rank photographs the way a person does", not "is the relationship linear".
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Below this, the metric is not measuring what a person means by quality. Not a
# pass mark anybody has agreed to -- a line for the report to speak plainly
# against, chosen where rank correlation stops being arguable.
USEFUL_CORRELATION = 0.5

# Fewer than this and the correlation is noise whatever it says.
MIN_SAMPLES = 20

# The buckets a person is asked to reproduce, and why there are three rather
# than five. `top` and `weak` are quality boundaries -- 85 and 45 -- and those
# are the two numbers the product's central promise rests on. The stock/personal
# split is not a quality judgement at all; it is about whether a frame has a
# market, and asking somebody to guess that conflates two different questions
# and makes the agreement figure meaningless.
HUMAN_PILES = ("top", "good", "weak")

# What the tool's own five collapse to, for comparison.
PILE_OF_CATEGORY = {
    "TOP": "top",
    "GOOD_STOCK": "good",
    "GOOD_PERSONAL": "good",
    "NEEDS_DECISION": "good",
    "WEAK": "weak",
}

# Below this, the thresholds do not reproduce a person's sorting well enough to
# be described as calibrated. Chosen, not derived: three piles guessed at random
# would agree about a third of the time, and 0.70 is where the result stops
# being arguable.
USEFUL_AGREEMENT = 0.70


@dataclass
class Correlation:
    rho: float = 0.0
    n: int = 0
    note: str = ""

    @property
    def meaningful(self) -> bool:
        return self.n >= MIN_SAMPLES

    def to_dict(self) -> dict:
        return {"spearman": round(self.rho, 4), "n": self.n, "note": self.note}


@dataclass
class PileAgreement:
    """How often the thresholds put a photograph where a person put it."""

    n: int = 0
    agreed: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    top_precision: float = 0.0
    top_recall: float = 0.0
    # The (top, weak) pair that would have agreed most often, and by how much.
    best_top: int = 0
    best_weak: int = 0
    best_agreement: float = 0.0
    current_top: int = 0
    current_weak: int = 0

    @property
    def meaningful(self) -> bool:
        return self.n >= MIN_SAMPLES

    @property
    def agreement(self) -> float:
        return self.agreed / self.n if self.n else 0.0

    @property
    def calibrated(self) -> bool:
        return self.meaningful and self.agreement >= USEFUL_AGREEMENT

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "agreement": round(self.agreement, 4),
            "confusion": self.confusion,
            "top_precision": round(self.top_precision, 4),
            "top_recall": round(self.top_recall, 4),
            "current_thresholds": {"top": self.current_top, "weak": self.current_weak},
            "best_thresholds": {"top": self.best_top, "weak": self.best_weak},
            "best_agreement": round(self.best_agreement, 4),
            "calibrated": self.calibrated,
        }


@dataclass
class BenchResult:
    quality: Correlation = field(default_factory=Correlation)
    uplift: Correlation = field(default_factory=Correlation)
    piles: PileAgreement = field(default_factory=PileAgreement)
    missing: list[str] = field(default_factory=list)
    labelled: int = 0

    @property
    def validates_quality(self) -> bool:
        return self.quality.meaningful and self.quality.rho >= USEFUL_CORRELATION

    @property
    def validates_uplift(self) -> bool:
        return self.uplift.meaningful and self.uplift.rho >= USEFUL_CORRELATION

    def to_dict(self) -> dict:
        return {
            "labelled": self.labelled,
            "quality": self.quality.to_dict(),
            "uplift": self.uplift.to_dict(),
            "piles": self.piles.to_dict(),
            "missing": self.missing,
            "validates_quality": self.validates_quality,
            "validates_uplift": self.validates_uplift,
            "threshold": USEFUL_CORRELATION,
        }


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation, ties averaged. Plain arithmetic, no scipy."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(rx)
    mean_x, mean_y = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def read_labels(path: Path) -> dict[str, dict]:
    """Every CSV in a directory, or one file. Keyed by filename.

    A rank is inverted into a score so both label styles end up on one scale
    where higher is better -- otherwise a run against rank data reports a
    correlation with the sign flipped and reads as a metric that is exactly
    wrong rather than one that agrees.
    """
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    out: dict[str, dict] = {}
    for file in files:
        with open(file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = (row.get("filename") or "").strip()
                if not name:
                    continue
                entry: dict = {}
                if row.get("human_score"):
                    entry["score"] = _number(row["human_score"])
                elif row.get("human_rank"):
                    rank = _number(row["human_rank"])
                    entry["score"] = -rank if rank is not None else None
                if row.get("human_score_edited"):
                    entry["edited"] = _number(row["human_score_edited"])
                pile = (row.get("human_pile") or "").strip().lower()
                if pile in HUMAN_PILES:
                    entry["pile"] = pile
                # A row with only a pile is still a row. Sorting 300 frames into
                # three piles is an evening; scoring 300 frames 0-100 is not,
                # and demanding both is how a labelling task never gets done.
                if entry.get("score") is not None or entry.get("pile"):
                    out[name] = entry
    return out


def _number(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


# The photographer has already done the labelling; it is sitting in their
# catalogue. A week of working in Lightroom leaves stars and picks in the
# sidecars beside the originals, and reading them back turns every user into a
# labeller who spent no time on it. It is also the only way the personal model
# and the monitor ever see the thousands of decisions their gates ask for.
# Where a photographer's stars sit is a personal convention. For many people one
# star is "bin it" and two is already "kept", which is exactly where the
# good/weak line falls -- so getting this wrong shifts the whole measurement.
# Override with PHOTO_AI_STARS, as `5=top,4=top,3=good,2=weak,1=weak`.
DEFAULT_PILE_BY_STARS = {5: "top", 4: "top", 3: "good", 2: "good", 1: "weak"}


def pile_by_stars() -> dict[int, str]:
    import os

    mapping = dict(DEFAULT_PILE_BY_STARS)
    for pair in os.environ.get("PHOTO_AI_STARS", "").split(","):
        stars, _, pile = pair.partition("=")
        stars, pile = stars.strip(), pile.strip().lower()
        if stars.isdigit() and pile in HUMAN_PILES:
            mapping[int(stars)] = pile
    return mapping


def read_catalog_labels(folder: Path, *, written_before: float | None = None) -> dict[str, dict]:
    """Human piles from `xmp:Rating` in the sidecars beside a shoot.

    Zero and unrated are skipped, not read as "weak": in every catalogue zero
    means nobody looked, and counting that as a judgement would fill the set
    with frames the photographer never opened.

    `written_before` is the guard against the loop closing on itself. A
    photographer edits the frames the tool put in the top pile, and edited
    frames get five stars, and the tool then reports that it agreed with them --
    a measurement of its own influence. Sidecars touched after the run they are
    being compared against are dropped, and the caller says how many.
    """
    import re

    out: dict[str, dict] = {}
    mapping = pile_by_stars()
    for path in sorted(Path(folder).rglob("*.xmp")):
        if written_before is not None and path.stat().st_mtime > written_before:
            _CONTAMINATED.append(path.name)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        found = re.search(r'xmp:Rating\s*=\s*"(\d+)"', text)
        if not found:
            found = re.search(r"<xmp:Rating>\s*(\d+)\s*</xmp:Rating>", text)
        if not found:
            continue
        stars = int(found.group(1))
        pile = mapping.get(stars)
        if not pile:
            continue
        # The sidecar is named after the photograph, whatever its extension.
        for candidate in (path.stem + ext for ext in (".JPG", ".jpg", ".RW2", ".ARW", ".CR3", ".NEF", ".DNG")):
            out[candidate] = {"pile": pile, "score": float(stars)}
        out[path.stem] = {"pile": pile, "score": float(stars)}
    return out


# Filled by `read_catalog_labels` so the caller can report it. A list rather
# than a count: naming the files is what lets somebody check the guard is not
# throwing away the whole set for a clock-skew reason.
_CONTAMINATED: list[str] = []


def contaminated() -> list[str]:
    return list(_CONTAMINATED)


def write_template(records, path: Path, *, seed: int = 0) -> int:
    """A CSV of filenames with an empty `human_pile` column, shuffled.

    Shuffled, and without the tool's own verdict in it, on purpose. A sheet that
    shows what the tool decided measures how persuadable the labeller is, not
    whether the thresholds are right, and an ordered sheet lets the previous row
    anchor the next one. Both turn an evening's work into a confirmation.
    """
    import random

    rows = [r.filename for r in records if r.filename]
    random.Random(seed).shuffle(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "human_pile"])
        for name in rows:
            writer.writerow([name, ""])
    return len(rows)


def compare(records, labels: dict[str, dict]) -> BenchResult:
    """Correlate stored measurements against the human labels."""
    result = BenchResult()
    by_name = {r.filename: r for r in records}
    result.missing = sorted(set(labels) - set(by_name))

    quality_pairs, uplift_pairs = [], []
    piles: list[tuple[str, int]] = []
    for name, label in labels.items():
        record = by_name.get(name)
        if record is None:
            continue
        result.labelled += 1
        scores = record.scores or {}
        if label.get("pile"):
            piles.append((label["pile"], int(record.final_score or 0)))
        if label.get("score") is None:
            continue
        quality_pairs.append((float(scores.get("current_quality", 0)), label["score"]))
        if "edited" in label and label["edited"] is not None:
            measured_gain = float(scores.get("post_edit_potential", 0)) - float(
                scores.get("current_quality", 0)
            )
            uplift_pairs.append((measured_gain, label["edited"] - label["score"]))

    if quality_pairs:
        result.quality = Correlation(
            rho=spearman([a for a, _ in quality_pairs], [b for _, b in quality_pairs]),
            n=len(quality_pairs),
            note="frame_quality against the human ranking",
        )
    if uplift_pairs:
        result.uplift = Correlation(
            rho=spearman([a for a, _ in uplift_pairs], [b for _, b in uplift_pairs]),
            n=len(uplift_pairs),
            note="measured uplift against the human before/after delta",
        )
    else:
        result.uplift = Correlation(
            note="no before/after labels: add a human_score_edited column"
        )
    result.piles = agreement(piles)
    return result


def pile_for(score: int, top: int, weak: int) -> str:
    """Which of the three piles a score falls in, at the given thresholds."""
    if score >= top:
        return "top"
    if score < weak:
        return "weak"
    return "good"


def agreement(pairs: list[tuple[str, int]]) -> PileAgreement:
    """How well the thresholds reproduce the human sorting, and what would do better.

    The sweep is the point. Reporting "the thresholds agree 61% of the time" is
    a complaint; reporting "and 82 / 51 would agree 74% of the time" is a
    number somebody can act on. It is not applied automatically -- a threshold
    fitted to one person's evening is that person's threshold, and the tool
    saying so out loud is the difference between calibration and overfitting.
    """
    # Read from the categoriser rather than repeated here: a report of how well
    # the thresholds agree has to use the thresholds that actually ran.
    from curation import DEFAULT_THRESHOLDS

    top_now, weak_now = DEFAULT_THRESHOLDS.top, DEFAULT_THRESHOLDS.weak
    out = PileAgreement(current_top=top_now, current_weak=weak_now)
    if not pairs:
        return out

    out.n = len(pairs)
    out.confusion = {human: dict.fromkeys(HUMAN_PILES, 0) for human in HUMAN_PILES}
    for human, score in pairs:
        tool = pile_for(score, top_now, weak_now)
        out.confusion[human][tool] += 1
        if tool == human:
            out.agreed += 1

    called_top = sum(out.confusion[h]["top"] for h in HUMAN_PILES)
    human_top = sum(out.confusion["top"].values())
    hit = out.confusion["top"]["top"]
    out.top_precision = hit / called_top if called_top else 0.0
    out.top_recall = hit / human_top if human_top else 0.0

    # Exhaustive rather than clever: 100 x 100 pairs over a few hundred frames
    # is milliseconds, and a closed form would need assumptions the data does
    # not support.
    best = (out.agreement, top_now, weak_now)
    for top in range(50, 100):
        for weak in range(10, top):
            hits = sum(1 for human, score in pairs if pile_for(score, top, weak) == human)
            share = hits / out.n
            if share > best[0]:
                best = (share, top, weak)
    out.best_agreement, out.best_top, out.best_weak = best
    return out


def format_report(result: BenchResult) -> str:
    lines = [
        "",
        "=" * 66,
        "QUALITY METRIC AGAINST HUMAN LABELS",
        "=" * 66,
        f"  Labelled photographs found:  {result.labelled}",
    ]
    if result.missing:
        lines.append(f"  Labelled but not analysed:   {len(result.missing)}")

    for name, correlation, validated in (
        ("frame_quality", result.quality, result.validates_quality),
        ("uplift", result.uplift, result.validates_uplift),
    ):
        lines.append("")
        lines.append(f"  {name}")
        if not correlation.n:
            lines.append(f"    not measured: {correlation.note}")
            continue
        lines.append(f"    Spearman rho: {correlation.rho:+.3f}  (n = {correlation.n})")
        if not correlation.meaningful:
            lines.append(
                f"    Too few samples to mean anything; {MIN_SAMPLES} is the minimum."
            )
        elif validated:
            lines.append("    Agrees with the human ranking well enough to report as measured.")
        else:
            lines.append(
                f"    Below {USEFUL_CORRELATION}: this number does not rank photographs "
                "the way a person does."
            )

    piles = result.piles
    lines.append("")
    lines.append("  the five piles, against a person's three")
    if not piles.n:
        lines.append(
            "    not measured: no human_pile column. This is the one that matters --\n"
            "    the thresholds are the product's central promise and nothing has\n"
            "    checked them against anybody."
        )
    else:
        lines.append(
            f"    Agreement: {piles.agreement:.0%}  (n = {piles.n}, "
            f"thresholds top {piles.current_top} / weak {piles.current_weak})"
        )
        lines.append(
            f"    Top pile:  precision {piles.top_precision:.0%}, "
            f"recall {piles.top_recall:.0%}"
        )
        lines.append("")
        lines.append("           tool: top   good   weak")
        for human in HUMAN_PILES:
            row = piles.confusion.get(human, {})
            counts = "".join(f"{row.get(t, 0):>7}" for t in HUMAN_PILES)
            lines.append(f"    human {human:<6}{counts}")
        lines.append("")
        if not piles.meaningful:
            lines.append(
                f"    Too few to mean anything; {MIN_SAMPLES} is the minimum and "
                "200-300 is where it starts being an answer."
            )
        elif piles.calibrated:
            lines.append(
                "    The thresholds reproduce this person's sorting well enough to "
                "be called calibrated."
            )
        else:
            lines.append(
                f"    Below {USEFUL_AGREEMENT:.0%}: these thresholds do not sort "
                "photographs the way this person does."
            )
        if (piles.best_top, piles.best_weak) != (piles.current_top, piles.current_weak):
            lines.append(
                f"    Best fit here: top {piles.best_top} / weak {piles.best_weak} "
                f"would agree {piles.best_agreement:.0%}."
            )
            lines.append(
                "    Not applied. A threshold fitted to one evening of labelling is\n"
                "    that evening's threshold; try it with `reclassify --profile-file`\n"
                "    and look at the photographs before believing it."
            )

    lines += [
        "",
        "  Why this exists: frame_quality is both the objective the preview",
        "  search optimises and the ruler that reports the result, so uplift",
        "  measures how far the search moved its own metric. Only an outside",
        "  ranking can say whether that corresponds to a better photograph.",
        "=" * 66,
    ]
    return "\n".join(lines)
