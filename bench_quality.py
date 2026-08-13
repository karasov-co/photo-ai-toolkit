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
class BenchResult:
    quality: Correlation = field(default_factory=Correlation)
    uplift: Correlation = field(default_factory=Correlation)
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
                if entry.get("score") is not None:
                    out[name] = entry
    return out


def _number(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def compare(records, labels: dict[str, dict]) -> BenchResult:
    """Correlate stored measurements against the human labels."""
    result = BenchResult()
    by_name = {r.filename: r for r in records}
    result.missing = sorted(set(labels) - set(by_name))

    quality_pairs, uplift_pairs = [], []
    for name, label in labels.items():
        record = by_name.get(name)
        if record is None:
            continue
        result.labelled += 1
        scores = record.scores or {}
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
    return result


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

    lines += [
        "",
        "  Why this exists: frame_quality is both the objective the preview",
        "  search optimises and the ruler that reports the result, so uplift",
        "  measures how far the search moved its own metric. Only an outside",
        "  ranking can say whether that corresponds to a better photograph.",
        "=" * 66,
    ]
    return "\n".join(lines)
