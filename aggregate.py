"""Stitching per-group rankings into one global order.

Stage 2 ranks twelve frames against each other and never assigns an absolute
score, which is the point -- but routing thresholds need a number per frame
across the whole archive. That is what this does.

Each group ranking is expanded into pairwise comparisons: if frame X placed
above frame Y in some group, that is one win for X. Bradley-Terry then fits a
strength per frame from all those pairs at once. Groups have to overlap for this
to work -- a frame that never shares a group with any other cluster of frames
cannot be placed relative to them, so `build_groups` deliberately overlaps.

The output is 0-100 per axis, which is what `RoutingConfig` thresholds compare
against.
"""

from __future__ import annotations

import math
from collections import defaultdict

DEFAULT_OVERLAP = 3
BT_ITERATIONS = 60
BT_PRIOR = 0.5


def build_groups(filenames: list[str], size: int = 12, overlap: int = DEFAULT_OVERLAP) -> list[list[str]]:
    """Split into groups of `size` that share `overlap` frames with the previous.

    The shared frames are the bridge: without them each group is its own island
    and the global ranking is meaningless across islands.
    """
    if size < 2:
        raise ValueError("group size must be at least 2")
    overlap = max(0, min(overlap, size - 1))
    if len(filenames) <= size:
        return [list(filenames)] if filenames else []

    step = size - overlap
    groups = []
    start = 0
    while start < len(filenames):
        group = filenames[start : start + size]
        if len(group) < 2 and groups:
            groups[-1].extend(group)
            break
        groups.append(group)
        if start + size >= len(filenames):
            break
        start += step
    return groups


def ranks_to_pairs(ranked: list[tuple[str, int]]) -> list[tuple[str, str]]:
    """Every ordered pair within one group's ranking: (winner, loser)."""
    ordered = sorted(ranked, key=lambda item: item[1])
    pairs = []
    for i, (better, rank_i) in enumerate(ordered):
        for worse, rank_j in ordered[i + 1 :]:
            if rank_i != rank_j:
                pairs.append((better, worse))
    return pairs


def bradley_terry(pairs: list[tuple[str, str]], iterations: int = BT_ITERATIONS) -> dict[str, float]:
    """Fit a strength per item from pairwise wins.

    Standard MM update with a smoothing prior, so an item that won or lost every
    comparison does not run off to infinity.
    """
    wins: dict[str, float] = defaultdict(float)
    meetings: dict[tuple[str, str], float] = defaultdict(float)
    items: set[str] = set()

    for winner, loser in pairs:
        wins[winner] += 1.0
        key = (winner, loser) if winner < loser else (loser, winner)
        meetings[key] += 1.0
        items.update((winner, loser))

    if not items:
        return {}

    strength = dict.fromkeys(items, 1.0)
    for _ in range(iterations):
        updated = {}
        for item in items:
            denominator = BT_PRIOR
            for (a, b), count in meetings.items():
                if item == a:
                    other = b
                elif item == b:
                    other = a
                else:
                    continue
                denominator += count / (strength[item] + strength[other])
            updated[item] = (wins[item] + BT_PRIOR) / denominator
        mean = sum(updated.values()) / len(updated)
        strength = {k: v / mean for k, v in updated.items()}
    return strength


def to_percentile_scores(strength: dict[str, float]) -> dict[str, int]:
    """Map fitted strengths onto 0-100 by rank, so the scale is always populated.

    Using the rank rather than the raw strength keeps the distribution flat: the
    thresholds in RoutingConfig are meant to mean "top ~30%", not "stronger than
    some arbitrary fitted value".
    """
    if not strength:
        return {}
    if len(strength) == 1:
        return dict.fromkeys(strength, 100)

    ordered = sorted(strength.items(), key=lambda kv: (-kv[1], kv[0]))
    last = len(ordered) - 1
    return {name: round(100 * (last - i) / last) for i, (name, _) in enumerate(ordered)}


def aggregate_axis(group_rankings: list[list[tuple[str, int]]]) -> dict[str, int]:
    """One axis, many overlapping groups, out to a global 0-100 per frame."""
    pairs: list[tuple[str, str]] = []
    for ranked in group_rankings:
        pairs.extend(ranks_to_pairs(ranked))
    return to_percentile_scores(bradley_terry(pairs))


def aggregate_all_axes(
    group_results: list[list[dict]],
) -> dict[str, dict[str, int]]:
    """Aggregate axis_a, axis_b and axis_c independently.

    `group_results` is one list per group of parsed model objects, each carrying
    `filename` and the three within-group ranks. The axes are never combined --
    a frame can top axis_b and sit at the bottom of axis_a, which is exactly the
    signal the three-axis split exists to keep.
    """
    scores: dict[str, dict[str, int]] = {}
    for axis in ("axis_a", "axis_b", "axis_c"):
        rankings = [
            [(item["filename"], int(item[axis])) for item in group if axis in item]
            for group in group_results
        ]
        scores[axis] = aggregate_axis([r for r in rankings if len(r) >= 2])
    return scores


def spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Rank correlation over the keys the two share. Used by --bench."""
    shared = sorted(set(a) & set(b))
    n = len(shared)
    if n < 2:
        return float("nan")

    ranks_a = _ranks([a[k] for k in shared])
    ranks_b = _ranks([b[k] for k in shared])
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n

    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ranks_a, ranks_b, strict=True))
    var_a = math.sqrt(sum((x - mean_a) ** 2 for x in ranks_a))
    var_b = math.sqrt(sum((y - mean_b) ** 2 for y in ranks_b))
    if var_a == 0 or var_b == 0:
        return float("nan")
    return cov / (var_a * var_b)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks, so ties do not distort the correlation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared_rank
        i = j + 1
    return ranks
