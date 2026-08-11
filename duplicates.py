"""Near-duplicates, and keeping the flagship gallery from becoming one burst.

Two related problems with one shared notion of similarity.

**Clustering** collapses a burst to the frame worth keeping. Similarity alone is
not enough: two photographs of the same facade taken on different days are
near-identical by perceptual hash and are not duplicates of each other. So a
timestamp, where one exists, has to agree. Frames with no timestamp are only
ever grouped on similarity, and the window is skipped rather than assumed.

**Diversity selection** is the same measurement used for the opposite purpose.
Ranking by score alone fills a flagship gallery with twenty frames of the same
sunset, because if one of them is the best photograph of the shoot, the other
nineteen are the next nineteen. Maximal marginal relevance fixes this by making
each pick compete against what has already been picked:

    value = lambda * quality - (1 - lambda) * (similarity to nearest chosen)

The similarity function is injectable. The default is perceptual-hash distance
combined with genre, which is cheap, offline, and good enough to catch bursts
and repeated framings. A CLIP or SigLIP embedding would separate "two different
photographs that happen to share a colour palette" more reliably; that is the
documented extension point, and `embedding_similarity` is where it plugs in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

DEFAULT_DISTANCE = 8
HASH_BITS = 64
BURST_WINDOW_SECONDS = 3.0
# Frames further apart than this are never one cluster, however alike they look.
SAME_SCENE_SECONDS = 120.0


@dataclass
class DupItem:
    """The minimum a frame must expose to be clustered."""

    key: str
    phash: str = ""
    date_shot: str | None = None
    quality: float = 0.0
    genre: str = "other"


@dataclass
class Cluster:
    items: list[DupItem] = field(default_factory=list)
    best_key: str = ""

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def is_singleton(self) -> bool:
        return self.size <= 1

    def keys(self) -> list[str]:
        return [i.key for i in self.items]

    def mean_similarity(self) -> float:
        """How alike the cluster is internally, 0..1. Feeds the uniqueness score."""
        if self.size < 2:
            return 0.0
        pairs = [
            phash_similarity(a.phash, b.phash)
            for i, a in enumerate(self.items)
            for b in self.items[i + 1 :]
        ]
        return round(sum(pairs) / len(pairs), 3) if pairs else 0.0


def hamming(a: str, b: str) -> int:
    """Hex perceptual hashes to a bit distance. Unknown hashes are maximally far."""
    if not a or not b or len(a) != len(b):
        return HASH_BITS
    try:
        return bin(int(a, 16) ^ int(b, 16)).count("1")
    except ValueError:
        return HASH_BITS


def phash_similarity(a: str, b: str) -> float:
    return round(1.0 - hamming(a, b) / HASH_BITS, 4)


def embedding_similarity(a: DupItem, b: DupItem) -> float:
    """Default similarity: perceptual hash, nudged by genre.

    Two frames of the same genre are more interchangeable in a portfolio than
    two of different genres even at equal visual distance, which is the property
    the diversity pass actually wants.
    """
    visual = phash_similarity(a.phash, b.phash)
    same_genre = 0.12 if a.genre == b.genre else 0.0
    return min(1.0, visual + same_genre)


def _seconds_between(a: str | None, b: str | None) -> float | None:
    if not a or not b:
        return None
    try:
        return abs((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())
    except (ValueError, TypeError):
        return None


def cluster_items(
    items: Sequence[DupItem],
    *,
    distance: int = DEFAULT_DISTANCE,
    window_seconds: float = SAME_SCENE_SECONDS,
) -> list[Cluster]:
    """Group near-identical frames, with time as a veto rather than a requirement.

    Union-find rather than a single greedy pass so that a chain A~B, B~C puts
    all three together even when A and C are individually too far apart -- a
    slow pan across a scene produces exactly that chain.
    """
    parent: dict[str, str] = {i.key: i.key for i in items}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ordered = list(items)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1 :]:
            if hamming(a.phash, b.phash) > distance:
                continue
            gap = _seconds_between(a.date_shot, b.date_shot)
            if gap is not None and gap > window_seconds:
                # Looks the same, taken hours apart: the same subject twice, not
                # a burst. Two separate photographs, both worth keeping.
                continue
            union(a.key, b.key)

    grouped: dict[str, list[DupItem]] = {}
    for item in ordered:
        grouped.setdefault(find(item.key), []).append(item)

    clusters = []
    for members in grouped.values():
        best = max(members, key=lambda i: (i.quality, i.key))
        clusters.append(Cluster(items=members, best_key=best.key))
    return sorted(clusters, key=lambda c: c.best_key)


def is_burst(a: DupItem, b: DupItem, *, window: float = BURST_WINDOW_SECONDS) -> bool:
    """Shot back to back and alike: the narrow case, for reporting."""
    gap = _seconds_between(a.date_shot, b.date_shot)
    return gap is not None and gap <= window and hamming(a.phash, b.phash) <= DEFAULT_DISTANCE


# --- diversity-aware selection ----------------------------------------------


@dataclass
class Candidate:
    key: str
    relevance: float
    item: DupItem

    @property
    def genre(self) -> str:
        return self.item.genre


def select_diverse(
    candidates: Sequence[Candidate],
    *,
    limit: int,
    lambda_: float = 0.65,
    max_per_genre: int | None = None,
    similarity: Callable[[DupItem, DupItem], float] = embedding_similarity,
) -> list[str]:
    """Maximal marginal relevance: quality, discounted by redundancy.

    The first pick is simply the best. Every pick after that is the one with the
    best (quality, unlike everything already chosen) trade-off, which is what
    keeps the gallery from being one scene photographed twenty times.
    """
    if limit <= 0 or not candidates:
        return []

    remaining = list(candidates)
    chosen: list[Candidate] = []
    per_genre: dict[str, int] = {}

    while remaining and len(chosen) < limit:
        best = None
        best_value = float("-inf")
        for candidate in remaining:
            if max_per_genre is not None and per_genre.get(candidate.genre, 0) >= max_per_genre:
                continue
            redundancy = max(
                (similarity(candidate.item, c.item) for c in chosen),
                default=0.0,
            )
            value = lambda_ * (candidate.relevance / 100.0) - (1.0 - lambda_) * redundancy
            if value > best_value:
                best_value, best = value, candidate
        if best is None:
            break
        chosen.append(best)
        per_genre[best.genre] = per_genre.get(best.genre, 0) + 1
        remaining.remove(best)

    return [c.key for c in chosen]


def strongest_per_genre(candidates: Sequence[Candidate]) -> dict[str, str]:
    """One winner per genre, so no genre is shut out by a tidier one.

    Landscape beats street on any shared scale because landscape is tidier. A
    purely global cut therefore empties the reportage and street buckets and
    fills the gallery with sunsets.
    """
    best: dict[str, Candidate] = {}
    for candidate in candidates:
        current = best.get(candidate.genre)
        if current is None or candidate.relevance > current.relevance:
            best[candidate.genre] = candidate
    return {genre: c.key for genre, c in best.items()}
