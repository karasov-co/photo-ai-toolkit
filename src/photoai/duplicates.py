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

The similarity function is injectable, and there are now two implementations of
it. `visual_similarity` is perceptual-hash distance combined with genre: cheap,
offline, no dependencies, and good enough to catch bursts and repeated framings.
`embedding_similarity` is the cosine between two CLIP image vectors, which is
what separates "two different photographs that happen to share a colour palette"
from "two frames of one moment". The default, `default_similarity`, uses the
second when both frames carry a vector and the first when they do not -- so the
tool behaves exactly as it always did with nothing extra installed, and gets
better when `photoai.embeddings` is available.

Deciding *which* is used is not done here, per pair. `photoai.embeddings` is
probed once per run; if it answers yes, the vectors are attached to `DupItem`
upstream, and their presence is the signal. A frame with no vector falls back on
its own without dragging the rest of the run down with it, and the run reports
which of the two measurements it actually used.

Clustering is a separate question and still runs on the hash alone: the BK-tree
below needs a metric with a triangle inequality over a discrete space, which
Hamming distance over 64 bits is and cosine over 512 floats is not.
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

# Two photographs of unrelated subjects still score around 0.5-0.7 on raw CLIP
# cosine, and a burst scores about 0.98. The number is informative but it does
# not use its range, and maximal marginal relevance subtracts it from a quality
# term on 0..1 -- so feeding the raw cosine in makes the redundancy term nearly
# constant and quietly turns the diversity pass back into a ranking by score.
#
# So the band that carries the signal is stretched over 0..1: everything at or
# below the floor is "unrelated", 1.0 stays "the same photograph". The floor is
# a calibration constant, not a law, and it is here rather than inline so that
# it can be measured against a labelled set later and changed in one place.
EMBEDDING_FLOOR = 0.55
# Same genre, same nudge as the hash path uses. Two frames of one genre are more
# interchangeable in a portfolio than two of different genres at equal distance,
# whichever way the distance was measured, so switching to embeddings does not
# silently change what `lambda_` means.
SAME_GENRE_NUDGE = 0.12


@dataclass
class DupItem:
    """The minimum a frame must expose to be clustered.

    `embedding` is optional and normally absent: it is a unit vector from
    `photoai.embeddings` when that encoder was available for the run, and `None`
    or `()` otherwise. Nothing in this module ever computes one -- it is handed
    in, so that clustering stays a pure function of what it was given.
    """

    key: str
    phash: str = ""
    date_shot: str | None = None
    quality: float = 0.0
    genre: str = "other"
    embedding: Sequence[float] | None = None


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


def visual_similarity(a: DupItem, b: DupItem) -> float:
    """Perceptual hash, nudged by genre. Not an embedding, and never was.

    This was once called `embedding_similarity`, which described the extension
    point rather than the function, and a name that promises a semantic
    comparison while doing a hash comparison is the kind of thing a reader
    believes. The name it has now is the whole of what it does.

    What it measures is layout and tone at 8x8. Two different subjects
    photographed with the same palette and framing score as near-identical here,
    and the diversity pass will treat one as a repeat of the other. That is the
    known limit of this function, and it is why `embedding_similarity` below
    exists -- but this one stays the default whenever no encoder is installed,
    because it needs nothing, costs nothing, and finds bursts correctly.

    The genre nudge stands on its own: two frames of the same genre are more
    interchangeable in a portfolio than two of different genres at equal visual
    distance, which is the property the diversity pass wants.
    """
    visual = phash_similarity(a.phash, b.phash)
    same_genre = SAME_GENRE_NUDGE if a.genre == b.genre else 0.0
    return min(1.0, visual + same_genre)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, in -1..1. 0.0 if either is empty.

    Written out rather than handed to NumPy because this module has no imports
    and is called in the inner loop of the selection pass, where the cost of
    building two arrays per pair is larger than the dot product itself.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))


def has_embedding(item: DupItem) -> bool:
    return bool(item.embedding)


def embedding_similarity(a: DupItem, b: DupItem) -> float:
    """Cosine between two CLIP image vectors, rescaled, nudged by genre.

    This is the semantic measurement the docstring above used to promise and not
    deliver. Two frames of a beige wall and a beige beach share a palette, a
    framing and a perceptual hash; they do not share a vector, and the diversity
    pass keeps both.

    Three things happen, in this order, and all three are visible in the number:

    1. The cosine of the two unit vectors, which is their dot product.
    2. A rescale from `EMBEDDING_FLOOR`..1 onto 0..1, clamped. Raw CLIP cosine
       does not use its range -- see the constant -- and the caller subtracts
       this from a 0..1 quality term.
    3. The same genre nudge `visual_similarity` applies, so that swapping one
       function for the other does not change what `lambda_` means.

    A frame with no vector falls through to `visual_similarity` for that pair
    alone. That is deliberate rather than an error: one unreadable preview in a
    run of three hundred should cost that frame its semantic comparison and
    nothing else.
    """
    if not has_embedding(a) or not has_embedding(b):
        return visual_similarity(a, b)
    raw = cosine(a.embedding, b.embedding)
    span = 1.0 - EMBEDDING_FLOOR
    scaled = (raw - EMBEDDING_FLOOR) / span if span > 0 else raw
    scaled = min(1.0, max(0.0, scaled))
    same_genre = SAME_GENRE_NUDGE if a.genre == b.genre else 0.0
    return round(min(1.0, scaled + same_genre), 4)


def default_similarity(a: DupItem, b: DupItem) -> float:
    """Semantic when both frames carry a vector, perceptual otherwise.

    The choice is made per pair on evidence that is already in hand, not by
    asking `embeddings.available()` here: that probe stats a 335 MB file, and
    this function is called O(picks x candidates) times per run. Availability is
    decided once, upstream, and arrives as vectors on the items.
    """
    if has_embedding(a) and has_embedding(b):
        return embedding_similarity(a, b)
    return visual_similarity(a, b)


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
    for a, b in _candidate_pairs(ordered, distance):
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


# A BK-tree over Hamming distance. Every pair still gets compared in the worst
# case -- an archive of one photograph repeated 2000 times genuinely is all one
# cluster -- but a real archive is mostly dissimilar, and the tree skips the
# comparisons the triangle inequality has already ruled out.
#
# At 300 files the quadratic scan is 45,000 comparisons and nobody notices. At
# 5,000 it is 12.5 million, which is the point somebody starts wondering whether
# the tool has hung.


class _BKTree:
    """Nodes keyed by integer hash; children indexed by distance from the parent.

    Hamming distance is a metric, so |d(query, node) - d(node, child)| bounds
    d(query, child) from below: any child outside [d-r, d+r] cannot be within r
    of the query and its whole subtree is skipped.
    """

    __slots__ = ("value", "payload", "children")

    def __init__(self, value: int, payload):
        self.value = value
        self.payload = payload
        self.children: dict[int, _BKTree] = {}

    def add(self, value: int, payload) -> None:
        node = self
        while True:
            gap = bin(node.value ^ value).count("1")
            child = node.children.get(gap)
            if child is None:
                node.children[gap] = _BKTree(value, payload)
                return
            node = child

    def within(self, value: int, radius: int) -> list:
        found = []
        stack = [self]
        while stack:
            node = stack.pop()
            gap = bin(node.value ^ value).count("1")
            if gap <= radius:
                found.append(node.payload)
            for step, child in node.children.items():
                if gap - radius <= step <= gap + radius:
                    stack.append(child)
        return found


def _candidate_pairs(items: list[DupItem], distance: int):
    """Every pair worth comparing, and as few others as possible.

    Items with an unusable hash are handled separately: `hamming` reports them
    as maximally far, so they can never join a cluster, and putting them in the
    tree would only cost lookups.
    """
    hashed: list[tuple[int, DupItem]] = []
    for item in items:
        try:
            hashed.append((int(item.phash, 16), item))
        except (TypeError, ValueError):
            continue

    if len(hashed) < 2:
        return

    tree = _BKTree(hashed[0][0], hashed[0][1])
    seen = [hashed[0][1]]
    for value, item in hashed[1:]:
        for other in tree.within(value, distance):
            yield other, item
        tree.add(value, item)
        seen.append(item)


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
    similarity: Callable[[DupItem, DupItem], float] = default_similarity,
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

    # Redundancy against the chosen set only grows, one entry per round, so the
    # running maximum is carried forward instead of recomputed. The loop used to
    # call `similarity` once per (remaining, chosen) pair on every round --
    # O(n * k^2) for k picks, and on a large archive with a generous flagship
    # cap that is the slowest thing in the run after decoding.
    redundancy: dict[str, float] = {c.key: 0.0 for c in candidates}

    while remaining and len(chosen) < limit:
        best = None
        best_value = float("-inf")
        for candidate in remaining:
            if max_per_genre is not None and per_genre.get(candidate.genre, 0) >= max_per_genre:
                continue
            value = (
                lambda_ * (candidate.relevance / 100.0)
                - (1.0 - lambda_) * redundancy[candidate.key]
            )
            if value > best_value:
                best_value, best = value, candidate
        if best is None:
            break
        chosen.append(best)
        per_genre[best.genre] = per_genre.get(best.genre, 0) + 1
        remaining.remove(best)
        for candidate in remaining:
            redundancy[candidate.key] = max(
                redundancy[candidate.key], similarity(candidate.item, best.item)
            )

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
