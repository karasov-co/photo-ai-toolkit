"""Clustering bursts, and keeping the flagship gallery from becoming one of them."""

import pytest

from photoai import duplicates
from photoai.duplicates import (
    Candidate,
    Cluster,
    DupItem,
    cluster_items,
    hamming,
    is_burst,
    phash_similarity,
    select_diverse,
    strongest_per_genre,
    visual_similarity,
)

# Hex phashes one bit apart, then progressively further.
H0 = "0000000000000000"
H1 = "0000000000000001"
H3 = "0000000000000007"
FAR = "ffffffffffffffff"


def item(key, phash=H0, when=None, quality=50.0, genre="landscape"):
    return DupItem(key=key, phash=phash, date_shot=when, quality=quality, genre=genre)


# --- distance ---------------------------------------------------------------


def test_identical_hashes_are_zero_apart():
    assert hamming(H0, H0) == 0


def test_one_differing_bit_is_distance_one():
    assert hamming(H0, H1) == 1


def test_an_unknown_hash_is_maximally_far():
    """Never let a missing hash read as 'identical' -- that would delete a photo."""
    assert hamming("", H0) == 64
    assert hamming(H0, "not hex") == 64


def test_similarity_is_the_inverse_of_distance():
    assert phash_similarity(H0, H0) == 1.0
    assert phash_similarity(H0, FAR) == 0.0


# --- clustering -------------------------------------------------------------


def test_a_burst_collapses_to_one_cluster():
    frames = [
        item("a", H0, "2026-03-14T09:26:53"),
        item("b", H1, "2026-03-14T09:26:54"),
        item("c", H3, "2026-03-14T09:26:55"),
    ]
    clusters = cluster_items(frames)
    assert len(clusters) == 1
    assert clusters[0].size == 3


def test_different_photographs_stay_separate():
    clusters = cluster_items([item("a", H0), item("b", FAR)])
    assert len(clusters) == 2


def test_the_sharpest_frame_wins_the_cluster():
    frames = [
        item("soft", H0, "2026-03-14T09:26:53", quality=30.0),
        item("sharp", H1, "2026-03-14T09:26:54", quality=80.0),
        item("softer", H3, "2026-03-14T09:26:55", quality=20.0),
    ]
    assert cluster_items(frames)[0].best_key == "sharp"


def test_the_same_facade_photographed_on_two_days_is_not_a_duplicate():
    """Perceptual hash alone would merge them; the timestamp vetoes it."""
    frames = [
        item("monday", H0, "2026-03-14T09:00:00"),
        item("tuesday", H1, "2026-03-15T09:00:00"),
    ]
    assert len(cluster_items(frames)) == 2


def test_frames_without_timestamps_are_clustered_on_looks_alone():
    """No timestamp means no veto is available, not that the veto fails closed."""
    assert len(cluster_items([item("a", H0), item("b", H1)])) == 1


def test_a_chain_of_similar_frames_becomes_one_cluster():
    """A slow pan gives A~B and B~C while A and C are individually too far."""
    frames = [
        item("a", "0000000000000000", "2026-03-14T09:26:50"),
        item("b", "00000000000000ff", "2026-03-14T09:26:51"),
        item("c", "000000000000ffff", "2026-03-14T09:26:52"),
    ]
    assert len(cluster_items(frames, distance=8)) == 1


def test_a_lone_frame_is_a_cluster_of_one():
    clusters = cluster_items([item("only")])
    assert clusters[0].is_singleton
    assert clusters[0].best_key == "only"


def test_no_frames_gives_no_clusters():
    assert cluster_items([]) == []


def test_cluster_similarity_reports_how_alike_the_group_is():
    tight = Cluster(items=[item("a", H0), item("b", H1)])
    loose = Cluster(items=[item("a", H0), item("b", FAR)])
    assert tight.mean_similarity() > loose.mean_similarity()


def test_a_singleton_has_no_internal_similarity():
    assert Cluster(items=[item("a")]).mean_similarity() == 0.0


def test_a_burst_is_recognised_by_time_and_looks_together():
    a = item("a", H0, "2026-03-14T09:26:53")
    b = item("b", H1, "2026-03-14T09:26:54")
    far_apart = item("c", H1, "2026-03-14T09:30:00")
    assert is_burst(a, b)
    assert not is_burst(a, far_apart)


# --- diversity --------------------------------------------------------------


def candidate(key, relevance, phash=H0, genre="landscape"):
    return Candidate(key=key, relevance=relevance, item=item(key, phash, genre=genre))


def test_the_best_frame_is_always_picked_first():
    picks = select_diverse(
        [candidate("weak", 40, FAR), candidate("best", 95, H0)], limit=1
    )
    assert picks == ["best"]


def test_twenty_near_identical_frames_do_not_fill_the_gallery():
    """Ranking by score alone gives the same sunset twenty times."""
    same_sunset = [candidate(f"sunset{i}", 90 - i, H0) for i in range(20)]
    different = [candidate("street", 60, FAR, genre="street")]
    picks = select_diverse(same_sunset + different, limit=3, lambda_=0.6)
    assert "street" in picks
    assert sum(1 for p in picks if p.startswith("sunset")) < 3


def test_a_high_lambda_prefers_quality_over_variety():
    same = [candidate(f"s{i}", 90 - i, H0) for i in range(5)]
    other = [candidate("other", 50, FAR)]
    quality_first = select_diverse(same + other, limit=2, lambda_=0.99)
    assert all(p.startswith("s") for p in quality_first)


def test_a_low_lambda_prefers_variety_over_quality():
    same = [candidate(f"s{i}", 90 - i, H0) for i in range(5)]
    other = [candidate("other", 50, FAR)]
    variety_first = select_diverse(same + other, limit=2, lambda_=0.1)
    assert "other" in variety_first


def test_a_genre_cap_is_respected():
    frames = [candidate(f"land{i}", 90 - i, f"{i:016x}", "landscape") for i in range(6)]
    picks = select_diverse(frames, limit=5, max_per_genre=2)
    assert len(picks) == 2


def test_the_limit_is_respected():
    frames = [candidate(f"f{i}", 90 - i, f"{i:016x}") for i in range(20)]
    assert len(select_diverse(frames, limit=4)) == 4


def test_asking_for_nothing_returns_nothing():
    assert select_diverse([candidate("a", 90)], limit=0) == []


def test_no_candidates_gives_no_picks():
    assert select_diverse([], limit=5) == []


def test_selection_never_repeats_a_frame():
    frames = [candidate(f"f{i}", 90 - i, f"{i:016x}") for i in range(6)]
    picks = select_diverse(frames, limit=6)
    assert len(picks) == len(set(picks))


def test_the_same_genre_counts_as_more_alike_than_a_different_one():
    """Measured at a visual distance where the clamp is not already binding."""
    a = item("a", "00000000000000ff", genre="landscape")
    same = item("b", "000000000000ff00", genre="landscape")
    other = item("c", "000000000000ff00", genre="street")
    assert visual_similarity(a, same) > visual_similarity(a, other)


def test_similarity_never_exceeds_one_even_with_the_genre_bonus():
    a = item("a", H0, genre="landscape")
    assert visual_similarity(a, a) == 1.0


# --- per-genre selection ----------------------------------------------------


def test_each_genre_keeps_its_own_winner():
    """A global cut empties street and reportage and fills the gallery with sunsets."""
    frames = [
        candidate("sunset", 95, genre="landscape"),
        candidate("hill", 90, genre="landscape"),
        candidate("market", 55, FAR, genre="street"),
        candidate("alley", 40, FAR, genre="street"),
    ]
    winners = strongest_per_genre(frames)
    assert winners == {"landscape": "sunset", "street": "market"}


@pytest.mark.parametrize("count", [1, 5, 50])
def test_selection_scales_without_error(count):
    frames = [candidate(f"f{i}", 100 - i, f"{i:016x}") for i in range(count)]
    assert len(select_diverse(frames, limit=10)) == min(10, count)


# --- the BK-tree returns exactly what the quadratic scan did ------------------


def _quadratic_clusters(items, distance=duplicates.DEFAULT_DISTANCE,
                        window=duplicates.SAME_SCENE_SECONDS):
    """The original all-pairs implementation, kept as the reference."""
    parent = {i.key: i.key for i in items}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    ordered = list(items)
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if duplicates.hamming(a.phash, b.phash) > distance:
                continue
            gap = duplicates._seconds_between(a.date_shot, b.date_shot)
            if gap is not None and gap > window:
                continue
            ra, rb = find(a.key), find(b.key)
            if ra != rb:
                parent[rb] = ra

    grouped = {}
    for item in ordered:
        grouped.setdefault(find(item.key), []).append(item)
    out = []
    for members in grouped.values():
        best = max(members, key=lambda i: (i.quality, i.key))
        out.append((best.key, sorted(m.key for m in members)))
    return sorted(out)


def _synthetic(n, seed=7):
    import random

    random.seed(seed)
    items = []
    for i in range(n):
        value = random.getrandbits(64)
        if i % 3 and items:
            # A near-duplicate of the previous frame, one bit away.
            value = int(items[-1].phash, 16) ^ (1 << random.randrange(64))
        items.append(
            duplicates.DupItem(
                key=f"f{i:05d}", phash=f"{value:016x}",
                date_shot=f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}",
                quality=(i * 37) % 100,
            )
        )
    return items


@pytest.mark.parametrize("n", [50, 500, 2000])
def test_the_tree_and_the_full_scan_agree_exactly(n):
    """Same clusters, same winners, same membership -- at 2000 files."""
    items = _synthetic(n)
    fast = [(c.best_key, sorted(i.key for i in c.items)) for c in duplicates.cluster_items(items)]
    assert sorted(fast) == _quadratic_clusters(items)


def test_a_chain_still_joins_through_its_middle():
    """A~B and B~C but not A~C: a slow pan. All three are one cluster."""
    items = [
        duplicates.DupItem(key="a", phash="0000000000000000", date_shot="2026-01-01T00:00:00"),
        duplicates.DupItem(key="b", phash="000000000000003f", date_shot="2026-01-01T00:00:01"),
        duplicates.DupItem(key="c", phash="0000000000000fc0", date_shot="2026-01-01T00:00:02"),
    ]
    clusters = duplicates.cluster_items(items, distance=6)
    assert len(clusters) == 1
    assert sorted(i.key for i in clusters[0].items) == ["a", "b", "c"]


def test_an_unusable_hash_never_joins_a_cluster():
    items = [
        duplicates.DupItem(key="ok", phash="0000000000000000", date_shot="2026-01-01T00:00:00"),
        duplicates.DupItem(key="broken", phash="", date_shot="2026-01-01T00:00:01"),
        duplicates.DupItem(key="nonsense", phash="zzzz", date_shot="2026-01-01T00:00:02"),
    ]
    clusters = duplicates.cluster_items(items)
    assert len(clusters) == 3


def test_two_thousand_files_cluster_quickly():
    """Not a benchmark -- a guard against the quadratic scan coming back."""
    import time

    items = _synthetic(2000)
    started = time.perf_counter()
    duplicates.cluster_items(items)
    assert time.perf_counter() - started < 5.0


def test_each_similarity_function_names_what_it_actually_does():
    """The rule this test has always enforced, now that there are two of them.

    It used to assert that no `embedding_similarity` existed at all, because the
    only function here compared perceptual hashes and had once carried that
    name. There is one now, and it really does compare vectors -- so the
    assertion becomes the honest version of the same rule: `visual_similarity`
    must still refuse to claim it embeds, and `embedding_similarity` must
    actually take the embeddings into account rather than being a second name
    for the hash comparison.
    """
    from photoai import duplicates

    # The hash function still says, in its own docstring, what it cannot do.
    doc = duplicates.visual_similarity.__doc__.lower()
    assert "not an embedding" in doc
    assert "perceptual hash" in doc
    assert "same palette and framing" in doc

    # And the embedding function is not the hash function wearing a better name.
    same_look = item("a", H0, genre="landscape")
    other_look = item("b", H0, genre="landscape")
    assert visual_similarity(same_look, other_look) == 1.0

    same_look.embedding = (1.0, 0.0, 0.0)
    other_look.embedding = (0.0, 1.0, 0.0)
    assert duplicates.embedding_similarity(same_look, other_look) < 0.5


# --- the semantic vector ------------------------------------------------------
#
# `embedding_similarity` compares two unit vectors; where those vectors come
# from is `photoai.embeddings`, and nothing below loads or downloads a model.
# The maths is checked directly, and the plumbing is checked with a stub
# encoder, which is the only way this can stay hermetic.


def test_cosine_of_a_vector_with_itself_is_one():
    v = (0.6, 0.8)
    assert duplicates.cosine(v, v) == pytest.approx(1.0)


def test_cosine_of_perpendicular_vectors_is_zero():
    assert duplicates.cosine((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


def test_cosine_of_opposite_vectors_is_minus_one():
    assert duplicates.cosine((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)


def test_cosine_ignores_length():
    """It is the angle, so scaling either vector must change nothing."""
    assert duplicates.cosine((1.0, 1.0), (3.0, 3.0)) == pytest.approx(1.0)


def test_cosine_of_a_missing_or_mismatched_vector_is_zero():
    """Never let a missing vector read as 'identical' -- see the hash version."""
    assert duplicates.cosine((), (1.0, 0.0)) == 0.0
    assert duplicates.cosine((1.0, 0.0), (1.0, 0.0, 0.0)) == 0.0
    assert duplicates.cosine((0.0, 0.0), (1.0, 0.0)) == 0.0


def test_the_encoder_returns_unit_vectors():
    """Normalisation is what makes a dot product a cosine at all."""
    from photoai import embeddings

    out = embeddings.normalise([3.0, 4.0])
    assert len(out) == 2
    assert sum(v * v for v in out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.6)


def test_a_zero_vector_normalises_to_nothing_rather_than_to_nan():
    from photoai import embeddings

    assert embeddings.normalise([0.0, 0.0, 0.0]) == ()


def test_an_identical_pair_of_vectors_is_maximally_similar():
    a = item("a", H0, genre="landscape")
    b = item("b", FAR, genre="landscape")
    a.embedding = b.embedding = (0.6, 0.8)
    assert duplicates.embedding_similarity(a, b) == 1.0


def test_embedding_similarity_never_leaves_the_unit_interval():
    """Including for the opposite vectors, where the raw cosine is -1."""
    a = item("a", H0, genre="landscape")
    b = item("b", H0, genre="street")
    a.embedding, b.embedding = (1.0, 0.0), (-1.0, 0.0)
    assert 0.0 <= duplicates.embedding_similarity(a, b) <= 1.0
    assert duplicates.embedding_similarity(a, b) == 0.0


def test_the_genre_nudge_survives_the_switch_to_vectors():
    """So that `lambda_` means the same thing under either measurement."""
    a = item("a", H0, genre="landscape")
    same = item("b", H0, genre="landscape")
    other = item("c", H0, genre="street")
    a.embedding = same.embedding = other.embedding = (0.7071, 0.7071)
    a.embedding = (1.0, 0.0)
    assert duplicates.embedding_similarity(a, same) > duplicates.embedding_similarity(a, other)


# --- choosing between the two -------------------------------------------------


def test_the_default_uses_vectors_when_both_frames_have_one():
    a = item("a", H0, genre="landscape")
    b = item("b", H0, genre="landscape")
    assert duplicates.default_similarity(a, b) == 1.0  # identical hashes

    a.embedding, b.embedding = (1.0, 0.0), (0.0, 1.0)
    assert duplicates.default_similarity(a, b) < 0.5


def test_the_default_falls_back_to_the_hash_when_a_vector_is_missing():
    """One unreadable preview costs that frame its vector, not the whole run."""
    a = item("a", H0, genre="landscape")
    b = item("b", H0, genre="landscape")
    a.embedding = (1.0, 0.0)
    b.embedding = None
    assert duplicates.default_similarity(a, b) == visual_similarity(a, b) == 1.0

    b.embedding = ()
    assert duplicates.default_similarity(a, b) == 1.0


def test_select_diverse_uses_the_default_and_therefore_the_vectors():
    """No `similarity=` passed: the caller gets the better one for free."""
    beige_wall = candidate("wall", 90, H0)
    beige_beach = candidate("beach", 88, H0)
    beige_wall.item.embedding = (1.0, 0.0, 0.0)
    beige_beach.item.embedding = (0.0, 1.0, 0.0)
    other = candidate("street", 40, FAR, genre="street")

    picks = select_diverse([beige_wall, beige_beach, other], limit=2, lambda_=0.6)
    assert picks == ["wall", "beach"]


def test_the_same_two_frames_are_merged_when_only_the_hash_is_available():
    """The other half of the test above: this is what the fallback costs.

    Not a bug being asserted -- a limit being pinned. With no encoder these two
    frames are one photograph as far as the tool can tell, and the weaker one
    loses its place to a frame fifty points worse.
    """
    beige_wall = candidate("wall", 90, H0)
    beige_beach = candidate("beach", 88, H0)
    other = candidate("street", 40, FAR, genre="street")

    picks = select_diverse([beige_wall, beige_beach, other], limit=2, lambda_=0.6)
    assert picks == ["wall", "street"]
