"""Clustering bursts, and keeping the flagship gallery from becoming one of them."""

import pytest

from duplicates import (
    Candidate,
    Cluster,
    DupItem,
    cluster_items,
    embedding_similarity,
    hamming,
    is_burst,
    phash_similarity,
    select_diverse,
    strongest_per_genre,
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
    assert embedding_similarity(a, same) > embedding_similarity(a, other)


def test_similarity_never_exceeds_one_even_with_the_genre_bonus():
    a = item("a", H0, genre="landscape")
    assert embedding_similarity(a, a) == 1.0


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
