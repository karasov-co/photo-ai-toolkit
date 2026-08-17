import math

import pytest

from photoai.aggregate import (
    aggregate_all_axes,
    aggregate_axis,
    bradley_terry,
    build_groups,
    ranks_to_pairs,
    spearman,
    to_percentile_scores,
)

# --- grouping ---------------------------------------------------------------


def test_groups_are_the_requested_size():
    groups = build_groups([f"f{i}" for i in range(40)], size=12, overlap=3)
    assert all(len(g) <= 12 for g in groups)
    assert groups[0] == [f"f{i}" for i in range(12)]


def test_consecutive_groups_overlap_so_they_can_be_compared():
    """Without shared frames each group is an island and cannot be stitched."""
    groups = build_groups([f"f{i}" for i in range(40)], size=12, overlap=3)
    for earlier, later in zip(groups, groups[1:], strict=False):
        assert set(earlier) & set(later), "groups must share frames"


def test_every_frame_appears_at_least_once():
    names = [f"f{i}" for i in range(37)]
    seen = {n for g in build_groups(names, size=12, overlap=3) for n in g}
    assert seen == set(names)


def test_a_short_archive_is_a_single_group():
    assert build_groups(["a", "b", "c"], size=12) == [["a", "b", "c"]]


def test_no_group_is_left_with_a_single_frame():
    """A group of one produces no comparisons and would strand that frame."""
    for total in range(2, 40):
        groups = build_groups([f"f{i}" for i in range(total)], size=12, overlap=3)
        assert all(len(g) >= 2 for g in groups), f"total={total}"


def test_an_empty_archive_produces_no_groups():
    assert build_groups([], size=12) == []


def test_a_degenerate_group_size_is_rejected():
    with pytest.raises(ValueError):
        build_groups(["a", "b"], size=1)


# --- ranks to pairs ---------------------------------------------------------


def test_a_ranking_becomes_every_ordered_pair():
    pairs = ranks_to_pairs([("a", 1), ("b", 2), ("c", 3)])
    assert set(pairs) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_tied_ranks_produce_no_comparison():
    assert ranks_to_pairs([("a", 1), ("b", 1)]) == []


# --- Bradley-Terry ----------------------------------------------------------


def test_a_consistent_winner_gets_the_highest_strength():
    strength = bradley_terry([("a", "b"), ("a", "c"), ("b", "c")])
    assert strength["a"] > strength["b"] > strength["c"]


def test_no_comparisons_gives_no_strengths():
    assert bradley_terry([]) == {}


def test_an_undefeated_item_stays_finite():
    """The smoothing prior is what stops this running off to infinity."""
    strength = bradley_terry([("a", "b")] * 50)
    assert math.isfinite(strength["a"])
    assert strength["a"] > strength["b"]


def test_transitivity_is_recovered_across_groups_that_only_overlap():
    """a beats b in one group, b beats c in another: a must still outrank c.

    a and c are never compared directly -- this is the whole reason the groups
    are built to overlap.
    """
    pairs = ranks_to_pairs([("a", 1), ("b", 2)]) + ranks_to_pairs([("b", 1), ("c", 2)])
    strength = bradley_terry(pairs)
    assert strength["a"] > strength["c"]


# --- scores -----------------------------------------------------------------


def test_scores_span_the_full_range():
    scores = to_percentile_scores({"a": 3.0, "b": 2.0, "c": 1.0})
    assert scores["a"] == 100
    assert scores["c"] == 0
    assert 0 < scores["b"] < 100


def test_a_single_frame_scores_top():
    assert to_percentile_scores({"only": 1.0}) == {"only": 100}


def test_empty_input_gives_empty_scores():
    assert to_percentile_scores({}) == {}


def test_scores_are_flat_not_clustered():
    """The collapse that absolute scoring produced must not reappear here."""
    strength = {f"f{i}": 1.0 + i * 0.001 for i in range(50)}
    scores = sorted(to_percentile_scores(strength).values())
    assert scores[0] == 0 and scores[-1] == 100
    # A flat distribution puts roughly a quarter of frames under 25.
    assert 8 <= sum(1 for s in scores if s < 25) <= 18


# --- end to end -------------------------------------------------------------


def test_a_known_order_is_recovered_from_overlapping_groups():
    """Ground truth f0 > f1 > ... > f19, seen only 12 at a time."""
    truth = {f"f{i}": 20 - i for i in range(20)}
    groups = build_groups(list(truth), size=12, overlap=4)
    rankings = [
        sorted(g, key=lambda n: -truth[n])  # rank within the group by truth
        for g in groups
    ]
    group_rankings = [[(n, i + 1) for i, n in enumerate(r)] for r in rankings]

    scores = aggregate_axis(group_rankings)
    assert spearman(scores, truth) > 0.95


def test_the_three_axes_stay_independent():
    """A frame can top one axis and bottom another; averaging would hide that."""
    group = [
        {"filename": "unique", "axis_a": 3, "axis_b": 1, "axis_c": 2},
        {"filename": "clean", "axis_a": 1, "axis_b": 3, "axis_c": 3},
        {"filename": "middle", "axis_a": 2, "axis_b": 2, "axis_c": 1},
    ]
    scores = aggregate_all_axes([group])
    assert scores["axis_a"]["clean"] > scores["axis_a"]["unique"]
    assert scores["axis_b"]["unique"] > scores["axis_b"]["clean"]


def test_aggregation_survives_a_group_the_model_answered_partially():
    groups = [
        [{"filename": "a", "axis_a": 1}, {"filename": "b", "axis_a": 2}],
        [{"filename": "b"}],  # model dropped the axis
    ]
    scores = aggregate_all_axes(groups)
    assert scores["axis_a"]["a"] > scores["axis_a"]["b"]


def test_aggregation_of_nothing_is_empty_not_an_error():
    assert aggregate_all_axes([]) == {"axis_a": {}, "axis_b": {}, "axis_c": {}}


# --- spearman, used by --bench ---------------------------------------------


def test_identical_orders_correlate_perfectly():
    a = {"x": 1, "y": 2, "z": 3}
    assert spearman(a, a) == pytest.approx(1.0)


def test_reversed_orders_correlate_negatively():
    assert spearman({"x": 1, "y": 2, "z": 3}, {"x": 3, "y": 2, "z": 1}) == pytest.approx(-1.0)


def test_spearman_only_uses_shared_keys():
    assert spearman({"x": 1, "y": 2, "gone": 9}, {"x": 1, "y": 2}) == pytest.approx(1.0)


def test_spearman_handles_ties_without_blowing_up():
    assert spearman({"a": 1, "b": 1, "c": 2}, {"a": 1, "b": 2, "c": 3}) == pytest.approx(0.866, abs=0.01)


def test_spearman_is_undefined_for_a_constant_series():
    assert math.isnan(spearman({"a": 1, "b": 1}, {"a": 1, "b": 2}))


def test_spearman_needs_at_least_two_points():
    assert math.isnan(spearman({"a": 1}, {"a": 1}))
