import pytest

from routing import (
    AXIS_MAX,
    Assessment,
    AssessmentParseError,
    Destination,
    Genre,
    Recover,
    RoutingConfig,
    assign_destinations,
    blocks_commercial_stock,
    parse_assessment,
    summarise,
)


def make(
    filename="P1.RW2",
    genre=Genre.LANDSCAPE,
    axis_a=50,
    axis_b=50,
    axis_c=50,
    recover=Recover.EASY,
    faces=False,
    logos=False,
    is_video=False,
    rejected=None,
    **kw,
):
    return Assessment(
        filename=filename,
        genre=genre,
        axis_a=axis_a,
        axis_b=axis_b,
        axis_c=axis_c,
        recover=recover,
        faces=faces,
        logos=logos,
        is_video=is_video,
        technically_rejected_for=list(rejected or []),
        **kw,
    )


def route_one(assessment, config=None):
    return assign_destinations([assessment], config)[0]


# --- THE BLOCKING RULE ------------------------------------------------------
# A release is legally required for a recognisable face or a logo. Getting this
# wrong once costs a batch of stock rejections, so it is tested hardest.


@pytest.mark.parametrize(
    ("faces", "logos"),
    [(True, False), (False, True), (True, True)],
    ids=["face", "logo", "both"],
)
def test_a_release_requiring_frame_never_reaches_commercial_stock(faces, logos):
    routed = route_one(make(axis_a=AXIS_MAX, faces=faces, logos=logos))
    assert routed.destination is not Destination.STOCK_COMMERCIAL
    assert routed.destination is Destination.EDITORIAL


def test_the_block_holds_at_a_perfect_commercial_score():
    routed = route_one(make(axis_a=100, axis_c=0, faces=True))
    assert routed.destination is Destination.EDITORIAL
    assert "release required" in routed.reason


def test_the_block_overrides_what_the_model_asked_for():
    """The model's own destination field is advisory and must not be trusted."""
    routed = route_one(
        make(axis_a=95, faces=True, model_destination="10_stock_commercial")
    )
    assert routed.destination is Destination.EDITORIAL
    assert routed.assessment.model_destination == "10_stock_commercial"


def test_a_release_requiring_frame_below_every_threshold_still_avoids_stock():
    routed = route_one(make(axis_a=0, axis_b=0, axis_c=0, logos=True))
    assert routed.destination is Destination.EDITORIAL


def test_no_frame_with_faces_or_logos_lands_in_stock_across_a_whole_population():
    """Population-level guarantee, including frames picked as flagship."""
    population = []
    for i in range(60):
        population.append(
            make(
                filename=f"F{i}.RW2",
                genre=list(Genre)[i % len(Genre)],
                axis_a=i % 101,
                axis_b=(i * 7) % 101,
                axis_c=(i * 3) % 101,
                faces=(i % 2 == 0),
                logos=(i % 3 == 0),
            )
        )
    for routed in assign_destinations(population):
        if routed.destination is Destination.STOCK_COMMERCIAL:
            assert not blocks_commercial_stock(routed.assessment)


def test_a_clean_frame_does_reach_stock():
    """The block must not be so broad that nothing is ever sellable."""
    routed = route_one(make(axis_a=AXIS_MAX, faces=False, logos=False))
    assert routed.destination is Destination.STOCK_COMMERCIAL


@pytest.mark.parametrize(
    ("faces", "logos", "expected"),
    [(False, False, False), (True, False, True), (False, True, True), (True, True, True)],
)
def test_blocks_commercial_stock(faces, logos, expected):
    assert blocks_commercial_stock(make(faces=faces, logos=logos)) is expected


# --- delete candidates ------------------------------------------------------


def test_a_technically_rejected_frame_becomes_a_delete_candidate():
    routed = route_one(make(axis_a=100, rejected=["out of focus (blur ratio 1.05 < 2.00)"]))
    assert routed.destination is Destination.DELETE_CANDIDATES
    assert "blur ratio" in routed.reason


def test_hopeless_recovery_becomes_a_delete_candidate():
    routed = route_one(make(axis_a=100, axis_b=100, recover=Recover.HOPELESS))
    assert routed.destination is Destination.DELETE_CANDIDATES


def test_a_technical_reject_outranks_every_other_signal():
    routed = route_one(
        make(axis_a=100, axis_b=100, axis_c=100, faces=True, rejected=["blown highlights (80%)"])
    )
    assert routed.destination is Destination.DELETE_CANDIDATES


def test_a_hopeless_frame_is_never_chosen_as_flagship():
    population = [make(filename=f"F{i}.RW2", axis_b=100, recover=Recover.HOPELESS) for i in range(5)]
    assert all(r.destination is Destination.DELETE_CANDIDATES for r in assign_destinations(population))


# --- video ------------------------------------------------------------------


def test_video_goes_to_the_video_bucket():
    assert route_one(make(is_video=True)).destination is Destination.VIDEO_STOCK


def test_video_bypasses_the_axis_thresholds():
    routed = route_one(make(is_video=True, axis_a=0, axis_b=0, axis_c=0))
    assert routed.destination is Destination.VIDEO_STOCK


def test_an_unusable_video_is_still_a_delete_candidate():
    routed = route_one(make(is_video=True, rejected=["out of focus"]))
    assert routed.destination is Destination.DELETE_CANDIDATES


def test_video_is_never_picked_as_flagship():
    population = [make(filename=f"V{i}.MOV", is_video=True, axis_b=100) for i in range(5)]
    assert all(r.destination is Destination.VIDEO_STOCK for r in assign_destinations(population))


# --- flagship: ranked inside its genre --------------------------------------


def genre_population(genre, count, start_b=0):
    return [
        make(filename=f"{genre.value}{i}.RW2", genre=genre, axis_b=start_b + i)
        for i in range(count)
    ]


def test_flagship_is_the_top_of_its_own_genre_not_the_global_top():
    """Street must not be crowded out by a tidier genre."""
    landscape = genre_population(Genre.LANDSCAPE, 20, start_b=80)  # b 80..99
    street = genre_population(Genre.STREET, 20, start_b=61)  # b 61..80
    routed = assign_destinations(landscape + street)

    picked = {r.filename for r in routed if r.destination is Destination.FLAGSHIP}
    assert any(f.startswith("street") for f in picked), "street got no flagship slot"
    assert any(f.startswith("landscape") for f in picked)


def test_flagship_picks_the_highest_axis_b_in_the_genre():
    population = genre_population(Genre.NIGHT, 30, start_b=70)  # b 70..99
    routed = assign_destinations(population)
    picked = [r for r in routed if r.destination is Destination.FLAGSHIP]
    assert picked
    best = max(population, key=lambda a: a.axis_b).filename
    assert best in {r.filename for r in picked}


def test_flagship_respects_a_minimum_axis_b():
    """A weak genre should not get a flagship slot just for existing."""
    config = RoutingConfig(flagship_min_axis_b=60)
    population = genre_population(Genre.DETAIL, 20, start_b=0)  # all well below 60
    routed = assign_destinations(population, config)
    assert not any(r.destination is Destination.FLAGSHIP for r in routed)


def test_flagship_quota_scales_with_genre_size():
    config = RoutingConfig(flagship_top_fraction=0.10, flagship_min_axis_b=0)
    routed = assign_destinations(genre_population(Genre.PORTRAIT, 100, start_b=0), config)
    picked = sum(1 for r in routed if r.destination is Destination.FLAGSHIP)
    assert picked == 10


def test_flagship_quota_is_capped():
    config = RoutingConfig(flagship_top_fraction=0.50, flagship_max_per_genre=5, flagship_min_axis_b=0)
    routed = assign_destinations(genre_population(Genre.STREET, 100, start_b=0), config)
    assert sum(1 for r in routed if r.destination is Destination.FLAGSHIP) == 5


def test_flagship_outranks_commercial_stock():
    config = RoutingConfig(flagship_min_axis_b=0, axis_a_stock=10)
    routed = assign_destinations([make(axis_a=100, axis_b=100)], config)
    assert routed[0].destination is Destination.FLAGSHIP


def test_flagship_may_contain_faces():
    """Only commercial stock is release-blocked; flagship is not."""
    config = RoutingConfig(flagship_min_axis_b=0)
    routed = assign_destinations([make(axis_b=100, faces=True)], config)
    assert routed[0].destination is Destination.FLAGSHIP


# --- the value buckets ------------------------------------------------------


def test_high_documentary_value_goes_editorial():
    config = RoutingConfig(axis_a_stock=70, axis_c_editorial=65)
    routed = route_one(make(axis_a=10, axis_c=90), config)
    assert routed.destination is Destination.EDITORIAL


def test_everything_unremarkable_is_held_not_deleted():
    routed = route_one(make(axis_a=10, axis_b=10, axis_c=10))
    assert routed.destination is Destination.HOLD


def test_thresholds_are_inclusive_at_the_boundary():
    config = RoutingConfig(axis_a_stock=70, flagship_min_axis_b=101)
    assert route_one(make(axis_a=70), config).destination is Destination.STOCK_COMMERCIAL
    assert route_one(make(axis_a=69, axis_c=0), config).destination is Destination.HOLD


def test_thresholds_come_from_config_not_from_constants():
    """--bench replaces these, so they must be swappable without a code change."""
    strict = RoutingConfig(axis_a_stock=95, flagship_min_axis_b=101)
    loose = RoutingConfig(axis_a_stock=20, flagship_min_axis_b=101)
    frame = make(axis_a=50, axis_c=0)
    assert route_one(frame, strict).destination is Destination.HOLD
    assert route_one(frame, loose).destination is Destination.STOCK_COMMERCIAL


def test_the_three_axes_are_never_averaged():
    """A frame strong on one axis alone must still route somewhere real."""
    config = RoutingConfig(axis_a_stock=70, axis_c_editorial=65, flagship_min_axis_b=101)
    assert route_one(make(axis_a=90, axis_b=0, axis_c=0), config).destination is (
        Destination.STOCK_COMMERCIAL
    )
    assert route_one(make(axis_a=0, axis_b=0, axis_c=90), config).destination is (
        Destination.EDITORIAL
    )


# --- population invariants --------------------------------------------------


def test_every_frame_gets_exactly_one_destination():
    population = [make(filename=f"F{i}.RW2", axis_a=i, axis_b=i, axis_c=i) for i in range(50)]
    routed = assign_destinations(population)
    assert len(routed) == len(population)
    assert [r.filename for r in routed] == [a.filename for a in population]


def test_summarise_counts_every_bucket():
    population = [make(filename="a.RW2", axis_a=100), make(filename="b.RW2", faces=True)]
    counts = summarise(assign_destinations(population))
    assert set(counts) == {d.value for d in Destination}
    assert sum(counts.values()) == 2


# --- parsing ----------------------------------------------------------------


VALID = {
    "genre": "street",
    "axis_a": 72,
    "axis_b": 40,
    "axis_c": 55,
    "recover": "easy",
    "faces": False,
    "logos": False,
    "note": "lift shadows, crop left edge",
}


def test_parses_a_well_formed_payload():
    a = parse_assessment(VALID, "P1.RW2")
    assert a.genre is Genre.STREET
    assert (a.axis_a, a.axis_b, a.axis_c) == (72, 40, 55)
    assert a.recover is Recover.EASY
    assert a.faces is False


@pytest.mark.parametrize("key", sorted({"genre", "axis_a", "axis_b", "axis_c", "recover", "faces", "logos"}))
def test_a_missing_required_key_is_an_error(key):
    payload = {k: v for k, v in VALID.items() if k != key}
    with pytest.raises(AssessmentParseError, match=key):
        parse_assessment(payload, "P1.RW2")


def test_an_unknown_genre_falls_back_rather_than_failing_the_frame():
    a = parse_assessment({**VALID, "genre": "macro-ish"}, "P1.RW2")
    assert a.genre is Genre.OTHER


@pytest.mark.parametrize(
    ("raw", "expected"), [(150, 100), (-20, 0), ("83", 83), (72.6, 73), (None, 0), ("x", 0)]
)
def test_axis_values_are_clamped_into_range(raw, expected):
    assert parse_assessment({**VALID, "axis_a": raw}, "P1.RW2").axis_a == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("true", True), ("false", False), ("no", False), (None, True)],
)
def test_release_flags_fail_safe_when_unclear(raw, expected):
    """Guessing 'no face' on an uncertain frame is the expensive way to be wrong."""
    assert parse_assessment({**VALID, "faces": raw}, "P1.RW2").faces is expected


def test_an_unparseable_face_flag_blocks_stock():
    a = parse_assessment({**VALID, "axis_a": 100, "faces": None}, "P1.RW2")
    assert assign_destinations([a])[0].destination is not Destination.STOCK_COMMERCIAL


def test_the_note_is_trimmed_to_twelve_words():
    long_note = " ".join(f"word{i}" for i in range(30))
    assert len(parse_assessment({**VALID, "note": long_note}, "P1.RW2").note.split()) == 12


def test_a_missing_note_is_empty_not_an_error():
    payload = {k: v for k, v in VALID.items() if k != "note"}
    assert parse_assessment(payload, "P1.RW2").note == ""


def test_stage_0_rejections_ride_along_into_the_assessment():
    a = parse_assessment(VALID, "P1.RW2", technically_rejected_for=["blown highlights (80%)"])
    assert a.technically_rejected
    assert assign_destinations([a])[0].destination is Destination.DELETE_CANDIDATES
