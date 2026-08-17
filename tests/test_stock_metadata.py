"""Generated submission metadata: relevant, ordered, capped, and never burned in."""

import csv

import pytest

from photoai import stock_metadata as sm
from photoai.scoring import Semantic


def semantic(**kwargs):
    base = {
        "present": True,
        "genre": "landscape",
        "description": "terraced rice fields under morning fog",
        "concepts": ["tourism", "agriculture"],
        "secondary_genres": ["travel"],
        }
    return Semantic(**{**base, **kwargs})


def generated(**kwargs):
    return sm.generate(semantic=semantic(**kwargs), route="commercial")


# --- keywords ---------------------------------------------------------------


def test_keywords_come_from_what_was_actually_described():
    keywords, _ = sm.build_keywords(
        description="a wooden fishing boat on still water",
        genre="landscape",
        concepts=["tourism"],
        secondary_genres=[],
    )
    assert "wooden" in keywords and "fishing" in keywords and "boat" in keywords


def test_keywords_are_capped_well_below_the_platform_maximum():
    """Irrelevant keywords are treated as spam at account level, not per file."""
    keywords, _ = sm.build_keywords(
        description=" ".join(f"word{i}" for i in range(200)),
        genre="landscape",
        concepts=[],
        secondary_genres=[],
    )
    assert len(keywords) <= sm.MAX_KEYWORDS


def test_the_genre_leads_the_keyword_list():
    """The first keywords carry the search weight at every platform covered."""
    keywords, _ = sm.build_keywords(
        description="fog over terraces", genre="landscape", concepts=[], secondary_genres=[]
    )
    assert keywords[0] == "landscape"


def test_keywords_are_not_repeated():
    keywords, _ = sm.build_keywords(
        description="boat boat boat", genre="landscape", concepts=["boat"], secondary_genres=[]
    )
    assert len(keywords) == len(set(keywords))


def test_stopwords_are_not_keywords():
    keywords, _ = sm.build_keywords(
        description="a boat on the water with the fog",
        genre="landscape", concepts=[], secondary_genres=[],
    )
    assert "the" not in keywords and "with" not in keywords


def test_very_short_words_are_not_keywords():
    keywords, _ = sm.build_keywords(
        description="an ox in a bay", genre="landscape", concepts=[], secondary_genres=[]
    )
    assert all(len(k) >= sm.MIN_KEYWORD_CHARS for k in keywords)


def test_keyword_confidence_reports_how_much_was_evidenced():
    """A thin result should be visible rather than padded out."""
    _, evidenced = sm.build_keywords(
        description="fog over terraces at dawn", genre="landscape",
        concepts=["tourism"], secondary_genres=[],
    )
    _, guessed = sm.build_keywords(
        description="", genre="landscape", concepts=[], secondary_genres=[],
        camera_keywords=["panasonic", "wide angle"],
    )
    assert evidenced > guessed


def test_no_description_still_produces_a_usable_genre_keyword():
    keywords, _ = sm.build_keywords(
        description="", genre="street", concepts=[], secondary_genres=[]
    )
    assert keywords == ["city and urban"]


# --- taxonomy ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("internal", "buyer_facing"),
    [
        ("street", "city and urban"),
        ("portrait", "people"),
        ("reportage", "documentary"),
        ("landscape", "landscape"),
        ("architecture", "architecture"),
        ("unknown-genre", "lifestyle"),
    ],
)
def test_the_ranking_vocabulary_maps_onto_the_buyer_vocabulary(internal, buyer_facing):
    """'detail' is a way of shooting; 'product and still life' is a way of selling."""
    assert sm.to_taxonomy(internal) == buyer_facing


def test_every_mapped_category_exists_in_the_published_taxonomy():
    assert set(sm.GENRE_ALIASES.values()) <= set(sm.GENRES)


# --- titles and descriptions ------------------------------------------------


def test_a_title_reads_like_a_caption_not_a_filename():
    title = sm.build_title("terraced rice fields under fog", "landscape")
    assert title.startswith("Terraced")
    assert ".jpg" not in title


def test_a_title_is_capped():
    assert len(sm.build_title("x" * 500, "landscape")) <= sm.MAX_TITLE_CHARS


def test_a_location_is_added_to_the_title_when_known():
    assert "Hanoi" in sm.build_title("a busy street", "street", location="Hanoi")


def test_a_location_is_not_duplicated_in_the_title():
    title = sm.build_title("a busy street in Hanoi", "street", location="Hanoi")
    assert title.lower().count("hanoi") == 1


def test_a_description_is_capped():
    assert len(sm.build_description("x" * 500, [], "commercial")) <= sm.MAX_DESCRIPTION_CHARS + 1


# --- assembly ---------------------------------------------------------------


def test_generated_metadata_is_complete_enough_to_submit():
    assert generated().is_complete


def test_metadata_without_a_description_is_not_complete():
    assert not sm.generate(semantic=Semantic(), route="commercial").is_complete


def test_the_ai_label_is_carried_through():
    meta = sm.generate(semantic=semantic(), route="commercial", provenance_label="Generative AI")
    assert meta.ai_label == "Generative AI"


def test_a_location_only_comes_from_something_recorded():
    """Coordinates are not a place name and must not be invented into one."""
    meta = sm.generate(
        semantic=semantic(), route="commercial", exif={"gps_lat": 21.02, "gps_lon": 105.83}
    )
    assert meta.location == ""


def test_a_recorded_location_is_used():
    meta = sm.generate(semantic=semantic(), route="commercial", exif={"location": "Sa Pa"})
    assert meta.location == "Sa Pa"


def test_focal_length_becomes_an_honest_keyword():
    meta = sm.generate(semantic=semantic(), route="commercial", exif={"focal_length": 200.0})
    assert "telephoto" in meta.keywords


# --- user edits -------------------------------------------------------------


def test_a_user_edit_wins_over_the_generated_value():
    meta = generated()
    edited = sm.apply_edits(meta, {"title": "My own title"})
    assert edited.title == "My own title"
    assert edited.edited_by_user


def test_an_empty_edit_leaves_the_metadata_untouched():
    meta = generated()
    assert not sm.apply_edits(meta, {}).edited_by_user


def test_the_edited_flag_cannot_be_faked_through_the_edit_payload():
    meta = sm.apply_edits(generated(), {"edited_by_user": False, "title": "x"})
    assert meta.edited_by_user


def test_unknown_edit_keys_are_ignored():
    meta = sm.apply_edits(generated(), {"not_a_field": "x"})
    assert not hasattr(meta, "not_a_field")


# --- writing ----------------------------------------------------------------


def test_the_xmp_sidecar_sits_beside_the_export_and_not_inside_the_original(tmp_path):
    """An original with metadata burned in is no longer what the camera produced."""
    target = tmp_path / "export" / "P1042675.jpg"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"jpeg bytes")

    path = sm.write_xmp_sidecar(generated(), target)

    assert path.name == "P1042675.xmp"
    assert target.read_bytes() == b"jpeg bytes"


def test_the_sidecar_contains_the_keywords(tmp_path):
    meta = generated()
    path = sm.write_xmp_sidecar(meta, tmp_path / "a.jpg")
    body = path.read_text(encoding="utf-8")
    assert meta.keywords[0] in body
    assert "<dc:subject>" in body


def test_the_sidecar_escapes_hostile_metadata(tmp_path):
    meta = generated()
    meta.title = 'Bad <tag> & "quote"'
    body = sm.write_xmp_sidecar(meta, tmp_path / "a.jpg").read_text(encoding="utf-8")
    assert "<tag>" not in body
    assert "&lt;tag&gt;" in body


def test_the_submission_csv_has_one_row_per_asset(tmp_path):
    rows = [("a.jpg", generated()), ("b.jpg", generated())]
    path = sm.write_submission_csv(rows, tmp_path / "submission.csv")
    with open(path, newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert [r["filename"] for r in written] == ["a.jpg", "b.jpg"]
    assert written[0]["title"]
    assert written[0]["keywords"]


# --- privacy ----------------------------------------------------------------


def test_coordinates_are_stripped_from_export_copies():
    """A home, a school, a route walked daily -- private by default."""
    stripped = sm.strip_gps({"gps_lat": 21.0, "gps_lon": 105.0, "camera_make": "Panasonic"})
    assert "gps_lat" not in stripped and "gps_lon" not in stripped
    assert stripped["camera_make"] == "Panasonic"


def test_stripping_gps_does_not_mutate_the_original_dict():
    exif = {"gps_lat": 21.0}
    sm.strip_gps(exif)
    assert "gps_lat" in exif


def test_the_submission_csv_has_no_release_columns():
    """Five tests lived here, filling release columns from a model's guess."""
    for gone in ("model_release_required", "property_release_required",
                 "logo_warning", "recognizable_person"):
        assert gone not in sm.SUBMISSION_CSV_FIELDS
        assert not hasattr(generated(), gone)
