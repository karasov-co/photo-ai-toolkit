"""The default report, the output tree, the edit recipes and the insights page.

These four make up everything a photographer sees. The tests fall into two
groups: what must be on the page, and what must never be. The second group is
the one that decays -- a helper gets reused, a field gets added to a card "just
for debugging", and the vocabulary the report exists to hide is back. So the
prohibitions are asserted against the rendered HTML rather than against the code
that generates it.
"""

import json
from pathlib import Path

import pytest
from synthetic import photo_like, write_jpeg

import curation
import insights as insights_module
import recipe_export
import simple_report
import workspace
from reports import AssetRecord


def record(**overrides) -> AssetRecord:
    payload = {
        "asset_id": "a1",
        "source_path": "/photos/a.jpg",
        "filename": "a.jpg",
        "media_type": "photo",
        "checksum": "abc",
        "asset_key": "a.jpg",
        "category": "GOOD_STOCK",
        "final_score": 72,
        "scores": {"current_quality": 48, "post_edit_potential": 72, "routing_score": 61},
        "final_score_detail": {"stage3_delta": 6, "components": {}},
        "category_reasons": ["final score 72, a photograph worth keeping"],
        "edit_recipe": ["Adjust exposure: +1.2 EV", "Recover highlights: moderate",
                        "Straighten: rotate 2.1 degrees", "Crop: keep 88% of the frame"],
        "route_class": "stock_standard",
        "genre": "landscape",
        "status": "ok",
        "stage3": {"status": "completed", "documentary_significance": 60},
    }
    payload.update(overrides)
    return AssetRecord(**payload)


@pytest.fixture
def collection() -> list[AssetRecord]:
    return [
        record(filename="top.jpg", asset_key="top.jpg", category="TOP", final_score=88,
               scores={"current_quality": 51, "post_edit_potential": 88}),
        record(filename="stock.jpg", asset_key="stock.jpg", category="GOOD_STOCK", final_score=74),
        record(filename="personal.jpg", asset_key="personal.jpg", category="GOOD_PERSONAL",
               final_score=66, commercial_blockers=["a model release is required"]),
        record(filename="weak.jpg", asset_key="weak.jpg", category="WEAK", final_score=32,
               category_reasons=["confident bad expression: eyes closed -- a technical "
                                 "score cannot make up for this"]),
    ]


# --- one ranking, one number --------------------------------------------------


def test_the_page_ranks_by_potential_after_editing(collection, tmp_path):
    ranked = simple_report.ordered(collection)
    assert [r.filename for r in ranked] == ["top.jpg", "stock.jpg", "personal.jpg", "weak.jpg"]


def test_a_dark_file_that_recovers_outranks_a_bright_one_that_does_not():
    """The whole premise, as an ordering.

    The dark frame looks worse right now by 30 points and is the better
    photograph. Ranking by current quality would invert them.
    """
    dark = record(filename="dark.jpg", final_score=81,
                  scores={"current_quality": 34, "post_edit_potential": 81})
    bright = record(filename="bright.jpg", final_score=55,
                    scores={"current_quality": 64, "post_edit_potential": 55})
    assert [r.filename for r in simple_report.ordered([bright, dark])] == ["dark.jpg", "bright.jpg"]


def test_current_quality_is_shown_but_never_ranks(collection, tmp_path):
    page = simple_report.write(collection, tmp_path / "report.html").read_text()
    assert "as shot" in page
    assert "51" in page  # top.jpg's current quality is still visible


def test_every_category_gets_a_section_even_when_empty(collection, tmp_path):
    page = simple_report.write(collection, tmp_path / "report.html").read_text()
    for name, _ in simple_report.SECTIONS:
        assert simple_report.t(f"category.{name}", "en") in page


def test_a_card_carries_a_reason_and_a_few_things_to_do(collection):
    steps = simple_report.recommendations(collection[0])
    assert 1 <= len(steps) <= simple_report.MAX_RECOMMENDATIONS
    assert simple_report.explanation(collection[0])


def test_recommendations_are_capped_at_three():
    assert len(simple_report.recommendations(record())) == 3


# --- what must never be on the page -------------------------------------------


@pytest.mark.parametrize("phrase", simple_report.FORBIDDEN_IN_DEFAULT_UI)
def test_no_internal_or_legal_vocabulary_reaches_the_default_report(collection, tmp_path, phrase):
    page = simple_report.write(collection, tmp_path / "report.html", expert=False).read_text()
    assert phrase not in page.lower()


def test_a_reason_carrying_legal_language_is_dropped_rather_than_shown():
    """A good family photograph must never be told it has a licensing problem."""
    personal = record(
        category="GOOD_PERSONAL",
        category_reasons=[
            "final score 66: worth keeping. Not for stock -- a model release is required"
        ],
        reasons=["faces present: a release is required, so commercial stock is blocked"],
    )
    text = simple_report.explanation(personal)
    assert "release" not in text.lower()
    assert text


def test_the_expert_block_is_where_the_hidden_numbers_live(collection, tmp_path):
    """Hidden is not discarded: somebody who wants the detail can still get it."""
    page = simple_report.write(collection, tmp_path / "report.html", expert=True).read_text()
    assert "Expert details" in page
    assert "stock_standard" in page  # the route class, in the expert table only
    body, expert = page.split("<details", 1)
    assert "stock_standard" not in body
    assert "stock_standard" in expert


def test_the_score_shown_is_the_potential_not_the_routing_score(collection, tmp_path):
    page = simple_report.write(collection, tmp_path / "report.html", expert=False).read_text()
    assert ">88" in page.replace("\n", "")
    assert "61" not in page.replace("\n", "")  # routing_score never appears


# --- the output tree ----------------------------------------------------------


def test_the_root_holds_only_what_a_photographer_opens(tmp_path):
    space = workspace.Workspace(tmp_path / "out").create()
    (space.reports / "analysis.json").write_text("{}")
    space.report.write_text("x")
    space.insights.write_text("x")

    visible = {p.name for p in space.root.iterdir() if not p.name.startswith(".")}
    assert visible == {
        "report.html", "photographer_insights.html", "edit_recipes",
        "top", "good_stock", "good_personal", "needs_decision", "weak",
    }


def test_everything_else_is_hidden_but_present(tmp_path):
    space = workspace.Workspace(tmp_path / "out").create()
    assert space.internal.name.startswith(".")
    assert space.previews.is_dir()
    assert space.reports.is_dir()
    assert space.internal in space.cache.parents
    assert space.internal in space.log.parents


def test_an_older_output_directory_is_tidied_on_the_next_run(tmp_path):
    out = tmp_path / "out"
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "analysis.json").write_text('{"assets": []}')
    (out / "previews").mkdir()
    (out / "processing.log").write_text("old")
    (out / "portfolio" / "flagship").mkdir(parents=True)

    moved = workspace.migrate(out)

    assert set(moved) >= {"reports", "previews", "processing.log", "portfolio"}
    assert (out / ".internal" / "reports" / "analysis.json").exists()
    assert not (out / "reports").exists()


def test_migration_never_destroys_a_file_it_cannot_place(tmp_path):
    out = tmp_path / "out"
    (out / ".internal" / "reports").mkdir(parents=True)
    (out / ".internal" / "reports" / "analysis.json").write_text("the newer one")
    (out / "reports").mkdir()
    (out / "reports" / "analysis.json").write_text("the older one")

    workspace.migrate(out)

    assert (out / ".internal" / "reports" / "analysis.json").read_text() == "the newer one"
    assert (out / ".internal" / "reports-1" / "analysis.json").read_text() == "the older one"


def test_migrating_a_directory_that_does_not_exist_is_harmless(tmp_path):
    assert workspace.migrate(tmp_path / "nothing") == []


def test_the_category_farm_is_links_only(tmp_path):
    source = tmp_path / "photos" / "a.jpg"
    write_jpeg(photo_like(80, 60), source)
    before = source.read_bytes()

    space = workspace.Workspace(tmp_path / "out").create()
    counts = workspace.build_category_farm(
        [record(source_path=str(source), category="TOP")], space
    )

    link = space.root / "top" / "a.jpg"
    assert link.is_symlink()
    assert counts["TOP"] == 1
    assert source.read_bytes() == before, "the original must not be touched"


def test_the_farm_is_rebuilt_rather_than_accumulated(tmp_path):
    source = tmp_path / "photos" / "a.jpg"
    write_jpeg(photo_like(80, 60), source)
    space = workspace.Workspace(tmp_path / "out").create()

    workspace.build_category_farm([record(source_path=str(source), category="TOP")], space)
    workspace.build_category_farm([record(source_path=str(source), category="WEAK")], space)

    assert not list((space.root / "top").iterdir())
    assert (space.root / "weak" / "a.jpg").is_symlink()


# --- edit recipes -------------------------------------------------------------


class FakeMeasurement:
    mean_luma = 96.0
    stddev_luma = 40.0
    channel_means = (128.0, 120.0, 112.0)
    noise = 3.0
    recipe = ["Straighten: rotate 2.4 degrees clockwise"]


def test_a_recipe_is_written_for_anything_worth_editing(tmp_path):
    source = tmp_path / "a.jpg"
    write_jpeg(photo_like(200, 150), source)
    subject = record(source_path=str(source), final_score=recipe_export.MIN_POTENTIAL)

    result = recipe_export.export_all(
        [subject], {"a.jpg": FakeMeasurement()}, tmp_path / "edit_recipes"
    )

    assert result["written"] == ["a.jpg"]
    assert (tmp_path / "edit_recipes" / "a.xmp").exists()
    assert subject.recipe_path


def test_nothing_is_written_below_the_threshold(tmp_path):
    subject = record(final_score=recipe_export.MIN_POTENTIAL - 1)
    result = recipe_export.export_all([subject], {"a.jpg": FakeMeasurement()}, tmp_path / "r")
    assert result["written"] == []


def test_a_weak_photograph_gets_no_recipe_however_it_scored():
    assert not recipe_export.should_export(record(category="WEAK", final_score=95))


def test_the_sidecar_is_camera_raw_and_names_itself_a_suggestion(tmp_path):
    source = tmp_path / "a.jpg"
    write_jpeg(photo_like(200, 150), source)
    recipe_export.export_all(
        [record(source_path=str(source), final_score=80)],
        {"a.jpg": FakeMeasurement()},
        tmp_path / "r",
    )
    xmp = (tmp_path / "r" / "a.xmp").read_text()
    assert "crs:Exposure2012" in xmp
    assert "photo-ai-toolkit" in xmp


def test_the_sidecar_never_lands_beside_the_original(tmp_path):
    """`<stem>.xmp` next to a RAW is what Lightroom overwrites without asking."""
    source = tmp_path / "photos" / "a.jpg"
    write_jpeg(photo_like(200, 150), source)
    recipe_export.export_all(
        [record(source_path=str(source), final_score=80)],
        {"a.jpg": FakeMeasurement()},
        tmp_path / "out" / "edit_recipes",
    )
    assert not (tmp_path / "photos" / "a.xmp").exists()


def test_the_folder_explains_how_to_use_it(tmp_path):
    source = tmp_path / "a.jpg"
    write_jpeg(photo_like(200, 150), source)
    recipe_export.export_all(
        [record(source_path=str(source), final_score=80)],
        {"a.jpg": FakeMeasurement()},
        tmp_path / "r",
    )
    text = (tmp_path / "r" / recipe_export.README_NAME).read_text()
    assert "STARTING POINT" in text
    assert "Read Metadata from File" in text
    assert "Lightroom Classic" in text


def test_a_creative_direction_has_to_be_earned():
    """A season is not a filter. An unrelated frame is offered nothing seasonal."""
    neutral = record(genre="portrait", stage3={"documentary_significance": 20})

    class Flat:
        mean_luma = 128.0
        stddev_luma = 10.0
        channel_means = (120.0, 120.0, 120.0)

    keys = {d.key for d in recipe_export.directions_for(neutral, Flat())}
    assert "warm_autumn" not in keys
    assert "cool_winter" not in keys


def test_a_warm_landscape_may_be_offered_autumn():
    warm = record(genre="landscape", stage3={"documentary_significance": 30})

    class Warm:
        mean_luma = 130.0
        stddev_luma = 40.0
        channel_means = (150.0, 120.0, 100.0)

    directions = recipe_export.directions_for(warm, Warm())
    assert any(d.key == "warm_autumn" for d in directions)
    assert all(d.because for d in directions)


def test_a_dark_frame_may_be_offered_low_key():
    class Dark:
        mean_luma = 62.0
        stddev_luma = 30.0
        channel_means = (100.0, 100.0, 100.0)

    keys = {d.key for d in recipe_export.directions_for(record(genre="street"), Dark())}
    assert "cinematic_low_key" in keys


def test_never_more_than_three_directions():
    class Anything:
        mean_luma = 70.0
        stddev_luma = 45.0
        channel_means = (150.0, 120.0, 100.0)

    subject = record(genre="street", stage3={"documentary_significance": 95})
    assert len(recipe_export.directions_for(subject, Anything())) <= 3


# --- photographer insights ----------------------------------------------------


def sample(n: int = 12) -> list[AssetRecord]:
    out = []
    for i in range(n):
        out.append(
            record(
                filename=f"f{i}.jpg",
                asset_key=f"f{i}.jpg",
                final_score=60 + i,
                genre="landscape" if i % 2 else "portrait",
                best_in_cluster=i % 4 != 0,
                expected_gain=9,
                scores={"current_quality": 40, "post_edit_potential": 60 + i,
                        "recoverability": 88},
                stage3={
                    "status": "completed",
                    "emotional_resonance": 70,
                    "moment_specificity": 35,
                    "distinctiveness": 30,
                    "documentary_significance": 55,
                    "visual_tension": 50,
                    "narrative_openness": 50,
                    "formal_coherence": 66,
                },
            )
        )
    return out


def test_insights_name_the_numbers_behind_every_claim():
    built = insights_module.build(sample())
    everything = (
        built.visual_habits + built.technical_strengths
        + built.artistic_strengths + built.weaknesses + built.improvements
    )
    assert everything
    for observation in everything:
        assert observation.evidence, observation.text


def test_insights_cite_actual_files():
    built = insights_module.build(sample())
    cited = {name for o in built.weaknesses for name in o.examples}
    assert cited
    assert all(name.endswith(".jpg") for name in cited)


def test_at_most_three_improvements():
    assert len(insights_module.build(sample()).improvements) <= 3


def test_an_improvement_is_specific_rather_than_a_platitude():
    built = insights_module.build(sample())
    text = " ".join(o.text.lower() for o in built.improvements)
    for platitude in ("rule of thirds", "improve your composition", "practice more"):
        assert platitude not in text


def test_a_pattern_needs_enough_frames_behind_it():
    """Two frames is a coincidence. The threshold is why it is not reported."""
    built = insights_module.build(sample(2))
    assert not built.visual_habits


def test_genres_are_ranked_by_result_not_by_count():
    records = [
        record(filename=f"l{i}.jpg", genre="landscape", final_score=40) for i in range(8)
    ] + [record(filename=f"p{i}.jpg", genre="portrait", final_score=85) for i in range(3)]
    built = insights_module.build(records)
    assert built.genres[0][0] == "portrait"


def test_inspiration_is_a_fixed_table_with_no_invented_claims():
    built = insights_module.build(sample())
    for group in built.inspiration:
        assert group["genre"] in insights_module.INSPIRATION
        for entry in group["entries"]:
            assert entry in insights_module.INSPIRATION[group["genre"]]
            assert '"' not in entry["note"], "a quotation would be an invented fact"


def test_no_inspiration_for_a_genre_nobody_shot():
    built = insights_module.build([record(genre="unknown") for _ in range(6)])
    assert built.inspiration == []


def test_strengths_are_reported_as_well_as_faults():
    built = insights_module.build(sample())
    assert built.technical_strengths or built.artistic_strengths


def test_an_empty_collection_produces_an_empty_page(tmp_path):
    built = insights_module.build([])
    page = insights_module.write(built, tmp_path / "insights.html")
    assert page.exists()
    assert built.total == 0


def test_the_insights_page_renders_every_section(tmp_path):
    built = insights_module.build(sample())
    page = insights_module.write(built, tmp_path / "insights.html").read_text()
    assert "What you shoot best" in page
    assert "The three things worth changing next" in page
    assert "Worth looking at" in page


def test_insights_round_trip_through_json(tmp_path):
    built = insights_module.build(sample())
    restored = json.loads(json.dumps(built.to_dict()))
    assert restored["total"] == built.total
    assert restored["improvements"][0]["text"] == built.improvements[0].text


# --- the whole command --------------------------------------------------------


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / "archive"
    for i in range(3):
        write_jpeg(photo_like(1400, 1000, seed=i + 1), root / f"p{i}.jpg")
    return root


def test_the_default_run_produces_the_simplified_tree(archive, tmp_path):
    import cli

    out = tmp_path / "out"
    assert cli.main(["analyze", "--input", str(archive), "--output", str(out)]) == 0

    assert (out / "report.html").exists()
    assert (out / "photographer_insights.html").exists()
    for folder in workspace.CATEGORY_DIRS.values():
        assert (out / folder).is_dir()
    assert (out / ".internal" / "reports" / "analysis.json").exists()
    assert not (out / "reports").exists()
    assert not (out / "previews").exists()


def test_the_default_report_has_no_legal_or_routing_vocabulary(archive, tmp_path):
    import cli

    out = tmp_path / "out"
    cli.main(["analyze", "--input", str(archive), "--output", str(out)])
    page = (out / "report.html").read_text().lower()

    body = page.split("<details", 1)[0]
    for phrase in simple_report.FORBIDDEN_IN_DEFAULT_UI:
        assert phrase not in body, phrase


def test_every_photograph_is_filed_in_exactly_one_category(archive, tmp_path):
    import cli

    out = tmp_path / "out"
    cli.main(["analyze", "--input", str(archive), "--output", str(out)])

    filed = [
        entry.name
        for folder in workspace.CATEGORY_DIRS.values()
        for entry in (out / folder).iterdir()
        if entry.is_symlink()
    ]
    assert sorted(filed) == ["p0.jpg", "p1.jpg", "p2.jpg"]


def test_the_run_leaves_the_originals_untouched(archive, tmp_path):
    import cli

    before = {p.name: p.read_bytes() for p in archive.iterdir()}
    cli.main(["analyze", "--input", str(archive), "--output", str(tmp_path / "out")])
    after = {p.name: p.read_bytes() for p in archive.iterdir()}
    assert before == after
    assert not list(archive.glob("*.xmp"))


def test_the_categories_and_the_report_agree(archive, tmp_path):
    import cli

    out = tmp_path / "out"
    cli.main(["analyze", "--input", str(archive), "--output", str(out)])
    rows, _ = __import__("reports").read_json(out / ".internal" / "reports" / "analysis.json")

    for row in rows:
        folder = workspace.CATEGORY_DIRS[row["category"]]
        assert (out / folder / row["filename"]).is_symlink()


def test_a_second_run_over_an_old_layout_tidies_it(archive, tmp_path):
    import cli

    out = tmp_path / "out"
    (out / "reports").mkdir(parents=True)
    (out / "reports" / "stale.json").write_text("{}")

    cli.main(["analyze", "--input", str(archive), "--output", str(out)])

    assert (out / ".internal" / "reports" / "stale.json").exists()
    assert (out / "report.html").exists()


def test_no_development_artefacts_in_the_output(archive, tmp_path):
    import cli

    out = tmp_path / "out"
    cli.main(["analyze", "--input", str(archive), "--output", str(out)])
    assert not (out / "test_results").exists()
    assert not any(p.name.endswith(".tmp") for p in out.rglob("*"))


def test_the_curation_categories_are_the_only_piles_shown():
    """The report's sections and the categories cannot drift apart."""
    assert {name for name, _ in simple_report.SECTIONS} == {
        c.value for c in curation.CATEGORY_ORDER
    }
    assert set(workspace.CATEGORY_DIRS) == {c.value for c in curation.CATEGORY_ORDER}


def test_the_report_links_to_the_insights(tmp_path, collection):
    page = simple_report.write(
        collection, tmp_path / "report.html", insights_link=workspace.INSIGHTS_NAME
    ).read_text()
    assert workspace.INSIGHTS_NAME in page


def test_a_preview_path_outside_the_report_directory_still_resolves(tmp_path):
    subject = record(preview_path=str(tmp_path / "elsewhere" / "a.jpg"))
    page = simple_report.write([subject], tmp_path / "out" / "report.html").read_text()
    assert "elsewhere" in page


def test_a_record_with_no_preview_renders_without_one(tmp_path):
    page = simple_report.write([record(preview_path="")], tmp_path / "report.html").read_text()
    assert "<img" not in page.split("<details", 1)[0]


def test_the_report_is_written_atomically_enough_to_be_readable(tmp_path, collection):
    path = simple_report.write(collection, tmp_path / "report.html")
    assert path.read_text().rstrip().endswith("</html>")


def test_paths_in_the_report_are_relative(tmp_path, collection):
    preview = tmp_path / "out" / ".internal" / "previews" / "top.jpg"
    preview.parent.mkdir(parents=True)
    write_jpeg(photo_like(40, 30), preview)
    collection[0].preview_path = str(preview)

    page = simple_report.write(collection, tmp_path / "out" / "report.html").read_text()
    assert ".internal/previews/top.jpg" in page
    assert str(tmp_path) not in page


def test_the_recipes_folder_is_named_where_the_report_says_it_is():
    assert workspace.RECIPES == "edit_recipes"
    assert Path(workspace.RECIPES).name in simple_report.t("report.recipe_ready", "en")
