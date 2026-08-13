"""The metric that scores its own output, and the command that checks it.

`frame_quality` is the objective the preview search optimises and the ruler
that reports the outcome, so `uplift` is the distance the search moved its own
number. These tests hold the two things that keep that honest: an outside
correlation anybody can run, and a flag in the report saying it has not been.
"""

import pytest

import bench_quality
import edit_recipe
from reports import AssetRecord


def record(name, current, potential=None):
    return AssetRecord(
        asset_id=name, source_path=f"/p/{name}", filename=name, media_type="photo",
        checksum=name, asset_key=name, status="ok",
        scores={"current_quality": current, "post_edit_potential": potential or current},
    )


def test_spearman_matches_a_known_value():
    assert bench_quality.spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    assert bench_quality.spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)
    assert bench_quality.spearman([1, 2, 3], [2, 2, 2]) == 0.0


def test_ties_are_averaged_rather_than_ordered_arbitrarily():
    """Two photographs a person called equal must not be given an order."""
    assert bench_quality.spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


def test_a_rank_column_is_inverted_so_higher_is_always_better(tmp_path):
    """Rank 1 is best; a score of 1 is worst. Reading one as the other reports
    a metric that is exactly wrong as one that agrees."""
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,human_rank\na.jpg,1\nb.jpg,5\n")
    labels = bench_quality.read_labels(csv)
    assert labels["a.jpg"]["score"] > labels["b.jpg"]["score"]


def test_a_score_column_is_taken_as_written(tmp_path):
    csv = tmp_path / "labels.csv"
    csv.write_text("filename,human_score\na.jpg,90\nb.jpg,20\n")
    labels = bench_quality.read_labels(csv)
    assert labels["a.jpg"]["score"] == 90


def test_every_csv_in_a_directory_is_read(tmp_path):
    (tmp_path / "one.csv").write_text("filename,human_score\na.jpg,9\n")
    (tmp_path / "two.csv").write_text("filename,human_score\nb.jpg,3\n")
    assert set(bench_quality.read_labels(tmp_path)) == {"a.jpg", "b.jpg"}


def test_a_metric_that_agrees_is_reported_as_agreeing(tmp_path):
    records = [record(f"{i}.jpg", current=i * 10) for i in range(1, 26)]
    labels = {f"{i}.jpg": {"score": float(i)} for i in range(1, 26)}
    result = bench_quality.compare(records, labels)

    assert result.quality.rho == pytest.approx(1.0)
    assert result.validates_quality
    assert "well enough" in bench_quality.format_report(result)


def test_a_metric_that_disagrees_says_so_plainly():
    records = [record(f"{i}.jpg", current=i * 10) for i in range(1, 26)]
    labels = {f"{i}.jpg": {"score": float(-i)} for i in range(1, 26)}
    result = bench_quality.compare(records, labels)

    assert result.quality.rho < 0
    assert not result.validates_quality
    assert "does not rank photographs the way a person does" in bench_quality.format_report(result)


def test_too_few_labels_is_not_a_result():
    records = [record(f"{i}.jpg", current=i * 10) for i in range(1, 4)]
    labels = {f"{i}.jpg": {"score": float(i)} for i in range(1, 4)}
    result = bench_quality.compare(records, labels)

    assert result.quality.rho == pytest.approx(1.0)
    assert not result.validates_quality, "three samples cannot validate anything"
    assert "Too few samples" in bench_quality.format_report(result)


def test_uplift_needs_before_and_after_labels():
    records = [record(f"{i}.jpg", current=40, potential=40 + i) for i in range(1, 26)]
    labels = {f"{i}.jpg": {"score": 40.0} for i in range(1, 26)}
    result = bench_quality.compare(records, labels)
    assert result.uplift.n == 0
    assert "human_score_edited" in result.uplift.note


def test_uplift_is_correlated_when_the_labels_carry_a_delta():
    records = [record(f"{i}.jpg", current=40, potential=40 + i) for i in range(1, 26)]
    labels = {f"{i}.jpg": {"score": 40.0, "edited": 40.0 + i} for i in range(1, 26)}
    result = bench_quality.compare(records, labels)
    assert result.uplift.rho == pytest.approx(1.0)
    assert result.validates_uplift


def test_labels_for_photographs_that_were_not_analysed_are_reported():
    result = bench_quality.compare([record("a.jpg", 50)], {"a.jpg": {"score": 1.0},
                                                           "ghost.jpg": {"score": 2.0}})
    assert result.missing == ["ghost.jpg"]


def test_the_report_states_why_the_command_exists():
    text = bench_quality.format_report(bench_quality.BenchResult())
    assert "optimises" in text and "ruler" in text


# --- the flag in the product --------------------------------------------------


def test_uplift_is_declared_unvalidated_until_somebody_measures_it():
    """A claim about evidence, so the code may not decide it for itself."""
    assert edit_recipe.UPLIFT_VALIDATED is False


def test_the_report_says_the_gain_is_unchecked(tmp_path):
    import simple_report

    page = simple_report.write(
        [record("a.jpg", 50, 70)], tmp_path / "report.html", expert=False
    ).read_text()
    assert "not been checked against a labelled set" in page


def test_a_validated_run_would_drop_the_caveat(tmp_path):
    import simple_report

    validated = record("a.jpg", 50, 70)
    validated.uplift_validated = True
    page = simple_report.write([validated], tmp_path / "report.html", expert=False).read_text()
    assert "not been checked against a labelled set" not in page


def test_the_flag_reaches_the_stored_record(tmp_path):
    """It has to be in the JSON, not only on the page."""
    assert AssetRecord(
        asset_id="a", source_path="p", filename="a.jpg", media_type="photo", checksum="c"
    ).to_dict()["uplift_validated"] is False
