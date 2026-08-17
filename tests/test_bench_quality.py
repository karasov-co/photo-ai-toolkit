"""The metric that scores its own output, and the command that checks it.

`frame_quality` is the objective the preview search optimises and the ruler
that reports the outcome, so `uplift` is the distance the search moved its own
number. These tests hold the two things that keep that honest: an outside
correlation anybody can run, and a flag in the report saying it has not been.
"""

import pytest

from photoai import bench_quality, edit_recipe
from photoai.reports import AssetRecord


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


def test_the_report_says_the_gain_compares_frames_rather_than_measuring_them(tmp_path):
    """It used to say only "not checked against a labelled set", which reads as
    a number that is probably right and merely unaudited. The circularity is
    sharper than that: the metric is both the objective the edit search climbs
    and the ruler that reports the climb, so +12 is a comparison between frames
    and cannot be a measurement of a photograph."""
    from photoai import simple_report

    page = simple_report.write(
        [record("a.jpg", 50, 70)], tmp_path / "report.html", expert=False
    ).read_text()
    assert "compares frames" in page
    assert "does not measure photographs" in page
    assert "bench-quality" in page


def test_a_validated_run_would_drop_the_caveat(tmp_path):
    from photoai import simple_report

    validated = record("a.jpg", 50, 70)
    validated.uplift_validated = True
    page = simple_report.write([validated], tmp_path / "report.html", expert=False).read_text()
    assert "not been checked against a labelled set" not in page


def test_the_flag_reaches_the_stored_record(tmp_path):
    """It has to be in the JSON, not only on the page."""
    assert AssetRecord(
        asset_id="a", source_path="p", filename="a.jpg", media_type="photo", checksum="c"
    ).to_dict()["uplift_validated"] is False


# --- the thresholds against a person ------------------------------------------
#
# The correlation above answers "does the score rank photographs the way a
# person does". It does not answer "do the five piles put a photograph where a
# person would put it", and that second question is the product's central
# promise. These cover the second one.


def _pairs(spec):
    """(human pile, final score) pairs from a compact spec."""
    return [(pile, score) for pile, scores in spec.items() for score in scores]


def test_perfect_thresholds_agree_completely():
    result = bench_quality.agreement(
        _pairs({"top": [90, 95], "good": [60, 70], "weak": [20, 30]})
    )
    assert result.agreement == 1.0
    assert result.top_precision == 1.0
    assert result.top_recall == 1.0


def test_the_confusion_matrix_says_which_way_it_is_wrong():
    """'61% agreement' is a complaint. 'It calls your top pile good' is a bug."""
    result = bench_quality.agreement(_pairs({"top": [70, 72, 74]}))
    assert result.confusion["top"]["good"] == 3
    assert result.confusion["top"]["top"] == 0
    assert result.top_recall == 0.0


def test_the_sweep_finds_thresholds_that_would_have_agreed_better():
    """The number somebody can act on, rather than one they can only regret."""
    # A person whose "top" starts around 70, not 85.
    result = bench_quality.agreement(
        _pairs({"top": [72, 75, 78, 80], "good": [50, 55, 60], "weak": [20, 25, 30]})
    )
    assert result.agreement < 1.0
    assert result.best_agreement > result.agreement
    assert result.best_top <= 72


def test_the_sweep_never_reports_worse_than_what_is_running():
    result = bench_quality.agreement(
        _pairs({"top": [90], "good": [60], "weak": [20]})
    )
    assert result.best_agreement >= result.agreement


def test_too_few_labels_is_not_a_calibration():
    result = bench_quality.agreement([("top", 90)] * 3)
    assert result.agreement == 1.0
    assert not result.meaningful
    assert not result.calibrated, "3 photographs must never read as calibrated"


def test_a_labels_file_can_carry_piles_alone(tmp_path):
    """Sorting 300 frames into three piles is an evening. Scoring them is not."""
    sheet = tmp_path / "labels.csv"
    sheet.write_text("filename,human_pile\na.jpg,top\nb.jpg,weak\nc.jpg,nonsense\n")
    labels = bench_quality.read_labels(sheet)
    assert labels["a.jpg"]["pile"] == "top"
    assert labels["b.jpg"]["pile"] == "weak"
    assert "c.jpg" not in labels, "an unknown pile is dropped, not guessed at"


def test_the_template_is_shuffled_and_carries_no_verdict(tmp_path):
    """A sheet showing the tool's answer measures how persuadable the labeller
    is, and an ordered sheet lets the previous row anchor the next one."""
    records = [
        AssetRecord(
            asset_id=f"p{i:03d}", source_path=f"/gone/p{i:03d}.jpg",
            filename=f"p{i:03d}.jpg", media_type="photo", checksum=f"c{i}",
            final_score=i, category="TOP",
        )
        for i in range(60)
    ]
    out = tmp_path / "sheet.csv"
    assert bench_quality.write_template(records, out) == 60

    lines = out.read_text().splitlines()
    assert lines[0] == "filename,human_pile"
    assert all(line.endswith(",") for line in lines[1:]), "no verdict may leak in"
    names = [line.split(",")[0] for line in lines[1:]]
    assert names != sorted(names), "an ordered sheet anchors the labeller"
    assert set(names) == {r.filename for r in records}


def test_the_report_says_plainly_when_nothing_has_been_labelled():
    result = bench_quality.BenchResult()
    text = bench_quality.format_report(result)
    assert "no human_pile column" in text
    assert "nothing has" in text
