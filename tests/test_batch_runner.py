import json

import pytest

from photoai import prompts
from photoai.batch_runner import (
    RESPONSES_ENDPOINT,
    MixedModelBatch,
    TokenLedger,
    assert_single_model,
    attach_filenames,
    parse_batch_output,
    parse_group_json,
    stage1_request,
    stage1_verdict,
    stage2_request,
    write_jsonl,
)


def frame(name, hi=0.01, sh=0.02):
    return {"filename": name, "clipped_highlights": hi, "clipped_shadows": sh, "encoded": "QUJD"}


# --- request shape ----------------------------------------------------------


def test_stage1_request_targets_the_responses_endpoint():
    r = stage1_request("f1", "QUJD", "gpt-5.6-luna")
    assert r["url"] == RESPONSES_ENDPOINT
    assert r["method"] == "POST"
    assert r["custom_id"] == "f1"


def test_stage1_is_capped_to_a_one_word_answer():
    body = stage1_request("f1", "QUJD", "gpt-5.6-luna")["body"]
    assert body["max_output_tokens"] == prompts.STAGE1_MAX_OUTPUT_TOKENS <= 20


def test_stage1_sends_the_image_at_low_detail():
    content = stage1_request("f1", "QUJD", "gpt-5.6-luna")["body"]["input"][0]["content"]
    images = [b for b in content if b["type"] == "input_image"]
    assert len(images) == 1
    assert images[0]["detail"] == "low"


def test_stage2_sends_every_image_at_high_detail():
    frames = [frame(f"f{i}.RW2") for i in range(12)]
    content = stage2_request("g1", frames, "gpt-5.6-sol")["body"]["input"][0]["content"]
    images = [b for b in content if b["type"] == "input_image"]
    assert len(images) == 12
    assert {b["detail"] for b in images} == {"high"}


def test_the_system_prompt_is_identical_across_requests_so_it_caches():
    a = stage2_request("g1", [frame("a.RW2")], "gpt-5.6-sol")["body"]["instructions"]
    b = stage2_request("g2", [frame("b.RW2")], "gpt-5.6-sol")["body"]["instructions"]
    assert a == b == prompts.STAGE2_SYSTEM


def test_the_system_prompt_rides_in_instructions_not_in_the_user_turn():
    """instructions is the front of the cached prefix."""
    body = stage2_request("g1", [frame("a.RW2")], "gpt-5.6-sol")["body"]
    assert body["instructions"] == prompts.STAGE2_SYSTEM
    user_text = " ".join(
        b.get("text", "") for b in body["input"][0]["content"] if b["type"] == "input_text"
    )
    assert "DO NOT PENALISE" not in user_text


def test_measured_clipping_reaches_the_prompt_with_the_hard_rule():
    frames = [frame("a.RW2", hi=0.012), frame("b.RW2", hi=0.081)]
    text = stage2_request("g1", frames, "gpt-5.6-sol")["body"]["input"][0]["content"][0]["text"]
    assert "1.2%" in text and "8.1%" in text
    assert "penalising highlights is forbidden" in text
    assert text.count("forbidden") == 1, "only the sub-3% frame gets the exemption"


def test_the_output_budget_grows_with_the_group():
    small = stage2_request("g", [frame("a.RW2")], "m")["body"]["max_output_tokens"]
    large = stage2_request("g", [frame(f"f{i}.RW2") for i in range(12)], "m")["body"][
        "max_output_tokens"
    ]
    assert large > small


def test_a_batch_mixing_two_models_is_refused_before_upload(tmp_path):
    """The API fails the whole batch with mismatched_model; catch it locally."""
    mixed = [
        stage1_request("f1", "QUJD", "gpt-5.6-luna"),
        stage2_request("g1", [frame("a.RW2")], "gpt-5.6-sol"),
    ]
    with pytest.raises(MixedModelBatch, match="separate batches"):
        write_jsonl(mixed, tmp_path / "in.jsonl")
    assert not (tmp_path / "in.jsonl").exists()


def test_a_single_model_batch_reports_its_model():
    requests = [stage1_request(f"f{i}", "QUJD", "gpt-5.6-luna") for i in range(3)]
    assert assert_single_model(requests) == "gpt-5.6-luna"


def test_an_empty_batch_is_not_a_model_mismatch():
    assert assert_single_model([]) == ""


def test_requests_serialise_to_one_json_line_each(tmp_path):
    requests = [stage1_request(f"f{i}", "QUJD", "gpt-5.6-luna") for i in range(3)]
    path = write_jsonl(requests, tmp_path / "in.jsonl")
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    assert [json.loads(line)["custom_id"] for line in lines] == ["f0", "f1", "f2"]


# --- parsing the batch output ----------------------------------------------


def output_line(custom_id, text, usage=None, status=200):
    return json.dumps(
        {
            "custom_id": custom_id,
            "response": {
                "status_code": status,
                "body": {"output_text": text, "usage": usage or {}},
            },
        }
    )


def test_results_are_keyed_by_custom_id():
    results = parse_batch_output([output_line("f1", "KEEP"), output_line("f2", "REJECT")])
    assert set(results) == {"f1", "f2"}
    assert results["f1"].text == "KEEP"


def test_a_failed_line_survives_as_an_error_not_a_silent_drop():
    line = json.dumps({"custom_id": "f9", "error": {"message": "rate limited"}})
    results = parse_batch_output([line])
    assert not results["f9"].ok
    assert "rate limited" in results["f9"].error


def test_an_http_error_is_recorded_against_the_frame():
    results = parse_batch_output([output_line("f1", "", status=500)])
    assert not results["f1"].ok
    assert "500" in results["f1"].error


def test_unparseable_lines_are_skipped_without_killing_the_run():
    results = parse_batch_output(["{not json", "", output_line("f1", "KEEP")])
    assert set(results) == {"f1"}


def test_text_is_recovered_from_the_block_form_too():
    line = json.dumps(
        {
            "custom_id": "f1",
            "response": {
                "status_code": 200,
                "body": {"output": [{"content": [{"type": "output_text", "text": "KEEP"}]}]},
            },
        }
    )
    assert parse_batch_output([line])["f1"].text == "KEEP"


# --- stage 1 verdict --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "keep"),
    [
        ("KEEP", True),
        ("keep", True),
        ("REJECT", False),
        ("reject — lens cap", False),
        ("Reject.", False),
        ("", True),
        ("I am not sure", True),
        ("The frame appears usable", True),
    ],
)
def test_only_a_clear_reject_discards_a_frame(text, keep):
    """An unreadable answer must never delete a photograph."""
    assert stage1_verdict(text) is keep


# --- stage 2 group JSON -----------------------------------------------------


VALID_ITEM = {
    "n": 1,
    "genre": "street",
    "axis_a": 1,
    "axis_b": 2,
    "axis_c": 3,
    "recover": "easy",
    "note": "lift shadows",
}


def test_a_clean_array_parses():
    assert len(parse_group_json(json.dumps([VALID_ITEM, {**VALID_ITEM, "n": 2}]))) == 2


def test_a_fenced_array_parses():
    text = f"```json\n{json.dumps([VALID_ITEM])}\n```"
    assert len(parse_group_json(text)) == 1


def test_an_array_buried_in_prose_parses():
    text = f"Here you go:\n{json.dumps([VALID_ITEM])}\nHope that helps."
    assert len(parse_group_json(text)) == 1


def test_an_object_wrapper_is_unwrapped():
    assert len(parse_group_json(json.dumps({"frames": [VALID_ITEM]}))) == 1


@pytest.mark.parametrize("text", ["", "no json at all", "{broken", "[1,2,3]"])
def test_unusable_group_output_yields_nothing_rather_than_garbage(text):
    assert parse_group_json(text) == []


# --- mapping objects back onto files ---------------------------------------


GROUP = ["a.RW2", "b.RW2", "c.RW2"]


def test_objects_map_onto_files_by_their_index():
    placed = attach_filenames([{"n": 3, "axis_a": 1}, {"n": 1, "axis_a": 2}], GROUP)
    assert [p["filename"] for p in placed] == ["c.RW2", "a.RW2"]


def test_a_missing_index_falls_back_to_position():
    placed = attach_filenames([{"axis_a": 1}, {"axis_a": 2}], GROUP)
    assert [p["filename"] for p in placed] == ["a.RW2", "b.RW2"]


def test_an_out_of_range_index_falls_back_rather_than_mislabelling():
    placed = attach_filenames([{"n": 99, "axis_a": 1}], GROUP)
    assert placed[0]["filename"] == "a.RW2"


def test_objects_that_cannot_be_placed_are_dropped():
    """Better to lose a ranking than to attach it to the wrong photograph."""
    placed = attach_filenames([{"n": 1}, {"n": 2}, {"n": 3}, {"n": 4}, {"n": 5}], ["only.RW2"])
    assert len(placed) == 1
    assert placed[0]["filename"] == "only.RW2"


def test_a_non_numeric_index_falls_back_to_position():
    placed = attach_filenames([{"n": "second", "axis_a": 1}], GROUP)
    assert placed[0]["filename"] == "a.RW2"


# --- token ledger -----------------------------------------------------------


def test_the_ledger_sums_per_stage():
    ledger = TokenLedger()
    ledger.add("stage1", {"input_tokens": 100, "output_tokens": 5})
    ledger.add("stage1", {"input_tokens": 120, "output_tokens": 6})
    ledger.add("stage2", {"input_tokens": 9000, "output_tokens": 800})
    assert ledger.stages["stage1"]["input_tokens"] == 220
    assert ledger.stages["stage1"]["calls"] == 2
    assert ledger.totals()["output_tokens"] == 811


def test_the_ledger_tracks_cached_tokens():
    ledger = TokenLedger()
    ledger.add("stage2", {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 900}})
    assert ledger.totals()["cached_tokens"] == 900
    assert "90%" in ledger.report(frames=1)


def test_the_report_gives_a_per_frame_figure():
    ledger = TokenLedger()
    ledger.add("stage2", {"input_tokens": 1000, "output_tokens": 200})
    report = ledger.report(frames=10)
    assert "per frame" in report
    assert "100 in" in report and "20 out" in report


def test_the_ledger_survives_missing_usage():
    ledger = TokenLedger()
    ledger.add("stage1", {})
    assert ledger.totals()["input_tokens"] == 0
    assert "TOKEN SPEND" in ledger.report(frames=0)
