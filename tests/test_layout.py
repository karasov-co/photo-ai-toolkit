"""Contact sheets and the token ledger.

The manifest and symlink-farm tests that used to be here went with
`routing.Destination`: a second layout, answering the same question as the
category folders, with no caller. The farm a run actually builds is covered in
`test_simple_report.py`.
"""


from PIL import Image

from layout import build_contact_sheet, report_token_spend


def test_contact_sheet_shows_every_candidate(tmp_path):
    previews = []
    for i in range(7):
        p = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (200, 150), (30 * i, 80, 120)).save(p)
        previews.append((f"frame{i}.RW2", p))
    sheet = build_contact_sheet(previews, tmp_path / "sheet.jpg", columns=4)
    assert sheet.exists()
    with Image.open(sheet) as img:
        assert img.width > 0 and img.height > 0


def test_no_sheet_is_written_when_there_is_nothing_to_review(tmp_path):
    assert build_contact_sheet([], tmp_path / "sheet.jpg") is None
    assert not (tmp_path / "sheet.jpg").exists()


def test_an_unreadable_preview_does_not_break_the_sheet(tmp_path):
    good = tmp_path / "good.jpg"
    Image.new("RGB", (120, 90), (10, 20, 30)).save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert build_contact_sheet([("ok.RW2", good), ("broken.RW2", bad)], tmp_path / "s.jpg")


# --- token reporting --------------------------------------------------------


def test_token_report_totals_every_stage():
    usage = {"stage1 luna": {"input_tokens": 1000, "output_tokens": 40},
             "stage2 sol": {"input_tokens": 8000, "output_tokens": 900}}
    out = report_token_spend(usage, frames=100)
    assert "9,000" in out
    assert "940" in out
    assert "per frame" in out


def test_token_report_survives_zero_frames():
    assert "TOKEN SPEND" in report_token_spend({}, frames=0)
