# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-07

First tagged release. The tool itself already worked; this release makes it
maintainable — current dependencies, a test suite, and CI.

### Added

- Test suite: 197 pytest tests covering EXIF parsing, preview generation,
  CSV/JSON output, and vision-response parsing. No test touches the network or
  reads an API key — `tests/conftest.py` enforces both with autouse fixtures.
- Binary fixtures in `tests/fixtures/`, rebuilt by `tests/generate_fixtures.py`.
  Four are synthetic (JPEG with full EXIF, JPEG without EXIF, JPEG with a
  corrupted EXIF segment, undecodable RW2). The fifth, `raw_header.rw2`, is the
  first 5 KB of a real Panasonic RW2 — exifread needs a genuine IFD to parse. It
  stops short of the embedded JPEG preview so it contains no image data, and the
  camera serial number is zeroed.
- `exifread==3.5.1`, for reading EXIF out of RAW files.
- GitHub Actions CI running `ruff check` and `pytest` on Python 3.12, 3.13 and 3.14.
- `requirements-dev.txt` for the test and lint toolchain.
- `pyproject.toml` holding ruff and pytest configuration.
- Dependabot, monthly, for pip and GitHub Actions.
- `LICENSE` (MIT). The README had claimed MIT since the initial commit, but no
  license file was ever committed, which left the code technically unlicensed.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), issue and
  pull request templates.

### Changed

- Migrated the vision call from Chat Completions to the OpenAI **Responses API**
  (`client.responses.create`). The image part is now `input_image` with a plain
  `image_url` string, the text part is `input_text`, `max_completion_tokens`
  became `max_output_tokens`, and the reply is read from `response.output_text`.
- Upgraded and re-pinned every dependency: `openai` 1.35.7 → 2.53.0,
  `Pillow` 10.4.0 → 12.3.0, `rawpy` 0.22.0 → 0.27.0, `python-dotenv` 1.0.1 →
  1.2.2, `tqdm` 4.66.4 → 4.70.0.
- The model is now requested with `reasoning={"effort": "low"}`, and the output
  ceiling was raised from 500 to 2000 tokens so that reasoning tokens cannot
  crowd out the JSON payload.

The scoring prompt and the output schema are unchanged. Scores from this
release remain comparable with scores produced before it.

### Fixed

- **EXIF was never extracted from RAW files.** `_extract_raw` delegated to
  Pillow, which has no decoder for `.RW2`, `.ARW`, `.CR3` or `.NEF`, so every
  RAW file recorded empty camera metadata — the format this tool is built for.
  It now reads through exifread and falls back to LibRaw for containers exifread
  cannot parse. Verified against real RW2 files: make, model, lens, ISO, shutter,
  aperture, focal length and date all populate where the columns were blank.
- **RAW+JPEG pairs overwrote each other's preview.** Previews were named
  `<stem>.jpg`, so `P1042675.RW2` and `P1042675.JPG` — what the camera produces
  in RAW+JPEG mode — resolved to the same file, and one photo was scored against
  the other's image. Previews are now named `<stem>_<ext>.jpg`.
- **An interrupted run could wipe `results.json`.** Each append rewrote the file
  in place, truncating it first, so an interrupt mid-write left an unparseable
  file that the next run silently restarted from empty. Writes now go to a temp
  file and are renamed over the target atomically, and a file that still fails to
  parse is moved aside as `results.json.corrupt` instead of being discarded.
- A vision response that parsed as a JSON *array* instead of an object raised
  `AttributeError` out of `_parse_vision_response`. It now raises
  `VisionParseError` like every other malformed response, so the caller's
  `except VisionAnalysisError` handles it and the photo is recorded as a normal
  error rather than an unexpected crash.
- The dry-run stub scored 5 on a 1–1000 scale, so `--dry-run` always reported an
  average of 5/1000 and an empty top-picks list. It now returns 500.
- `load_processed_filenames` used `row["filename"]`, which raised on a CSV row
  missing that column and aborted the resume scan. It skips such rows and logs.
- Removed an unused `numpy` import from `preview_generator.py`.
- Sorted imports in `main.py`; replaced a redundant open mode and a
  `try`/`except`/`pass` flagged by ruff.

[Unreleased]: https://github.com/karasov-co/photo-ai-toolkit/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/karasov-co/photo-ai-toolkit/releases/tag/v1.0.0
