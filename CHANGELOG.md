# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-08-07

First tagged release. The tool itself already worked; this release makes it
maintainable — current dependencies, a test suite, and CI.

### Added

- Test suite: 161 pytest tests covering EXIF parsing, preview generation,
  CSV/JSON output, and vision-response parsing. No test touches the network or
  reads an API key — `tests/conftest.py` enforces both with autouse fixtures.
- Generated binary fixtures in `tests/fixtures/` (JPEG with full EXIF, JPEG
  without EXIF, JPEG with a corrupted EXIF segment, undecodable RW2), rebuilt by
  `tests/generate_fixtures.py`. All synthetic — no real photographs in the repo.
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

- A vision response that parsed as a JSON *array* instead of an object raised
  `AttributeError` out of `_parse_vision_response`. It now raises
  `VisionParseError` like every other malformed response, so the caller's
  `except VisionAnalysisError` handles it and the photo is recorded as a normal
  error rather than an unexpected crash.
- Removed an unused `numpy` import from `preview_generator.py`.
- Sorted imports in `main.py`; replaced a redundant open mode and a
  `try`/`except`/`pass` flagged by ruff.

### Known limitations

- EXIF metadata is not extracted from RAW files. The RAW branch delegates to
  Pillow, which cannot decode `.RW2`, `.ARW`, `.CR3` or `.NEF`, so every RAW
  file yields empty metadata. Previews are unaffected — those go through rawpy.
  See [CONTRIBUTING.md](CONTRIBUTING.md#known-limitations).

[Unreleased]: https://github.com/karasov-co/photo-ai-toolkit/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/karasov-co/photo-ai-toolkit/releases/tag/v1.0.0
