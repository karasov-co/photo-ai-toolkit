# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-08-12

Turns the scoring tool into a culling, assessment and distribution system for
photo **and** video. The organising idea: current quality and realistic
post-edit potential are different questions and are now stored as different
numbers. A dark, tilted, flat RAW is an unedited photograph, not a bad one; a
sharp frame that missed focus is finished. One score cannot say both.

### Added

- **Ten separate score dimensions** per asset (`scoring.py`), on 0–100: current
  quality, recoverability, post-edit potential, aesthetic potential, stock
  potential, portfolio potential, legal readiness, uniqueness, confidence, and a
  single blended routing score. Only the last is a blend.
- **Bounded edit-potential preview search** (`edit_recipe.py`). Applies plausible
  non-destructive edits to a 512px working copy — exposure, white balance,
  highlight/shadow recovery, contrast, straightening, saliency-aware crops,
  restrained denoise — re-scores each, and charges the uplift for what it cost
  (crop area, exposure push, denoise, and a large discount for JPEG sources).
  Stores a human-readable recipe, never a modified pixel. Nothing generative.
- **Typed issues with fixability** (`issues.py`): 39 codes classified as fixable,
  partially fixable, or unrecoverable. An unrecoverable issue *caps* post-edit
  potential rather than penalising it, which is what stops a brighter
  out-of-focus frame from being promoted by measured uplift.
- **Video analysis** (`video_analyzer.py`). ffprobe container facts, temporal
  sampling across the whole clip, and dense frame bursts for what only exists
  between frames. Camera motion via phase correlation, which separates a
  deliberate pan from handheld shake — raw inter-frame difference calls a good
  pan unusable. Usable segments with edit handles, poster frame, focus
  consistency, exposure drift, flicker, rolling-shutter shear, black and frozen
  frames, VFR and log/HDR detection.
- **Five route classes** — trash, review, stock_standard, stock_strong, flagship
  — with explicit ordered precedence, plus secondary destination tags
  (editorial-only, needs-release, legal-review, archive-only, portfolio).
- **The commercial-stock block, enforced in code.** A face or a readable
  trademark cannot reach commercial stock at any score. Both require a release.
  The vision model supplies the flags and is not trusted to act on them.
- **Diversity-aware flagship selection** (`duplicates.py`) via maximal marginal
  relevance, with an absolute floor as well as a rank. Ranking by score alone
  fills a portfolio gallery with twenty frames of one sunset.
- **Near-duplicate clustering** by perceptual hash with a timestamp veto — the
  same facade on two days is not a burst — and union-find so a slow pan chains
  into one cluster.
- **Safe quarantine workflow** (`quarantine.py`): dry run by default, sidecars
  move with their RAW, idempotent by checksum, collision-safe, path traversal
  fenced after `resolve()`, symlinks never followed, append-only manifest, and a
  full restore. Permanent purge sits behind a typed phrase, a minimum age, a
  lock, and a manifest entry.
- **Marketplace rules as versioned data** (`data/marketplace_rules.json`) for
  eight platforms, each carrying its source URL and verification date. Adobe
  Stock, Shutterstock, Getty/iStock, Alamy, Pond5, Dreamstime, Depositphotos,
  123RF. Verified against official documentation on 2026-08-11.
- **Provenance tracking** (`provenance.py`) read from C2PA, IPTC
  `DigitalSourceType`, XMP and EXIF — declared, never guessed from appearance.
  Absence of AI metadata proves nothing and is reported as undeclared.
- **Stock metadata generation** (`stock_metadata.py`): titles, descriptions,
  ordered keywords capped below the platform maximum, categories, releases,
  XMP sidecars and a submission CSV. Originals are never modified; GPS is
  stripped from export copies.
- **Versioned calibration profiles** (`calibration.py`) with separate photo and
  video thresholds and stock-first / portfolio-first variants. Routing reads
  only stored dimensions, so `reclassify` retunes a whole collection in
  milliseconds without decoding a pixel or spending a token.
- **Manual overrides that persist** (`overrides.py`), keyed by content checksum
  so they survive renames, re-applied after every analysis. The tool's own
  conclusion is preserved beside the override rather than erased.
- **Reports** (`reports.py`) in JSON, CSV and HTML, plus a collection summary,
  and a `RedactingFilter` that scrubs credentials from every log handler and
  from anything written to disk.
- **Localisation** (`i18n.py`) with complete English and Russian catalogues,
  coverage enforced by a test.
- **New CLI** (`cli.py`) with `analyze`, `report`, `reclassify`, `quarantine`,
  `restore`, `purge`, `export`, `override`, `profiles`, `validate-profile`.
  Filtering and sorting on every stored dimension.
- **Media layer** (`media.py`): photo/video typing, content checksums, RAW+JPEG
  +XMP sidecar grouping, and a decompression-bomb guard that checks the header
  before the pixel buffer is allocated.
- 498 new tests (878 total), all hermetic. Every fixture image is generated from
  a seed; no private media is committed. CI now installs FFmpeg so the video
  integration tests actually run instead of skipping.

### Fixed

- **The quality scale was collapsed.** On a 45-frame RAW sample the first
  scoring function produced min 51 / median 81 / max 91 — the bottom half of the
  range was unreachable, the same failure as the absolute model scores it was
  meant to replace. Refitted against measured distributions to span 17–75 with a
  median of 56. Clipping now multiplies rather than adding, where it had been a
  flat 16-point bonus on nearly every frame.
- **Tilt estimation was measuring the pixel grid, not the photograph.** A
  three-point central difference spans one pixel, so any edge tilted less than
  about seven degrees quantised to zero; the function returned 0.5° for every
  input between −3° and +6°. Replaced with a block structure tensor, which
  recovers the angle to within about one degree, and gated on structural
  coherence so scenes without straight lines decline rather than returning a
  confident wrong angle.
- **Camera shake was measured across burst boundaries.** Frames from different
  sample positions — seconds apart — were compared as if consecutive, reporting
  22px of jitter on a 128px frame. Each burst is now analysed independently and
  combined worst-case. On a real panning clip this moved jitter from 22.5px to
  0.35px and correctly reclassified the shot as a deliberate pan.
- **Unexamined frames were scored as though they were guilty.** Pessimistic
  defaults on the release flags exist to make the commercial-stock block fail
  safe, but scoring an unchecked frame as if a face *and* a logo had been
  confirmed dragged every dimension down and collapsed the whole stock axis
  before the vision pass could run. Unknown is now its own state, carried by
  confidence, and reported as "unchecked" rather than as a finding.
- **A lens-cap frame reached manual review instead of trash**, because the
  confidence gate ran before the blocker check. Confirmed unrecoverable problems
  come from deterministic local measurement; not knowing what is *in* a frame
  does not make a pure-black one recoverable.
- **The flagship quota equalled the candidate count**, so every eligible asset
  became flagship — six of eight photographs in one run.
- **Near-ties were condemned as weaker duplicates.** On a real archive, sibling
  pairs measured 42/38, 40/39 and 69/66; only the first is a real difference. A
  duplicate is now only called weaker when it loses by a configurable margin,
  and inside that margin both frames survive for a human to choose. Trash on a
  64-file run dropped from 5 to 2.
- **Cancelling a run raised `KeyError`** instead of returning partial results.
- **The printed summary was computed before manual overrides were applied**, so
  it contradicted the class stored against each asset.
- **`reclassify` could never produce a flagship**, because the stored report did
  not carry the perceptual hash the diversity pass needs.

### Changed

- `main.py` is retained unchanged as the legacy single-photo entry point.
- Highlight and shadow thresholds loosened for RAW, which genuinely holds one to
  two stops the rendered preview cannot show.
- Being below a marketplace's 4 MP floor is no longer an issue on the
  photograph; it is a per-platform marketplace fact.

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
