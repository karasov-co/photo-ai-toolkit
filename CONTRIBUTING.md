# Contributing

Thanks for taking a look. This is a small, focused CLI tool, so the bar is
simple: changes should keep the pipeline predictable for someone re-running it
over an archive they have already partly processed.

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Setup

Python 3.12 or newer.

```bash
git clone https://github.com/karasov-co/photo-ai-toolkit.git
cd photo-ai-toolkit

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt
```

You only need an API key to run the tool against real photos, not to run the
tests:

```bash
cp .env.example .env             # then put your key in it
```

## Running the tests

```bash
pytest                           # whole suite
pytest tests/test_exif_reader.py # one module
pytest -k unicode                # one topic
```

The suite is hermetic by design. `tests/conftest.py` installs two autouse
fixtures that raise if anything tries to open a socket or read
`OPENAI_API_KEY`, so a test that accidentally reaches the network fails loudly
instead of silently costing money. Please keep it that way — stub
`client.responses.create` rather than calling it.

Binary fixtures live in `tests/fixtures/` and are generated, not photographed:

```bash
python tests/generate_fixtures.py
```

Regenerate and commit them only when a fixture actually needs to change.

One exception: `raw_header.rw2` is the first 5 KB of a real Panasonic RW2,
because exifread has to parse a genuine IFD. It stops short of the embedded
JPEG preview (offset 6144) so it holds no image data, and the camera serial
number is zeroed. Source RAWs are ~34 MB and are not in the repository — pass
one in to regenerate it:

```bash
python tests/generate_fixtures.py path/to/photo.RW2
```

## Linting

```bash
ruff check .
ruff check --fix .
```

CI runs `ruff check` and `pytest` on Python 3.12, 3.13, and 3.14. Both must be
green before a pull request can merge.

## Submitting a change

1. Branch off `main`.
2. Make the change, and add a test that fails without it.
3. Run `pytest` and `ruff check .`.
4. Add an entry under `## [Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
5. Open a pull request and fill in the template.

Small, single-purpose pull requests get reviewed much faster than large ones.

## Two things to be careful with

**The scoring prompt.** `VISION_PROMPT` in `vision_analyzer.py` defines the
1–1000 scale and its anchors. Editing it re-scores every future run, so scores
stop being comparable with results already sitting in someone's `results.csv`.
If you have a reason to change it, say so explicitly in the pull request.

**The output schema.** `CSV_FIELDNAMES` in `output_writer.py` and the four keys
returned by `_parse_vision_response` are what the resume-on-rerun logic reads.
Adding a column is fine; renaming or removing one is a breaking change and
belongs in a major release.

## How RAW metadata is read

`_extract_raw` tries exifread first and fills any gaps from LibRaw. The split
matters:

- **exifread** parses the TIFF-based formats (`.RW2`, `.ARW`, `.NEF`) and is the
  only one of the two that reports make, model and GPS.
- **LibRaw** (via rawpy) covers containers exifread cannot parse — `.CR3` is
  ISO BMFF, not TIFF — but exposes only ISO, shutter, aperture, focal length,
  timestamp and lens.

The fallback never overwrites a field exifread already found. Pillow is not in
this path at all; it has no RAW decoder, and routing RAW through it was the
reason RAW files returned empty metadata for so long.
