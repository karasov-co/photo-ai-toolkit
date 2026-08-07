# photo-ai-toolkit

[![CI](https://github.com/karasov-co/photo-ai-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/karasov-co/photo-ai-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

AI-powered CLI tool that analyzes, tags, and scores your photos using GPT-5.5 Vision.

---

## Screenshot

```
$ python main.py --input ./photos --output ./results

Processing photos: 100%|████████████████| 24/24 [02:18<00:00]

────────────────────────────────────────
 SUMMARY
────────────────────────────────────────
 Photos processed : 24
 Errors           : 1
 Average score    : 673.4/1000

 Top-8 by quality:
  1. [912/1000] DSC04821.ARW
  2. [889/1000] IMG_0034.CR3
  3. [856/1000] P1003847.RW2
  ...

 RECOMMENDED FOR STORIES (score 700+):
  [912/1000] DSC04821.ARW
  [889/1000] IMG_0034.CR3
  [856/1000] P1003847.RW2
  [741/1000] _DSC2201.NEF
────────────────────────────────────────

Results saved to ./results/results.csv and ./results/results.json
```

---

## Features

- Extracts EXIF metadata (camera, lens, ISO, shutter speed, aperture, focal length, GPS)
- Generates 512px JPEG previews from RAW and JPEG files
- Sends previews to GPT-5.5 Vision for scene analysis via the OpenAI Responses API
- Returns description, up to 10 tags, and a 1–1000 quality score per photo
- Exports results to CSV and JSON
- Highlights best shots for Instagram Stories (score ≥ 700)
- Skips already-processed files — safe to re-run on the same folder
- Retries on API errors with exponential backoff

---

## Quick Start

Requires Python 3.12 or newer.

```bash
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI API key
python main.py --input ./photos --output ./results
```

Use `--dry-run` to walk the whole pipeline without spending anything on API
calls, and `--force` to re-analyze files that are already in `results.csv`.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest          # 197 tests, no network, no API key needed
ruff check .
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup, and
[CHANGELOG.md](CHANGELOG.md) for release history.

---

## How It Works

1. **Scan** — finds all supported image files in the input folder
2. **EXIF** — extracts camera metadata using Pillow (JPEG/TIFF) or exifread with a LibRaw fallback (RAW)
3. **Preview** — generates a 512px JPEG thumbnail for each file
4. **Analyze** — sends the preview to GPT-5.5 Vision with a photography critic prompt
5. **Parse** — extracts structured JSON: description, tags, score (1–1000), reasoning
6. **Export** — appends results to `results.csv` and `results.json`
7. **Summary** — prints top photos and Stories recommendations to the terminal

---

## Supported Formats

| Format | Camera Brand |
|--------|-------------|
| `.RW2` | Panasonic Lumix |
| `.ARW` | Sony |
| `.CR3` | Canon |
| `.NEF` | Nikon |
| `.JPG` / `.JPEG` | Any |
| `.TIFF` / `.TIF` | Any |

---

## Cost Estimate

Using GPT-5.5 with low-detail vision mode:

| Photos | Estimated Cost |
|--------|---------------|
| 1      | ~$0.001       |
| 100    | ~$1.00         |
| 1 000  | ~$10.00         |

---

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

---

## License

[MIT](LICENSE) © Vitalii Karasov
