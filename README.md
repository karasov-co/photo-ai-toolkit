# photo-ai-toolkit

AI-powered CLI tool that analyzes, tags, and scores your photos using GPT-5.4 Vision.

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
- Sends previews to GPT-5.4 Vision for scene analysis
- Returns description, up to 10 tags, and a 1–1000 quality score per photo
- Exports results to CSV and JSON
- Highlights best shots for Instagram Stories (score ≥ 700)
- Skips already-processed files — safe to re-run on the same folder
- Retries on API errors with exponential backoff

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI API key
python main.py --input ./photos --output ./results
```

---

## How It Works

1. **Scan** — finds all supported image files in the input folder
2. **EXIF** — extracts camera metadata using Pillow (JPEG/TIFF) or rawpy (RAW)
3. **Preview** — generates a 512px JPEG thumbnail for each file
4. **Analyze** — sends the preview to GPT-5.4 Vision with a photography critic prompt
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

Using GPT-5.4 with low-detail vision mode:

| Photos | Estimated Cost |
|--------|---------------|
| 1      | ~$0.004        |
| 100    | ~$0.40         |
| 1 000  | ~$4.00         |

---

## License

MIT
