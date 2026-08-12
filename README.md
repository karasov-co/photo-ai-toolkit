# photo-ai-toolkit

[![CI](https://github.com/karasov-co/photo-ai-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/karasov-co/photo-ai-toolkit/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

A culling, assessment and distribution tool for photo and video archives.

It answers one question that ordinary quality scoring gets wrong: **what is this
file worth after a normal edit?** A dark, tilted, flat, noisy RAW is not a bad
photograph — it is an unedited one. A sharp, bright, perfectly exposed frame
whose subject missed focus is finished, and no amount of processing changes
that. Those two files score almost identically on any single "quality" number,
and they belong in opposite piles.

---

## What it does

```
$ python cli.py analyze --input ~/Pictures/vietnam --output ./run

  [   1/64] P1042930.RW2
  ...
==============================================================
COLLECTION SUMMARY
==============================================================
  Total assets                                64
  Photos                                      60
  Videos                                       4

  Usable stock                                41
  Needs manual review                         20
  Flagship / portfolio                         1
  Trash / reject                               2

  Duplicate clusters                           3
  Low confidence                               2
  Space recoverable from quarantine       117.4 MB
==============================================================

1 file(s) would move, 19.3 MB:
      2  unrecoverable: unusable_duration: 1.5s is below 3.0s

Nothing has been moved. Re-run with --apply to carry this out.
```

Per file you get ten separate scores, the problems split into what an edit can
fix and what it cannot, a concrete edit recipe, a route class, marketplace
eligibility, and a proposed filesystem action that has not been carried out.

---

## Current quality is not potential

These are stored as different numbers and must never be collapsed:

| Dimension | Question it answers |
|---|---|
| `current_quality` | How the unedited file looks right now |
| `recoverability` | How safely normal editing can move it |
| `post_edit_potential` | What it becomes after a realistic edit |
| `aesthetic_potential` | Whether the result is worth looking at |
| `stock_potential` | Sellable, findable, legally publishable |
| `portfolio_potential` | Represents the photographer's best work |
| `legal_readiness` | Releases, trademarks, identifiable people |
| `uniqueness` | Against the rest of *this* collection |
| `confidence` | How much of the above is actually evidenced |
| `routing_score` | The single permitted blend, from the calibration profile |

On a real 45-frame RAW sample the split looks like this — current quality has a
median of 56, and editing realistically gains a median of 12 points and up to 53:

```
current    min  17.2   p50  56.3   max  74.5
potential  min  49.4   p50  68.4   max  87.4
gain       min   0.0   p50  12.0   max  53.1
```

### How potential is estimated

By actually trying. A bounded search applies a handful of plausible,
**non-destructive** edits to a 512px working copy — exposure, white balance,
highlight and shadow recovery, contrast, straightening, several saliency-aware
crops, restrained denoise — re-scores each, and keeps the best. The uplift is
then charged for what it cost: heavy crops lose resolution, large exposure
pushes lose latitude, denoising loses detail, and a JPEG is credited with far
less recovery than a RAW because the data the recovery would use has already
been discarded by the encoder.

What is stored is a **recipe**, never a modified pixel:

```
Adjust exposure: +1.4 EV
Correct white balance: neutralise the cast, damped
Recover highlights and lift shadows: moderate
Crop: tight crop -- keep 78% of the frame
Straighten: rotate 3.4° clockwise
Do not apply aggressive sharpening: halos cost more than they buy
```

Nothing generative is ever proposed. That is both an honesty requirement and a
marketplace one — see *Provenance* below.

### The asymmetry that matters

An unrecoverable problem **caps** post-edit potential rather than penalising it.
Raising the exposure on an out-of-focus frame genuinely does improve it, and
without a hard ceiling that measured uplift would promote a photograph whose
subject will never be sharp.

Problems are typed at detection time, not inferred afterwards:

- **fixable** — underexposure, colour cast, tilt, weak crop, edge clutter, mild noise, unusable audio, needs trim
- **partially fixable** — heavy noise, some clipping, moderate shake (stabilisation costs a crop), soft focus, compression artifacts, rolling shutter
- **unrecoverable** — missed focus, severe motion blur, blown highlights with no data, crushed shadows, insufficient resolution, corrupt file, no usable video segment, weaker burst duplicate

---

## Route classes

| Class | Meaning |
|---|---|
| `trash` | Unrecoverable, corrupt, or a measurably weaker duplicate |
| `review` | Low confidence, conflicting signals, or below the stock floor but worth keeping |
| `stock_standard` | Becomes acceptable stock material after the suggested edit |
| `stock_strong` | Strong commercial stock potential |
| `flagship` | Portfolio-grade: memorable, recoverable, distinctive |

Precedence is explicit and ordered in `scoring._decide`. Two rules are worth
stating out loud:

**Faces or logos physically cannot reach commercial stock.** Not a weighting — a
branch that runs before any threshold. Both need a release, and submitting them
without one earns rejections in batches. The model is asked for the two flags
and is *not* trusted to act on them. When no vision pass has run, the release
status is *unknown*, which still blocks commercial stock but is reported as
"unchecked" rather than as "a face is present".

**Flagship is not "the top 5%".** On a weak shoot the top 5% is still weak, so an
asset must clear an absolute floor *and* win a place. Selection is
diversity-aware (maximal marginal relevance over perceptual similarity and
genre), because ranking by score alone fills a portfolio gallery with twenty
frames of the same sunset.

---

## Video

A clip is not a photograph with a duration, and its first frame is routinely its
worst — the operator is still settling the camera. Analysis samples across the
whole timeline, then looks *between* frames at a few positions for anything that
only exists there.

Camera motion is measured by phase correlation between consecutive frames, which
separates the two cases that matter: a smooth accumulating shift is a **pan** and
is intentional, while the same magnitude jittering around zero is **shake** and
is a defect. Anything that scores raw inter-frame difference calls a good pan
unusable.

Also derived: container and codec facts, variable frame rate, log/HDR clues,
audio presence, focus consistency, exposure drift, flicker, black and frozen
frames, rolling-shutter shear, usable segments with edit handles, and the
strongest frame as a poster.

---

## Safety

**Nothing is ever deleted automatically, and nothing is moved without being asked
twice.**

- Every filesystem command is a **dry run** unless `--apply` is passed
- Rejects are **moved to quarantine**, never deleted, preserving their original directory structure
- A RAW moves together with its JPEG twin and its `.xmp`, or not at all
- Operations are **idempotent** by checksum — re-running an interrupted move is a no-op
- Collisions never overwrite: `name.jpg` becomes `name_1.jpg`
- Both ends are fenced against path traversal, checked *after* `resolve()`
- Symlinks are never followed into a move
- Every move is recorded in an append-only manifest with enough to undo it
- `restore` puts everything back exactly where it came from
- The generated `delete.sh` moves files to the Trash — it never runs `rm`

Permanent deletion is a separate operation behind four gates: a typed
confirmation phrase, a minimum quarantine age, an unlocked directory, and a
manifest entry proving the file was quarantined by this tool rather than merely
present in the folder.

```bash
python cli.py quarantine --analysis run/reports/analysis.json \
    --quarantine ./quarantine --input ~/Pictures/vietnam          # dry run
python cli.py quarantine ... --apply                              # moves
python cli.py restore --quarantine ./quarantine --apply           # undoes
python cli.py purge --quarantine ./quarantine \
    --confirm 'PERMANENTLY DELETE' --older-than 30 --apply        # last resort
```

---

## Commands

| Command | What it does |
|---|---|
| `analyze` | Measure, score, route, write reports and a plan |
| `report` | Filter, sort and re-render a stored run |
| `reclassify` | Redo routing with different thresholds — no re-analysis, no tokens |
| `quarantine` | Carry out a plan (dry run unless `--apply`) |
| `restore` | Undo a quarantine operation |
| `purge` | Permanently delete, behind four gates |
| `export` | Build a marketplace-ready package |
| `override` | Record a manual decision future runs must respect |
| `profiles` / `validate-profile` | Inspect and dump calibration profiles |
| `darkroom` | Show the edit suggestions from a stored run |
| `apply-recipe` | Write a recipe beside the RAW (dry run; refuses to clobber) |
| `ask` | The questions worth five minutes, most informative first |
| `record` | Record one decision for the personal model |
| `policy` | What would be automated, and what is holding each gate shut |
| `monitor` | False-trash rate, drift, calibration; can switch automation off |
| `trash` | Carry out `delete_plan.json` in Python (dry run unless `--apply`) |

Filtering and sorting on `report`: `--media`, `--route-class`, `--route`,
`--genre`, `--marketplace`, `--min-score`, `--min-potential`, `--min-confidence`,
`--needs-release`, `--cluster`, `--duplicates-only`, `--sort`, `--format`.

---

## Output

```
run/
  reports/
    analysis.json          every dimension, every reason, machine-readable
    analysis.csv           the same, for a spreadsheet
    report.html            a page with the previews on it
    distribution.csv       one row per file with a `destination` column
    delete_candidates.txt  paths and reasons; nothing removed
    delete.sh              moves to Trash, run by hand after looking
    contact_sheet_delete.jpg
  previews/
  trash_quarantine/  manual_review/
  stock/{standard,strong,editorial,by_genre,marketplace_packages}/
  portfolio/{flagship,by_genre}/
  archive/
```

Everything under the class folders is a **symlink farm** pointing back at the
originals. No file is copied or moved to produce it.

`contact_sheet_delete.jpg` exists because looking at a grid of what you are about
to lose is the practice that caught four good photographs an earlier version of
the technical filter had marked for deletion. Fog and haze have no
high-frequency content, so absolute sharpness reads them as out of focus; the
fix was to threshold on **blur ratio** — the frame against a blurred copy of
itself — which is independent of scene contrast.

---

## Calibration

Weights and thresholds are data, not code. Routing reads only stored dimensions,
so retuning is instant and free on a collection that cost an hour and real money
to analyse the first time:

```bash
python cli.py profiles --name stock-first --dump my-profile.json
# edit thresholds
python cli.py reclassify --analysis run/reports/analysis.json --profile-file my-profile.json
```

Built-ins: `default-photo`, `default-video`, `stock-first`, `portfolio-first`.
Photo and video carry separate thresholds because a threshold tuned on stills
routes clips wrongly.

> **The shipped numbers are provisional.** They are starting points measured
> against one archive, not fitted against a labelled set, and every profile says
> so in its version string. Treat them as a beginning.

### Manual overrides win, permanently

```bash
python cli.py override --analysis run/reports/analysis.json P1042721.RW2 \
    --set-class flagship --note "fog, I want it"
```

Overrides are keyed by content checksum, so they survive renames and
reorganisation, and they are re-applied after every analysis. The tool's own
conclusion is kept beside the override rather than erased, which makes the
disagreement auditable — and is the shape of data a future personalised
calibration would learn from.

---

## Marketplaces

Rules live in [`data/marketplace_rules.json`](data/marketplace_rules.json), each
platform carrying the source URL it came from and the date it was last checked.
Covered: Adobe Stock, Shutterstock, Getty/iStock, Alamy, Pond5, Dreamstime,
Depositphotos, 123RF.

The AI policies genuinely differ and the difference matters:

- **Adobe Stock** accepts generative content when correctly declared
- **Shutterstock** does not accept contributor-submitted AI from external tools, and scans metadata (C2PA, IPTC `DigitalSourceType`, XMP/EXIF signatures) — an undeclared match is auto-rejected and can carry an account strike
- **Alamy** classifies AI-generated imagery as unsuitable material, but explicitly permits AI tools that mimic conventional retouching such as denoise

### Provenance is declared, never guessed

There is no reliable way to look at an image and know whether a generative tool
touched it, and a wrong guess is expensive in both directions. Positive evidence
is read from metadata; its absence proves nothing, because metadata is trivially
stripped. Undeclared provenance is reported as undeclared.

### Export, not upload

Nothing is ever submitted anywhere. `export` builds a package with a submission
CSV, per-file XMP sidecars, a release checklist and manual upload instructions.
Direct upload adapters are **not implemented** — see *Known limitations*.

---

## Metadata

Generated for stock-ready assets: title, description, ordered keywords, primary
and secondary category, concepts, location, commercial/editorial route, people
count, release requirements, trademark warning, AI label, suggested platforms.

Keywords come only from what something actually observed, are deduplicated,
ordered by centrality, and capped **well below** the platform maximum.
Irrelevant keywords degrade search for the buyer and are treated as spam at
account level, not per file. `keyword_confidence` reports how much of the list
is model-derived rather than inferred, so a thin result is visible rather than
padded.

Originals are never modified. Metadata is written to sidecars or derived export
copies. GPS coordinates are stripped from export copies by default — a home, a
school, a route walked daily.

---

## Install

Requires Python 3.12+. FFmpeg is optional and only needed for video.

```bash
pip install -r requirements.txt
brew install ffmpeg          # macOS; apt-get install ffmpeg on Debian/Ubuntu
cp .env.example .env         # only needed for the optional vision pass
```

**The default pipeline is entirely local, offline and free.** It needs no API
key and makes no network calls; a culling tool that cannot run without a network
is not a culling tool. The paid vision pass is opt-in via `--semantic`, and
without it confidence is reduced — which the confidence score states rather than
hides.

```bash
python cli.py analyze --input ./photos --output ./run              # local only
python cli.py analyze --input ./photos --output ./run --semantic   # + vision
python cli.py --lang ru analyze --input ./photos --output ./run    # Russian UI
```

### Models and licences

| Component | Licence | Notes |
|---|---|---|
| Pillow | MIT-CMU | Decoding, previews, filters |
| NumPy | BSD-3-Clause | All deterministic metrics |
| rawpy / LibRaw | MIT / LGPL-2.1 | RAW decoding |
| exifread | BSD-3-Clause | RAW EXIF |
| imagehash | BSD-2-Clause | Perceptual hashing |
| FFmpeg | LGPL-2.1+ / GPL | Video probe and frame extraction; optional, invoked as a subprocess |
| OpenAI API | commercial terms | Optional semantic pass only |

No non-commercial or research-only weights are used, and no model weights are
downloaded. Every deterministic metric is plain NumPy: Laplacian variance,
structure tensors, phase correlation, perceptual hashing, Bradley–Terry fitting.
There is no GPU path and none is needed.

---

## Development

```bash
pip install -r requirements-dev.txt
pytest          # 1020 tests, no network, no API key
ruff check .
```

The suite is hermetic: `conftest.py` turns any outbound socket into a loud
failure and strips API keys from the environment. Every fixture image is
generated from a seed — no private media is committed.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

---

## Known limitations

- **The calibration is provisional.** Thresholds have not been fitted against a hand-labelled set. Verify against your own archive before trusting the classes.
- **No direct marketplace upload.** Export packages only. No platform in the matrix offers a contributor upload API this project is authorised to use, so implementing one would mean inventing it.
- **Genre and concepts require the vision pass.** Offline, everything is `other`, and stock/portfolio scoring falls back to technical potential at reduced confidence.
- **No learned quality models.** PyIQA, Q-Align, TOPIQ, MUSIQ, CLIP-IQA, DOVER and COVER were evaluated and not integrated: their weight licences are inconsistent and several are research-only. The provider interfaces exist; the integrations do not, and the tool does not pretend otherwise.
- **No CLIP/SigLIP embeddings.** Duplicate and diversity similarity uses perceptual hashing plus genre. `duplicates.embedding_similarity` is the injection point for a real embedding.
- **Tilt estimation declines on scenes without straight lines.** Foliage and open water have a dominant direction only by accident; the estimator returns "no tilt" rather than a confident wrong angle.
- **Subject-relative sharpness is approximated** by the sharpest tile, not by detecting the subject. A portrait with a sharp face and a blurred background scores correctly; a frame where the *wrong* object is sharp will not be caught.
- **Faces, logos and identifiable people come from the vision model**, not a detector. Offline they are unknown, which blocks commercial stock by design.
- **Analysis is single-process.** Roughly 1.5s per RAW, dominated by decoding; the darkroom pass adds about a second per frame.
- **The darkroom mixes two domains on purpose.** Display appearance says what the frame visually lacks and is necessarily measured on the developed preview; RAW capacity says how far that can safely be corrected and is measured on the sensor plane; render validation says whether it actually helped. The sensor data *bounds* the tonal moves rather than originating all of them.
- **The darktable and RawTherapee adapters have never been executed.** Neither binary was available; they are written from the documented CLIs and report themselves `[unverified]`.
- **The skin-hue check is an HSV heuristic** and sand, wood and sunsets fall in the same band. Without a confirmed face it is advisory and does not veto.
- **The adaptive loop runs in shadow mode.** It records what it would do; `--no-shadow-mode` is not enough to make it act, because nine other gates and a healthy monitor are also required.

---

## Disclaimer

Scores are **recommendations**, not guarantees of artistic quality, marketplace
acceptance, or sales. Release and intellectual-property detection is **advisory
and does not replace legal review**. Marketplace policies change without notice;
every rule in `data/marketplace_rules.json` carries the date it was last checked
and must be re-verified before you rely on it.

---

## Legacy entry point

`main.py` is the original single-photo describe-and-score tool and still works
unchanged:

```bash
python main.py --input ./photos --output ./results [--dry-run] [--force]
```

It produces one absolute 1–1000 score per photo. That approach is why this
project moved to ranking: every live call against a real archive came back 548,
560, 694, 762 — a scale that does not discriminate is not a scale.

---

## Contributing

Issues and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).
Participants are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

[MIT](LICENSE) © Vitalii Karasov
