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
$ python cli.py analyze --input ~/Pictures/japan --output ./run

  [   1/47] P1019368.JPG
  ...
==================================================================
COLLECTION SUMMARY
==================================================================
  Analysis mode                               local + semantic

  Total assets                                     47

  What these photographs are:
    Top                                             3
    Good — stock                                   13
    Good — personal                                26
    Needs decision                                  0
    Weak                                            5

  Duplicate clusters                                8

  Top photos:
    [ 88] P1019412.JPG
    [ 86] P1019389.JPG
    [ 85] P1019399.JPG

  These are suggestions, not verdicts -- a score is one opinion
  about a photograph. Nothing has been moved, changed or deleted:
  the folders hold links to your original files.
==================================================================
```

and an output directory holding what you came for and nothing else:

```
run/
  report.html                  the five piles, ranked
  photographer_insights.html   what the collection says about your photography
  top/  good_stock/  good_personal/  needs_decision/  weak/
  edit_recipes/                one XMP sidecar per photograph worth editing
  .internal/                   previews, JSON, CSV, log, cache, diagnostics
```

The folders hold **symlinks**. Analysis never copies, moves, alters or deletes
an original.

---

## One number: potential after editing

Everything is ranked by one question -- *how good can this photograph become
after a normal edit?* -- and never by how the untouched file looks right now. A
dark, flat, tilted RAW that will come back beautifully outranks a bright JPEG
that will not, which is the entire point of the tool. Current quality is shown
beside it in small text, because you still want to know whether you are looking
at a finished frame or a raw one.

| Category | Meaning |
|---|---|
| `TOP` | The strongest work here. Shown first, and linked in `top/` |
| `GOOD — STOCK` | A good photograph that also works as stock material |
| `GOOD — PERSONAL` | A good photograph worth keeping. Not stock material |
| `WEAK` | Blinks, dead moments, accidental frames, weaker takes of a shot you already have |
| `NEEDS DECISION` | Genuinely borderline, and rare by design |

Three things the score deliberately does **not** contain: whether the photograph
can be licensed, whether anyone would buy it, and whether a stranger walked into
the frame. Those checks still run, and all they do is separate `GOOD — STOCK`
from `GOOD — PERSONAL`. They never lower a score, never block `TOP`, and never
make a photograph `WEAK`. Somebody photographing their own family should never
be told their picture has a licensing problem, so the default report contains no
legal vocabulary at all -- no releases, no trademarks, no editorial-only. It is
in `.internal/reports/analysis.json` and behind the report's *Expert details*
block for whoever actually sells work.

`WEAK` is a shelf, not a bin. No category is grounds for deletion; that still
requires a demonstrable unrecoverable fault recorded as evidence.

---

## Edit recipes

Every photograph scoring 70 or above gets its own Camera Raw sidecar in
`edit_recipes/`, built from that photograph's own measurements. One preset
cannot fit a collection -- the whole premise of the analysis is that these
frames differ -- so there is no single "look" to apply.

They are a **starting point, not a finished edit**: they undo what the camera
got wrong (exposure, colour cast, a tilted horizon, clipped highlights) and stop
there. `edit_recipes/HOW-TO-USE.txt` explains the Lightroom Classic import
(*Metadata > Read Metadata from File*) and what to do in Camera Raw, Capture
One, darktable and RawTherapee.

Nothing here writes beside your originals. A converter looks for `<stem>.xmp`
next to the file, so writing there would silently replace work you had already
done; these stay in the output directory until you choose to copy one across.

Where the analysis supports it, a frame is also offered up to three creative
directions -- documentary neutral, cinematic low key, warm autumn, cool winter,
restrained black and white. Each has to be *earned* by something measured: a
season applied to an unrelated photograph is a lie about the picture, so a frame
that earns none is offered none. No extra model calls are made for any of this.

---

## Photographer insights

`photographer_insights.html` reports patterns across the whole collection:
strongest genres, recurring visual habits, what you do reliably well, what is
costing you frames, and the three things most worth changing next -- each with
the count and the filenames behind it.

Per-file feedback teaches you about that file. "Eleven of your fourteen
portraits put the subject dead centre" teaches you something you can use on the
next shoot. Nothing is reported unless enough frames support it, there are no
invented quotations, and the reading list is a small fixed table of real
photographers and books to look up rather than a paragraph of invented
commentary about them.

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
- **unrecoverable** — missed focus, severe motion blur, blown highlights with no data, crushed shadows, insufficient resolution, corrupt file, no usable video segment

A **weaker duplicate** and a **short clip** are deliberately *not* in that list.
Sharpness picks the winner of a burst and cannot see which take has the better
expression; a three-second floor is a marketplace submission rule and not a
property of the footage. Both go to a review class, and neither can ever be
permanently purged.

---

## How the score is built

`final_score` (0–100) blends technical potential, the content pass, the artistic
read, the portrait analysis, documentary value and standing within the
collection. Two asymmetries hold it together, and they point in opposite
directions on purpose.

**Technical excellence cannot rescue a failed photograph.** Eyes shut, an
accidental frame, a dead moment, no subject, an unrecoverable fault — each
applies a *ceiling*, not a penalty. A penalty can be outvoted by a big enough
number elsewhere; a ceiling cannot, which is the whole reason for using one.

**Evidence can rescue an unconventional photograph.** A confident Stage 3 read
showing high documentary significance or distinctiveness applies a *floor*,
keeping a technically poor and commercially useless frame out of `WEAK`. It only
overrules a *guess* — "this looks like a dead moment" — never an observation like
a closed eye.

The same split runs through the portrait gate, and it looks inconsistent until
you notice which way each half errs. Declining to *promote* a frame costs
nothing, so the label `AWKWARD` alone keeps it out of `TOP`. Writing a frame off
costs the photograph, so `WEAK` needs the model's own numbers — expression
quality and publishability both low, at confidence — and not merely a word from
a list.

Every record carries `stage3_delta`: how many points the artistic read moved
that frame, positive or negative, measured by scoring it twice — once with the
read and once without. A run where that column is all zeros is a run where
Stage 3 did not happen.

---

## Route classes

Still here, and still what drives the filesystem: the category says what a
photograph *is*, the route class says what may be *done* with it.

| Class | Meaning |
|---|---|
| `trash` | Unrecoverable or corrupt. **Never a duplicate and never a short clip.** |
| `duplicate_candidate` | A sharper frame exists in the group — a comparison for a person, not a deletion |
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

## Stage 3: the artistic read

Sharpness, exposure and rarity are the three axes a technical pass can rank, and
none of them can tell whether a photograph is any good. Stage 3 is a separate
call that looks at a frame and answers eight questions the ranking cannot —
emotional resonance, visual tension, narrative openness, moment specificity,
formal coherence, distinctiveness, documentary significance, conventional beauty
— each with its own reasoning, plus a confidence in the whole read.

**Stage 3 is a term in the score, not a veto on the end of it.** The eight
dimensions are weighted into the final score, so a strong read raises a frame and
a weak one lowers it; `stage3_delta` reports by how much. It is also the only
thing that can authorise `TOP` or `flagship`: a pending, skipped, failed or
half-parsed assessment blocks promotion and caps the score at 79. That is the
rule that matters most here, because the failure it replaces was silent — the
artistic fields were `null` in every report while `flagship` was being assigned
from three technical axes alone.

The vision passes run by default. `--no-semantic` turns them off; `--semantic`
makes a missing key an error rather than a downgrade; with no key at all the run
says what it is skipping and continues locally.

The read is bounded, not universal. It runs on every keep and hero candidate, on
anything with a face, and on frames whose defect might be deliberate — a
motion-blurred pan and an intentional silhouette are exactly what a technical
filter judges worst. It does not run on a corrupt file or a confidently
unrecoverable one.

### Faces are a separate question

A blink is not an aesthetic property. When a face is the subject, the same call
returns eyes state, expression, expression quality, pose, occlusion, blink and
grimace probabilities, publishability, and — separately — how confident it is in
the expression reading. Closed eyes, a confidently bad expression, a soft face,
or low confidence each block promotion on their own; a beautiful photograph of a
bad moment is still a bad moment. Two guards keep that from overreaching: a face
occupying a fraction of a landscape is not a portrait and gates nothing, and
closed eyes can be overridden only when the artistic read *says* they are
deliberate — never by a high aesthetic score.

A confident bad expression is a decision, not a question for a person. Only an
uncertain one goes to review.

Stage 3 is cached separately from the semantic pass, keyed by checksum, model and
prompt version, so a valid Stage 2 entry can never stand in for a missing
artistic read.

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

`analyze` takes `--expert` to print the route classes, the two stock counters
and release status alongside the five piles. They answer a stock seller's
questions rather than a photographer's, which is why they are no longer in the
default summary — nothing was removed, and all of it is still in
`.internal/reports/analysis.json`.

Filtering and sorting on `report`: `--media`, `--route-class`, `--route`,
`--genre`, `--marketplace`, `--min-score`, `--min-potential`, `--min-confidence`,
`--needs-release`, `--cluster`, `--duplicates-only`, `--sort`, `--format`.

---

## Output

```
run/
  report.html                  the five piles, ranked by potential after editing
  photographer_insights.html   patterns across the whole collection
  top/  good_stock/  good_personal/  needs_decision/  weak/
  edit_recipes/                one .xmp per photograph scoring 70+, plus HOW-TO-USE.txt
  .internal/
    reports/analysis.json      every dimension, every reason, machine-readable
    reports/analysis.csv       the same, for a spreadsheet
    reports/full_report.html   the expert page: every score, every warning
    reports/insights.json
    reports/delete_candidates.txt   paths and reasons; nothing removed
    reports/delete.sh          moves to Trash, run by hand after looking
    reports/contact_sheet_delete.jpg
    routing/                   stock/, portfolio/, by_genre/, marketplace packages
    previews/  processing.log  analysis_cache.json  quarantine/
```

The category folders are a **symlink farm** pointing back at the originals. No
file is copied, moved or altered to produce it, and re-running rebuilds it from
`analysis.json` without decoding anything.

`.internal/` is hidden rather than absent. Every command that reads a previous
run — `report`, `reclassify`, `quarantine`, `restore` — reads it from there, and
an output directory from an older version is tidied into it automatically on the
next run.

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
cp .env.example .env         # then put your key in it, once
```

Edit `.env` and set your key:

```
OPENAI_API_KEY=your_key_here
```

That is the whole setup. **Do not `source .env` and do not `export` anything** —
the application loads the file itself, from the project root, no matter which
directory you run it from:

```bash
python cli.py --lang ru analyze --input ./photos --output ./run --semantic
```

`.env` is in `.gitignore` and is never committed. `.env.example` holds only a
placeholder.

### Where the key is looked for

In order, highest priority first:

1. a variable already set in your shell environment
2. `<project root>/.env`
3. `<current directory>/.env`, if that is a different directory

A variable exported in your shell always wins — `.env` never overrides it. Every
run with `--semantic` prints where the credential came from, and never the key:

```
Semantic credentials: loaded from /path/to/photo-ai-toolkit/.env
```

### The model

Precedence: `--model` → `OPENAI_MODEL` → the documented default
(`gpt-5.6-sol`). If that model is not available to your account, set
`OPENAI_MODEL` in the same `.env`. The model actually used is recorded in every
report.

---

## Two modes, and the difference matters

| Mode | What runs | What you get |
|---|---|---|
| **local-only** (default) | Decoding, technical metrics, edit-potential search, duplicate clustering | Technical routing. **Genre is `unknown`. Faces, logos and releases are not checked.** |
| **local + semantic** (`--semantic`) | The above, plus a vision model over 512px previews | Genre, concepts, faces, logos, release requirements, marketplace eligibility |

**What is sent to the API:** 512px JPEG previews of your photographs, in groups
of twelve, plus the measured clipping figures. No original file is uploaded. No
video frames are sent.

**Semantic costs money.** It is opt-in for that reason, and off by default.

### When the semantic pass fails

`--semantic` **fails fast**. If the key is missing, the run stops with a non-zero
exit code *before any photograph is decoded* — it does not spend minutes
measuring files and then present a local-only result as a finished analysis.

If the key exists but the API rejects it, the model is unavailable, or the reply
is unusable, the run also stops with a non-zero exit code. To accept a
local-only result instead, ask for it explicitly:

```bash
python cli.py analyze --input ./photos --output ./run --semantic --allow-semantic-fallback
```

That report is stamped `SEMANTIC ANALYSIS DID NOT RUN` in the console and at the
top of the HTML, and every record carries
`analysis_mode=local_only_after_semantic_failure`.

### Re-running after adding a key

Just run the same command again. Local measurements come from the cache; the
semantic pass runs. `--force` is not needed, and a local-only cache entry can
never be mistaken for a semantic result.

---

## Nothing moves without `--apply`

`analyze` only ever writes reports, symlinks and proposals. Moving files is a
separate command, and it is a dry run unless you pass `--apply`.

**The default pipeline is entirely local, offline and free.** It needs no API
key and makes no network calls; a culling tool that cannot run without a network
is not a culling tool. The paid vision pass is opt-in via `--semantic`, and
without it confidence is reduced — which the confidence score states rather than
hides.



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
pytest          # 1316 tests, no network, no API key
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
