# Internals, safety and limits

Moved out of the README, which had grown to 949 lines. Nothing here is abridged.

## For the engineer reading this

The design is defensive in one direction on purpose, and the asymmetry runs
through every decision:

**A false "weak" is unrecoverable; a false "keep" costs a folder.** So a
photograph is never destroyed on a low score. Destruction requires a
*demonstrable* fault — missed focus, a corrupt file, no usable video segment —
recorded as machine-readable evidence, and aesthetic judgements are deliberately
not representable in that field.

**Missing data is never a low score.** Every model output is either a validated
number or an explicit status saying why there is not one. A frame cannot reach
the top pile because the analysis that would have judged it timed out — an
unfinished artistic read is itself a blocker.

**Defects apply a ceiling, not a penalty.** A penalty can be outvoted by a big
enough number elsewhere; a ceiling cannot. That is the only reliable way to stop
a technically immaculate photograph of a blink from ranking well.

**Evidence can rescue, opinion cannot.** A confident artistic read applies a
floor that keeps an unconventional frame out of the weak pile — but it only ever
overrules a *guess* ("this looks like a dead moment"), never an *observation*
("the subject's eyes are closed").

Concretely, the pipeline is ordered so the cheapest thing that can eliminate
work runs first — checksum and grouping, then local decoding and measurement,
then clustering, and only then the paid vision passes. Everything after scoring
reads stored numbers, which is what makes `reclassify` free: change a threshold
and redo the routing in milliseconds without decoding a pixel or spending a
token.

Some of the load-bearing details, each of which exists because the naive version
failed on a real archive:

- Tilt from a **block structure tensor** with a coherence gate, not per-pixel
  gradients — the latter returned 0.5° for every input.
- Camera motion by **phase correlation** between frames, which separates a pan
  from shake; anything scoring raw inter-frame difference calls a good pan
  unusable. Bursts are analysed independently, because measuring across a burst
  boundary reported 22.5px of shake on a clean pan.
- Halo detection by **local-deviation growth**, not mean brightening — unsharp
  overshoot cancels out in the mean.
- Assets keyed on **relative path**, not basename: two memory cards both hold
  `P1000001.RW2`, and keying by filename merged two different photographs.
- Stage 2 **ranks within a group** rather than scoring absolutely, because every
  live absolute call against this archive came back 548, 560, 694, 762. A
  near-ranking with one duplicated rank is repaired rather than discarded —
  discarding one cost twelve photographs their content analysis.
- Stage 3 is cached separately from Stage 2, keyed by checksum, model *and*
  prompt version, so a valid content result can never stand in for a missing
  artistic one.
- Deletion plans are **JSON executed by Python**, never a generated shell
  script: a filename containing a quote character was enough to inject commands
  into the first version.

**1,357 tests**, no network, no API key: `conftest.py` turns any outbound socket
into a loud failure and redirects the project root so the suite cannot read a
real `.env`. Every fixture image is generated.

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

## The model, and why there is no fallback

`grok-4.6` on xAI, verified against your key before any work starts.
Configurable via `--model` or `OPENAI_MODEL`, and **never substituted**: if the
configured model is unavailable the run stops and tells you so.

The key comes from `XAI_API_KEY`, falling back to `OPENAI_API_KEY` so an
existing key keeps working without being renamed. Where both are set, the xAI
one wins.

Other providers are selectable with `--provider`: `openai`, `anthropic`,
`gemini`, or `openai-compatible` with a `--base-url` for vLLM, Ollama or LM
Studio. **Only the OpenAI path has been run against a live endpoint** — the
others are written from each vendor's documented request shape, and the CLI
says so when you pick one.

That is deliberate, and it is the one place this tool refuses to be helpful. An
older model would produce a report that looks identical and means something
different — the artistic read is the whole product, and it does not survive a
model generation. Silently downgrading would hand you a worse analysis under
the name of the one you asked for, and you would have no way to tell.

So there is no automatic fallback, no legacy family in any code path, and no
error message that suggests one. If your key lacks access, the fix is to the
account, and the tool says exactly that.

**What is sent:** 512px JPEG previews, in groups of twelve, plus the measured
clipping figures. No original file is uploaded. No video frames are sent. The
preflight sends a generated 32×32 test image and never one of your photographs.

**What it costs:** a few cents per hundred photographs, and only for the ones
that are new. The run prints the number of calls it made.

### `measure`: local only, for developers

```bash
python3 cli.py measure --input ./photos --output ./run
```

Decoding, technical metrics, the edit-potential search and duplicate clustering,
with no API key and no network. It exists because the deterministic half of the
pipeline is genuinely useful for debugging, and it announces in its own output
that it is **not a photo analysis**: no content check, no artistic read, and
nothing can be ranked as a top photograph. It is not `analyze`, and it never
produces a report that claims to be one.

---

## Nothing moves without `--apply`

`analyze` only ever writes reports, symlinks and proposals. Moving files is a
separate command, and it is a dry run unless you pass `--apply`.

**The deterministic half still runs offline**, under `measure`, for development
and debugging. It is not `analyze` and does not claim to be an analysis.



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
pytest          # 1357 tests, no network, no API key
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
- **Genre, content and the artistic read require the vision pass.** Offline the genre is `unknown` rather than `other`, nothing can reach the top pile, and scoring falls back to technical potential at reduced confidence.
- **The top pile can be empty, and that is not a bug.** It starts at 85, an absolute bar rather than a top 5%: on a weak shoot the top 5% is still weak. On the 47-frame archive this was built against, the strongest photograph scored 75 and the report says so in those words. If you would rather the bar were calibrated to your own archive, it is one number in `curation.CurationThresholds`.
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

