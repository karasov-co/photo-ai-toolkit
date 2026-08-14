# How the score is built

Moved out of the README, which had grown to 949 lines. Nothing here is abridged.

## The thing it gets right that scoring tools get wrong

Almost every "photo culling AI" rates the file in front of it. That is the wrong
question, because **nobody publishes a RAW as shot.**

A dark, flat, tilted, noisy frame is not a bad photograph. It is an *unedited*
one, and two sliders away it may be the best thing you shot that week. A bright,
sharp, perfectly exposed frame whose subject missed focus is finished, and no
amount of processing will change that.

Those two files score almost identically on any single "quality" number, and
they belong in opposite piles.

So this tool asks one question and ranks everything by the answer:

> **How good can this photograph become after a normal edit?**

It does not guess. It actually applies a bounded set of plausible,
non-destructive edits to a working copy — exposure, white balance, highlight and
shadow recovery, straightening, several crops — re-scores each result and keeps
the best, then charges the result for what the edit cost: a heavy crop loses
resolution, a big exposure push loses latitude, and a JPEG gets far less credit
than a RAW because the data the recovery would use has already been thrown away
by the encoder.

On a real 45-frame RAW sample, editing gained a median of **12 points** and up
to **53**. Those are the frames a "current quality" score files in the bin.

### What that number is not

`frame_quality` is used twice, and you should know it before trusting the gain.
It is the objective the preview search hill-climbs on, *and* it is the ruler
that reports the result. So `uplift` measures how far the search moved a number
it was optimising — which is a weaker claim than "the photograph got better",
however plausible the first makes the second sound. A metric scoring its own
output will always report progress.

That circularity cannot be settled from the inside. It needs photographs a
person has ranked:

```bash
python3 cli.py bench-quality --analysis ./run/.internal/reports/analysis.json \
                             --labels ./labels.csv
```

with `filename,human_score` (or `human_rank`, or a `human_score_edited` column
for the gain half). It reports Spearman correlation against your ranking.

Until that has been run, every record carries `uplift_validated: false` and the
report says the gain is an estimate from an unchecked metric. **It is false
today.** No labelled set ships with this project, and the correlation has not
been measured — so treat the gain as a plausible ordering signal and not as a
measurement.

---

## Current quality is not potential

These are stored as different numbers and must never be collapsed:

| Dimension | Question it answers |
|---|---|
| `current_quality` | How the unedited file looks right now |
| `recoverability` | How safely normal editing can move it |
| `post_edit_potential` | What it becomes after a realistic edit |
| `aesthetic_potential` | Whether the result is worth looking at |
| `stock_potential` | Sellable and findable |
| `portfolio_potential` | Represents the photographer's best work |
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

The vision passes are not optional in `analyze`, and their availability is
verified before any photograph is opened.

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

**Nothing is blocked from stock any more.** A branch used to run before any
threshold: a face or a logo sent a photograph to a separate editorial pile,
off the model's answer to a question about paperwork it cannot answer. The
branch, the pile and the question are gone. The only thing that separates
*stock* from *personal* is whether the frame has a market.

**Flagship is not "the top 5%".** On a weak shoot the top 5% is still weak, so an
asset must clear an absolute floor *and* win a place. Selection is
diversity-aware (maximal marginal relevance over perceptual similarity and
genre), because ranking by score alone fills a portfolio gallery with twenty
frames of the same sunset.

---

## What a card tells you

```
┌──────────────────────────────┐
│         [ preview ]          │
│                              │
│ P1019399.JPG           75    │
│              after editing   │
│ technical quality now: 58    │
│                              │
│ a clean, legible photograph  │
│ that also works as stock     │
│                              │
│ • Adjust exposure: +0.4 EV   │
│ • Neutralise the cast        │
│ • Recover highlights, lift   │
│   shadows: moderate          │
│                              │
│ Edit recipe → edit_recipes/  │
└──────────────────────────────┘
```

One number, one sentence, three things to do. Nothing else — no confidence
interval, no routing class, no marketplace table. The two numbers measure
different things and are labelled so: how good the *photograph* is, and how
clean the *file* currently is.

---

## Your five piles

| | What it means |
|---|---|
| **Top** | The strongest work in the shoot. Start here. |
| **Good — stock** | A good photograph that also happens to be sellable |
| **Good — personal** | A good photograph worth keeping and printing |
| **Needs decision** | Genuinely borderline. A glance settles it. Rare by design |
| **Weak** | Blinks, dead moments, accidental frames, the weaker of two near-identical takes |

**Weak is a shelf, not a bin.** Nothing is deleted, ever, by an analysis run.
Deleting requires a separate command, a demonstrable unrecoverable fault, a
contact sheet you have to look at, and a script that moves files to the Trash
rather than calling `rm`. That ritual exists because an early version of the
technical filter marked four good photographs for deletion, and the only thing
that caught it was looking at them.

**No photograph is ever worse for a legal reason.** Whether a picture has a
market decides *stock* from *personal* and nothing else — it never lowers a
score, never blocks the top pile, never makes anything weak. There is
no legal vocabulary anywhere, in the report or in the code: no releases, no
trademarks, no editorial-only pile. A vision model cannot know whether a release
exists, and the guess used to decide which pile a photograph landed in. Somebody
photographing their own family is never told their picture has a licensing
problem, because nothing in here can form that thought.

---

