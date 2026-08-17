# What it does not do

Kept as its own page rather than as the second section of the README. It was
the second section for a while, and a front page that opens with eight
paragraphs of what a tool cannot do is not honesty, it is a tool nobody
reaches the third section of. None of it is softened here; it is just not
the first thing a stranger reads.

Everything below is linked from the README and from `photoai --help` where
it applies.

## The list

What a stranger should know before running this on their own archive.

**The five piles are not calibrated against anybody but the author.** The
thresholds (top at 85, weak under 45) were tuned on one person's archive. That
is the one genuinely weak claim in this repository, and it is measurable — an
evening of sorting answers it:

```bash
photoai bench-quality --analysis run/.internal/reports/analysis.json \
                      --template labels.csv        # blank sheet, shuffled
# fill human_pile with top / good / weak, 200-300 rows
photoai bench-quality --analysis run/.internal/reports/analysis.json \
                      --labels labels.csv --json agreement.json
```

It prints the agreement, a confusion matrix, precision and recall on the top
pile, and the thresholds that *would* have agreed best — which it does not apply
on its own, because a threshold fitted to one evening is that evening's
threshold. `reclassify --profile-file` re-sorts a finished run at different
numbers without spending a token, so trying the suggestion is free.

The sheet is shuffled and carries no score, on purpose: a sheet showing what the
tool decided measures how persuadable you are, not whether the thresholds are
right.

**Two of five provider adapters have met a live endpoint.** grok (the default,
281 photographs through it) and openai. anthropic, gemini and openai-compatible
are written from the documented request shape, have never been run, and say so
at startup.

**The gain figure compares frames; it does not measure photographs.** The metric
it comes from is both what the edit search optimises and the ruler that reports
the result, so "+12" means this frame has more room than that one, not that
editing makes a picture 12 points better. The report says so on every page, and
`bench-quality` against your own ranking is the only thing that changes it.

**Focus is confirmed at 100%, but the subject is not detected.** The 512px pass
finds the sharpest region and a second pass re-measures that region at native
resolution, which is what catches a frame that is smooth at preview size and
missed at full size. What it still does not do is know *what* should have been
sharp: a portrait focused on an ear instead of an eye has a genuinely sharp
region and passes. Subject and eye detection is the missing half.

**Near-duplicates are grouped by perceptual hash, not by content.** Two
different subjects with the same palette and framing can be merged, and the
weaker one filed as a repeat. There is no embedding model in here.

**The Adobe export is written from the documented format, not from watching it
load.** A user found that the sidecar was not a preset at all -- imported, it
appeared as `<x:xmpmet` with a slider that did nothing, and beside a JPEG
Lightroom ignored it outright. There is now a real preset per photograph, plus
star ratings and colour labels the catalogue reads, and every file is checked as
XML by the tests. What has still not happened is somebody watching Lightroom
open one. The darktable and RawTherapee renderers have never been run at all: no
binaries were available, and they report themselves `[unverified]`.

**The recipes correct, they do not style.** Exposure, white balance, highlight
and shadow recovery, some contrast and clarity, with the measurement that earned
each step. No HSL, no tone curve, no split toning. Next to a photographer's
preset the result will look like almost nothing happened, and that is the
design: a style applied to an unrelated photograph is a lie about the picture.
If you want a look, this is what goes under it.

**Nothing auto-deletes, and it is not one flag away.** Ten gates have to open
together — 1,000 recorded decisions, 3,000 holdout checks, a certified monitor,
and six more. `--no-shadow-mode` opens exactly one of them. `photoai policy`
prints the other nine and what each is waiting for.

**A score is one opinion.** Nothing is a verdict, nothing moves without
`--apply`, and every original file is untouched throughout.

---

