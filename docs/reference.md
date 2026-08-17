# Commands, output and configuration

Moved out of the README, which had grown to 949 lines. Nothing here is abridged.

## Commands

| Command | What it does |
|---|---|
| `analyze` | **The command.** Preflight, measure, content check, artistic read, report |
| `measure` | Developer tool: local measurement only, explicitly not an analysis |
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

`analyze` takes `--insights-scope all` to describe the whole archive rather than
just this batch, and `--expert` to print the route classes and the two
stock counters alongside the five piles. They answer a stock seller's
questions rather than a photographer's, which is why they are no longer in the
default summary — nothing was removed, and all of it is still in
`.internal/reports/analysis.json`.

Filtering and sorting on `report`: `--media`, `--route-class`, `--route`,
`--genre`, `--marketplace`, `--min-score`, `--min-potential`, `--min-confidence`,
`--cluster`, `--duplicates-only`, `--sort`, `--format`.

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

> **The shipped numbers are a starting point, and they are meant to move.**
> `bench-quality --from-catalog` reads the stars you gave a finished run and
> reports which thresholds would have matched your sorting more closely;
> `reclassify --profile-file` applies them without spending a token. Every
> profile carries a version string so a run records which numbers produced it.

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
CSV, per-file XMP sidecars and manual upload instructions.
Direct upload adapters are **not implemented** — see *Known limitations*.

---

## Metadata

Generated for stock-ready assets: title, description, ordered keywords, primary
and secondary category, concepts, location, people count, AI label and
suggested platforms.

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

## Edit recipes you can actually use

Every photograph scoring 70 or above gets its own **Camera Raw sidecar** in
`edit_recipes/`, built from that frame's own measurements. There is no single
"look" to apply, because the whole premise of the analysis is that these frames
differ — one is two stops under, one has a cast, one is three degrees off level
and one is fine.

In Lightroom Classic: copy the `.xmp` next to the photograph, select it, then
**Metadata → Read Metadata from File**. The sliders move. Undo works normally.
`edit_recipes/HOW-TO-USE.txt` covers Camera Raw and Bridge too, and says which
route works for JPEG and which does not.

They are a **starting point, not a finished edit**: they undo what the camera
got wrong and stop there. Every creative decision is still yours.

Nothing is ever written beside your originals — a converter looks for
`<name>.xmp` next to the file, so writing there would silently replace work you
had already done.

Where the frame supports it you also get up to three creative directions —
documentary neutral, cinematic low key, warm autumn, cool winter, restrained
black and white. Each has to be *earned* by something measured: a season applied
to an unrelated photograph is a lie about the picture, so a frame that earns
none is offered none.

---

## And a page about your photography, not your photographs

`photographer_insights.html` looks across the whole collection. Real output from
a real run:

> **Your visual habits**
> You photograph people inside their surroundings rather than close in on the
> face — the setting is doing as much work as the subject.
> *22 of 27 portraits put the face under 4% of the frame*
>
> **What is costing you frames**
> You shoot several near-identical frames of the same thing and keep them all.
> The extra takes are not adding a better version — they are adding work later.
> *10 of 47 frames lost to a sharper sibling*
>
> **The three things worth changing next**
> 1. Before you press the shutter on the obvious view, take two steps in any
>    direction and look again. *distinctiveness averages 46 of 100*
> 2. Wait longer in the good light. Most of these frames record a place
>    correctly; the strongest ones caught something happening in it.

Every line names the number and the filenames behind it. Nothing is reported
unless enough frames support it, and there is no generic advice — you will not
be told to use the rule of thirds. The reading list is a short fixed table of
real photographers and books to look up, with no invented commentary about them.

---

