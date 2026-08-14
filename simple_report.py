"""The report a photographer opens, and the only thing they should have to read.

The page this replaces showed both systems at once: a category and a routing
class, a potential score and a routing score and a stock score and a portfolio
score and a confidence, plus release status, trademark warnings, marketplace
eligibility, internal tags and darkroom engine versions. Twelve numbers per
photograph, several of which contradict each other by design -- `stock_potential`
is *supposed* to disagree with `portfolio_potential` -- and no indication of
which one answers the question the reader actually has.

So this page answers one question, in one number, in one order:

    how good can this photograph become after a normal edit?

Everything that is not that lives in `.internal/` and in an Expert details block
that starts closed. Three rules hold the page honest:

**One ranking.** Every section is sorted by potential after editing. Current
quality appears in small grey text beside it, because a photographer needs to
know whether they are looking at a finished frame or a raw one -- but it never
decides an order, and a dark, flat, tilted file that will come back beautifully
outranks a bright one that will not.

**No legal vocabulary.** Not "model release", not "editorial only", not
"trademark". Those checks no longer exist at all: the model was being asked a
legal question it cannot answer, and the answer decided which pile a photograph
landed in. Somebody shooting their family should never have been told their
picture has a licensing problem, and now nothing can.

**No scores without a sentence.** Every card carries one plain explanation of
why it landed where it did, and up to three concrete things to do in the edit.
A number with no reason cannot be argued with, and a tool a photographer cannot
argue with is one they stop trusting the first time it is wrong.
"""

from __future__ import annotations

import html
from pathlib import Path

from curation import DEFAULT_THRESHOLDS as CURATION_THRESHOLDS
from i18n import t

# The order the piles are shown in, and the accent each gets.
SECTIONS = (
    ("TOP", "#a978c4"),
    ("GOOD_STOCK", "#5fb98a"),
    ("GOOD_PERSONAL", "#6aa9d8"),
    ("NEEDS_DECISION", "#d8c06a"),
    ("WEAK", "#8a8a8a"),
)

# Phrases that must never reach the default page. Asserted by a test rather
# than trusted: every one of them was on the previous version, and the way they
# come back is a helper being reused rather than anyone deciding to add them.
FORBIDDEN_IN_DEFAULT_UI = (
    "model release",
    "property release",
    "editorial only",
    "legal readiness",
    "trademark",
    "commercial blocker",
    "routing score",
    "stock potential",
    "portfolio potential",
    "flagship",
    "stage3",
    "marketplace",
)

# Words that make an ordinary photograph sound like a liability. A person
# photographing their own family has no licensing question to answer, and being
# told they do is the failure this list exists to prevent. Kept separate from
# the list above because these are *only* filtered out of sentences -- the
# checks behind them still run, and still decide stock from personal.
LEGAL_WORDS = (
    "editorial",
    "release",
    "licen",  # licence, license, licensing
    "commercial",
    "cleared",
    "rights",
)

# `editorial` is back on the forbidden list above: the bucket it named is gone,
# so any appearance of the word is a leak from something that should have been
# deleted.

MAX_RECOMMENDATIONS = 3

# Read from the categoriser rather than repeated here: the page has to say the
# same number the decision used, or its explanation of an empty TOP section is
# fiction the moment somebody tunes the threshold.
TOP_THRESHOLD = CURATION_THRESHOLDS.top
WEAK_THRESHOLD = CURATION_THRESHOLDS.weak


# Vanilla, inline, no libraries. A report that fetches anything is a report that
# stops working on a plane, behind a firewall, or in five years -- and the whole
# point of the folder is that it survives being moved and emailed.
SCRIPT = """
(function () {
  var box = document.getElementById('lightbox');
  var img = box.querySelector('img');
  function close() { box.hidden = true; img.removeAttribute('src'); }
  document.addEventListener('click', function (e) {
    var thumb = e.target.closest('.card img');
    if (thumb) { img.src = thumb.dataset.full || thumb.src; box.hidden = false; return; }
    if (e.target.closest('#lightbox')) { close(); return; }
    var chip = e.target.closest('.chip');
    if (!chip) return;
    var want = chip.dataset.bucket;
    document.querySelectorAll('.chip').forEach(function (c) {
      c.classList.toggle('on', c === chip);
    });
    document.querySelectorAll('section[data-bucket]').forEach(function (s) {
      s.hidden = want !== 'ALL' && s.dataset.bucket !== want;
    });
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !box.hidden) close();
  });
})();
"""


def potential(record) -> int:
    """The one number. Named here so nothing has to guess the field."""
    return int(record.final_score or 0)


def current_quality(record) -> int:
    """How clean the unedited file is. Secondary, always, and a different axis.

    Labelled "technical quality" rather than "as shot" on purpose. The numbers
    on a card measure different things -- one is how good the photograph is,
    another is how clean the file is, a third is what is in the frame -- and a
    technically pristine picture of nothing legitimately shows 87 here beside a
    67 above it. Calling them "as shot" and "after editing" made that pair read
    as the same measurement taken twice, which says editing made the photograph
    worse. "now" went too: it implied the number would change, and it does not.
    """
    return int((record.scores or {}).get("current_quality", 0))


def content(record) -> int | None:
    """What is in the frame, or None when nothing looked.

    This is the content axis of the Stage 2 read, aggregated across the groups a
    photograph appeared in. It was already computed and already fed two other
    dimensions; it was simply never carried out to where a person could see it.
    None rather than 0 for an offline run: no number at all is honest, and a
    zero would read as a verdict.
    """
    value = int((record.scores or {}).get("content", 0))
    return value or None


def explanation(record, language: str = "en") -> str:
    """One sentence, in the reader's language, with no jargon in it.

    Falls back through the localised category reason, then the plain English
    one. An empty string is better than a leaked internal phrase, so anything
    that mentions the machinery is dropped rather than shown.
    """
    for text in list(record.category_reasons or []) + list(record.reasons or []):
        cleaned = _plain(str(text))
        if cleaned:
            return cleaned
    return t(f"category.{record.category}", language) if record.category else ""


def _plain(text: str) -> str:
    """Trim an internal sentence to its useful half, or drop it entirely.

    Trimming first is what makes this worth doing. The internal reasons are
    written as `<the finding> -- <the machinery>`, and the half before the
    dash is usually exactly what a person needs: "final score 66: worth
    keeping" is a good sentence, and "Not for stock, a model release is
    required" is the half that has to go.

    What survives the trim is then checked as a whole sentence, so a photograph
    legitimately *of* a shop sign is not filtered because its description
    mentions a trademark. Anything still carrying internal vocabulary is
    dropped rather than reworded -- rewording would be guesswork, and the next
    candidate reason is usually fine.
    """
    for separator in (" -- ", " — ", ". Not for stock"):
        if separator in text:
            text = text.split(separator)[0]
    lowered = text.lower()
    if any(phrase in lowered for phrase in FORBIDDEN_IN_DEFAULT_UI + LEGAL_WORDS):
        return ""
    return text.strip()


def recommendations(record, limit: int = MAX_RECOMMENDATIONS) -> list[str]:
    """Up to three things to actually do, from the recipe the run produced."""
    out: list[str] = []
    for step in record.edit_recipe or []:
        step = str(step).strip()
        if step and step not in out:
            out.append(step)
        if len(out) >= limit:
            break
    return out


def ordered(records) -> list:
    """Every asset, best potential first. The only ordering in the report."""
    return sorted(records, key=lambda r: (-potential(r), r.filename))


def by_section(records) -> list[tuple[str, str, list]]:
    """(category, accent, records) for each pile, each internally ranked."""
    ranked = ordered(records)
    return [
        (name, accent, [r for r in ranked if r.category == name])
        for name, accent in SECTIONS
    ]


# --- the page -----------------------------------------------------------------


STYLE = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { background:#121212; color:#ececec; margin:0; padding:32px 24px 64px;
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
header { max-width:1180px; margin:0 auto 28px; }
h1 { font-size:24px; margin:0 0 6px; font-weight:600; }
.lede { color:#9d9d9d; font-size:14px; max-width:72ch; margin:0; }
.counts { display:flex; flex-wrap:wrap; gap:10px; margin:22px 0 0; }
.count { background:#1d1d1d; border-radius:10px; padding:10px 16px; min-width:132px; }
.count b { display:block; font-size:22px; font-weight:600; }
.count span { color:#9d9d9d; font-size:12px; letter-spacing:.02em; }
main { max-width:1180px; margin:0 auto; }
section { margin:40px 0 0; }
.section-head { display:flex; align-items:baseline; gap:12px; padding-bottom:10px;
                border-bottom:1px solid #2b2b2b; margin-bottom:18px; }
.section-head h2 { font-size:17px; margin:0; font-weight:600; letter-spacing:.01em; }
.section-head .n { color:#8c8c8c; font-size:13px; }
.section-note { color:#8c8c8c; font-size:13px; margin:-8px 0 16px; max-width:70ch; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(290px,1fr)); gap:18px; }
.card { background:#1b1b1b; border:1px solid #292929; border-radius:12px; overflow:hidden;
        display:flex; flex-direction:column; }
.card img { width:100%; display:block; background:#000; aspect-ratio:3/2; object-fit:contain; }
.card .body { padding:13px 14px 14px; display:flex; flex-direction:column; gap:9px; }
.headline { display:flex; align-items:baseline; justify-content:space-between; gap:10px; }
.name { font-weight:600; font-size:14px; word-break:break-all; }
.score { font-size:26px; font-weight:600; line-height:1; }
.score small { display:block; font-size:11px; font-weight:400; color:#8c8c8c;
               letter-spacing:.03em; text-transform:uppercase; }
.now { color:#8c8c8c; font-size:12px; }
.why { color:#c9c9c9; font-size:13px; }
.todo { margin:0; padding-left:17px; font-size:12.5px; color:#a9c9b9; }
.todo li { margin:2px 0; }
.recipe { font-size:12px; color:#8c8c8c; }
.empty { color:#7c7c7c; font-size:13px; font-style:italic; }
details.expert { margin:44px auto 0; max-width:1180px; border-top:1px solid #2b2b2b;
                 padding-top:16px; }
details.expert summary { cursor:pointer; color:#8c8c8c; font-size:13px; }
details.expert table { width:100%; border-collapse:collapse; margin-top:14px; font-size:12px; }
details.expert th, details.expert td { text-align:left; padding:5px 9px;
                                       border-bottom:1px solid #242424; }
details.expert th { color:#8c8c8c; font-weight:500; }
.foot { max-width:1180px; margin:34px auto 0; color:#767676; font-size:12px; }
.caveat { color:#8a7a5a; }
.scale { color:#8c8c8c; font-size:13px; margin:8px 0 0; }
.missing { color:#c8a06a; font-size:13px; margin:12px 0 0; max-width:72ch; }
.noimg { aspect-ratio:3/2; display:flex; align-items:center; justify-content:center;
         background:#191919; color:#8a8a8a; font-size:12px; text-align:center;
         padding:12px; border-bottom:1px solid #292929; }
.filters { position:sticky; top:0; z-index:5; background:#121212ee;
           backdrop-filter:blur(6px); padding:10px 0 12px; margin:0 auto 4px;
           max-width:1180px; display:flex; flex-wrap:wrap; gap:8px;
           border-bottom:1px solid #242424; }
.chip { background:#1d1d1d; color:#c9c9c9; border:1px solid #2e2e2e; border-radius:99px;
        padding:6px 13px; font:inherit; font-size:13px; cursor:pointer; }
.chip .n { color:#8c8c8c; margin-left:7px; font-size:12px; }
.chip.on { background:#2b2b2b; color:#fff; border-color:#454545; }
.card img { cursor:zoom-in; }
#lightbox { position:fixed; inset:0; background:#000000ee; z-index:20;
            display:flex; align-items:center; justify-content:center; }
#lightbox[hidden] { display:none; }
#lightbox img { max-width:94vw; max-height:94vh; object-fit:contain; }
#lightbox button { position:absolute; top:14px; right:18px; background:none; border:0;
                   color:#ddd; font-size:34px; line-height:1; cursor:pointer; }
@media (max-width:640px) {
  body { padding:18px 12px 48px; }
  .grid { grid-template-columns:1fr; }
  .filters { gap:6px; }
}
a { color:#8fb8d8; }
"""


def write(
    records,
    path: Path,
    *,
    language: str = "en",
    insights_link: str | None = None,
    expert: bool = True,
    assets=None,
    standalone: bool = False,
) -> Path:
    """The default report. One number, five piles, nothing else.

    `assets` maps each record to the image it should show, or to the reason it
    cannot show one. When it is None the page renders placeholders throughout,
    which is the honest result for a caller that did not build derivatives --
    and it is never a silent black tile.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sections = by_section(records)
    counts = {name: len(items) for name, _, items in sections}
    best = max((potential(r) for r in records), default=0)
    lookup = (assets.derivatives if assets else {}) or {}
    missing = (assets.reasons() if assets else {}) or {}

    body = "".join(
        _section(name, accent, items, language, best, lookup)
        for name, accent, items in sections
    )

    link = (
        f'<p class="lede" style="margin-top:10px"><a href="{html.escape(insights_link)}">'
        f'{html.escape(t("report.insights_link", language))}</a></p>'
        if insights_link
        else ""
    )

    document = f"""<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(t("report.title", language))}</title>
<style>{STYLE}</style>
</head>
<body>
<header>
<h1>{html.escape(t("report.title", language))}</h1>
<p class="lede">{html.escape(t("report.lede", language))}</p>
<p class="scale">{html.escape(t("report.scale", language, top=TOP_THRESHOLD, weak=WEAK_THRESHOLD))}</p>
{link}
{_missing_html(missing, language)}
<div class="counts">{_counts_html(counts, language)}</div>
</header>
<nav class="filters">{_filter_html(counts, language)}</nav>
<main>
{body}
</main>
{_expert_html(records, language) if expert else ""}
<p class="foot">{_uplift_note(records, language)}{html.escape(t("report.footer", language))}</p>
<div id="lightbox" hidden><img alt=""><button type="button" aria-label="close">&times;</button></div>
<script>{SCRIPT}</script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def write_folder(
    records,
    report_dir: Path,
    *,
    cache_dir: Path,
    language: str = "en",
    insights_link: str | None = None,
    expert: bool = True,
):
    """A portable `report/` directory: index.html plus its own assets.

    Every `src` is `assets/...`, relative to the page, so the folder can be
    zipped and opened anywhere. Pointing at images elsewhere in the run is what
    broke before: the page was rendered in a staging directory, the paths were
    correct relative to *that*, and publishing moved the file two levels up.
    """
    import report_assets

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    assets = report_assets.build(records, report_dir, cache_dir)
    write(
        records,
        report_dir / "index.html",
        language=language,
        insights_link=insights_link,
        expert=expert,
        assets=assets,
    )
    return assets


def write_standalone(
    records,
    path: Path,
    *,
    language: str = "en",
    expert: bool = True,
    width: int = None,
    quality: int = None,
):
    """One file, every thumbnail inlined. Nothing external, nothing beside it."""
    import report_assets

    assets = report_assets.inline(
        records,
        width=width or report_assets.EMBED_PX,
        quality=quality or report_assets.EMBED_QUALITY,
    )
    write(records, Path(path), language=language, expert=expert, assets=assets)
    return assets


def _missing_html(reasons: dict[str, int], language: str) -> str:
    """Say how many cards have no image, and why. Never leave it to be noticed."""
    if not reasons:
        return ""
    total = sum(reasons.values())
    detail = "; ".join(f"{count} — {reason}" for reason, count in sorted(reasons.items()))
    return (
        f'<p class="missing">{html.escape(t("report.missing_images", language, count=total))} '
        f"({html.escape(detail)})</p>"
    )


def _filter_html(counts: dict[str, int], language: str) -> str:
    buttons = "".join(
        f'<button type="button" data-bucket="{name}" class="chip">'
        f'{html.escape(t(f"category.{name}", language))}'
        f'<span class="n">{counts.get(name, 0)}</span></button>'
        for name, _ in SECTIONS
    )
    return (
        f'<button type="button" data-bucket="ALL" class="chip on">'
        f'{html.escape(t("report.all", language))}'
        f'<span class="n">{sum(counts.values())}</span></button>{buttons}'
    )


def _uplift_note(records, language: str) -> str:
    """One line, when the gain figure has never been checked against a person.

    Shown rather than hidden because the number is prominent on every card and
    reads as a measurement. It is an estimate from a metric that has not been
    compared with a human ranking, and saying so costs one sentence.
    """
    if any(getattr(r, "uplift_validated", False) for r in records):
        return ""
    return f'<span class="caveat">{html.escape(t("report.uplift_unvalidated", language))}</span><br>'


def _counts_html(counts: dict[str, int], language: str) -> str:
    return "".join(
        f'<div class="count"><b>{counts.get(name, 0)}</b>'
        f'<span>{html.escape(t(f"category.{name}", language))}</span></div>'
        for name, _ in SECTIONS
    )


def _section(
    name: str, accent: str, items: list, language: str, best: int = 0,
    lookup: dict | None = None,
) -> str:
    heading = html.escape(t(f"category.{name}", language))
    note = t(f"category.note.{name}", language)
    if not items:
        # An empty pile explains itself. "Nothing here" beside a TOP section
        # leaves the reader unable to tell a strict threshold from a broken one.
        empty = t("report.empty_section", language)
        if name == "TOP" and best:
            empty = t("report.no_top", language, best=best, threshold=TOP_THRESHOLD)
        return (
            f'<section data-bucket="{name}">'
            f'<div class="section-head" style="border-color:{accent}44">'
            f"<h2>{heading}</h2><span class=\"n\">0</span></div>"
            f'<p class="empty">{html.escape(empty)}</p></section>'
        )
    cards = "".join(_card(record, accent, language, lookup or {}) for record in items)
    return (
        f'<section data-bucket="{name}">'
        f'<div class="section-head" style="border-color:{accent}44">'
        f'<h2>{heading}</h2><span class="n">{len(items)}</span></div>'
        f'<p class="section-note">{html.escape(note)}</p>'
        f'<div class="grid">{cards}</div></section>'
    )


def _card(record, accent: str, language: str, lookup: dict) -> str:
    derivative = lookup.get(record.asset_key or record.filename)
    if derivative is not None and derivative.thumb:
        image = (
            f'<img src="{html.escape(derivative.thumb)}" alt="" loading="lazy" '
            f'data-full="{html.escape(derivative.full or derivative.thumb)}">'
        )
    else:
        # A visible statement, not a black rectangle. A missing image that
        # looks like a rendering bug sends somebody debugging the report.
        why = derivative.reason if derivative is not None else t("report.no_preview", language)
        image = f'<div class="noimg">{html.escape(why)}</div>'

    steps = recommendations(record)
    todo = (
        "<ul class=\"todo\">"
        + "".join(f"<li>{html.escape(step)}</li>" for step in steps)
        + "</ul>"
        if steps
        else ""
    )
    why = explanation(record, language)
    recipe = (
        f'<div class="recipe">{html.escape(t("report.recipe_ready", language))}</div>'
        if getattr(record, "recipe_path", "")
        else ""
    )
    # Two secondary numbers, on one line, under the score. `content` is absent
    # rather than zero when no content pass ran.
    secondary = f'{html.escape(t("report.current", language))}: {current_quality(record)}'
    read = content(record)
    if read is not None:
        secondary += f' &middot; {html.escape(t("report.content", language))}: {read}'

    return f"""<div class="card">
{image}
<div class="body">
<div class="headline">
  <span class="name">{html.escape(record.filename)}</span>
  <span class="score" style="color:{accent}">{potential(record)}
    <small>{html.escape(t("report.potential", language))}</small></span>
</div>
<div class="now">{secondary}</div>
{f'<div class="why">{html.escape(why)}</div>' if why else ""}
{todo}
{recipe}
</div></div>"""


def _expert_html(records, language: str) -> str:
    """Everything the default view hides, one click away and closed by default.

    It exists so that hiding is not the same as discarding. A person who wants
    to know why a frame was filed as personal rather than stock can find out;
    nobody is made to read it to use the tool.
    """
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(r.filename)}</td>"
        f"<td>{potential(r)}</td>"
        f"<td>{current_quality(r)}</td>"
        f"<td>{html.escape(r.category)}</td>"
        f"<td>{html.escape(r.route_class)}</td>"
        f"<td>{(r.final_score_detail or {}).get('stage3_delta', 0):+d}</td>"
        f"<td>{html.escape('; '.join(r.commercial_blockers or []))}</td>"
        "</tr>"
        for r in ordered(records)
    )
    headers = "".join(
        f"<th>{html.escape(t(f'expert.{key}', language))}</th>"
        for key in (
            "file", "potential", "current", "category", "route_class",
            "stage3_delta", "stock_blockers",
        )
    )
    return (
        f'<details class="expert"><summary>{html.escape(t("report.expert", language))}</summary>'
        f'<p class="lede">{html.escape(t("report.expert_note", language))}</p>'
        f"<table><tr>{headers}</tr>{rows}</table></details>"
    )


# `_relative` used to live here, turning `record.preview_path` into a path
# relative to wherever the page happened to be written. It is gone with the
# whole idea: the report owns its images now, so there is no path to get wrong.
