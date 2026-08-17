"""What a whole collection says about the photography, rather than the photographs.

Per-file feedback is the easy half and the less useful one. A photographer who
learns that frame 27 is underexposed learns about frame 27; a photographer who
learns that thirty of their forty-seven frames are underexposed by roughly the
same amount has learned something about how they meter, which is a thing they
can change on the next shoot.

So everything below is a count across the run, and nothing is reported unless
enough frames support it. Three rules:

**Only what was measured.** Every observation names the number and the files
behind it. "Your compositions could be stronger" is not an observation; "eleven
of your fourteen portraits put the subject dead centre" is, and the second one
tells you what to try.

**No invented facts.** The inspiration list is a small fixed table of real
photographers, books and projects, chosen per genre and shown as names to look
up. There are no quotations here, no biography and no claims about what those
photographers said or believed, because a plausible invented quotation is worse
than no quotation.

**Encouraging, and specific about it.** The strengths are computed the same way
as the weaknesses and reported first. A tool that only reports faults on a
person's own photographs will be closed and not reopened, and it will also be
wrong: a collection that produced thirteen good stock frames did something
right, and saying so is as factual as saying what it did badly.
"""

from __future__ import annotations

import html
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from photoai.i18n import t

# A pattern needs this many frames behind it before it is a habit rather than a
# coincidence, and this share of the collection before it is worth acting on.
MIN_FRAMES = 3
MIN_SHARE = 0.15

# Stage 3 dimensions, as strengths and as weaknesses.
STRONG_DIMENSION = 62
WEAK_DIMENSION = 42


@dataclass
class Observation:
    """One finding, the number behind it, and the files that produced it."""

    text: str
    evidence: str = ""
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "evidence": self.evidence, "examples": self.examples}


@dataclass
class Insights:
    total: int = 0
    genres: list[tuple[str, int]] = field(default_factory=list)
    visual_habits: list[Observation] = field(default_factory=list)
    technical_strengths: list[Observation] = field(default_factory=list)
    artistic_strengths: list[Observation] = field(default_factory=list)
    weaknesses: list[Observation] = field(default_factory=list)
    improvements: list[Observation] = field(default_factory=list)
    inspiration: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "genres": self.genres,
            "visual_habits": [o.to_dict() for o in self.visual_habits],
            "technical_strengths": [o.to_dict() for o in self.technical_strengths],
            "artistic_strengths": [o.to_dict() for o in self.artistic_strengths],
            "weaknesses": [o.to_dict() for o in self.weaknesses],
            "improvements": [o.to_dict() for o in self.improvements],
            "inspiration": self.inspiration,
        }


# --- the reference table ------------------------------------------------------
#
# Names to look up, per genre. Deliberately short, deliberately without
# commentary: a one-line summary of a photographer's work is usually wrong, and
# an invented one is indefensible. Nothing here is a quotation.

INSPIRATION: dict[str, list[dict]] = {
    "portrait": [
        {"name": "Rineke Dijkstra", "kind": "photographer", "note": "Beach Portraits"},
        {"name": "Judith Joy Ross", "kind": "photographer", "note": "Portraits of people at ease"},
        {"name": "Paul Strand", "kind": "book", "note": "Tir a'Mhurain"},
    ],
    "landscape": [
        {"name": "Fay Godwin", "kind": "photographer", "note": "Land"},
        {"name": "Robert Adams", "kind": "book", "note": "The New West"},
        {"name": "Rinko Kawauchi", "kind": "photographer", "note": "Illuminance"},
    ],
    "street": [
        {"name": "Saul Leiter", "kind": "photographer", "note": "Early Color"},
        {"name": "Daido Moriyama", "kind": "photographer", "note": "Japan: A Photo Theater"},
        {"name": "Joel Meyerowitz", "kind": "book", "note": "Cape Light"},
    ],
    "reportage": [
        {"name": "Alex Webb", "kind": "photographer", "note": "The Suffering of Light"},
        {"name": "Gregory Halpern", "kind": "book", "note": "ZZYZX"},
        {"name": "World Press Photo", "kind": "project", "note": "Annual contest archive"},
    ],
    "architecture": [
        {"name": "Bernd and Hilla Becher", "kind": "photographer", "note": "Typologies"},
        {"name": "Lucien Hervé", "kind": "photographer", "note": "Architectural details"},
        {"name": "Iwan Baan", "kind": "photographer", "note": "Buildings in use"},
    ],
    "detail": [
        {"name": "Wolfgang Tillmans", "kind": "photographer", "note": "Still lifes"},
        {"name": "Rinko Kawauchi", "kind": "book", "note": "Utatane"},
        {"name": "Sophie Ristelhueber", "kind": "photographer", "note": "Fait"},
    ],
    "night": [
        {"name": "Todd Hido", "kind": "photographer", "note": "House Hunting"},
        {"name": "Rut Blees Luxemburg", "kind": "photographer", "note": "London night"},
        {"name": "Brassaï", "kind": "book", "note": "Paris de Nuit"},
    ],
}


# --- building the insights ----------------------------------------------------


def build(records, measurements: dict | None = None) -> Insights:
    """Everything the collection supports, and nothing it does not."""
    ok = [r for r in records if r.status == "ok"]
    insights = Insights(total=len(ok))
    if not ok:
        return insights

    measurements = measurements or {}
    insights.genres = _genres(ok)
    insights.visual_habits = _habits(ok, measurements)
    insights.technical_strengths = _technical_strengths(ok, measurements)
    insights.artistic_strengths = _artistic_strengths(ok)
    insights.weaknesses = _weaknesses(ok, measurements)
    insights.improvements = _improvements(ok, insights)
    insights.inspiration = _inspiration(insights.genres)
    return insights


def _names(records, limit: int = 3) -> list[str]:
    return [r.filename for r in sorted(records, key=lambda r: -int(r.final_score or 0))[:limit]]


def _genres(records) -> list[tuple[str, int]]:
    """Ranked by how well the genre did, not only by how often it appears.

    Shooting forty landscapes says what you point the camera at. Landscapes
    scoring eight points above your average says what you are good at, and only
    the second one is worth telling somebody.
    """
    counts = Counter(r.genre for r in records if r.genre and r.genre != "unknown")
    if not counts:
        return []
    overall = statistics.mean(int(r.final_score or 0) for r in records)
    ranked = []
    for genre, count in counts.items():
        if count < MIN_FRAMES:
            continue
        scores = [int(r.final_score or 0) for r in records if r.genre == genre]
        ranked.append((genre, count, statistics.mean(scores) - overall))
    ranked.sort(key=lambda row: (-row[2], -row[1]))
    return [(genre, count) for genre, count, _ in ranked]


def _habits(records, measurements) -> list[Observation]:
    """Recurring choices, framed as choices rather than as faults."""
    out: list[Observation] = []
    total = len(records)

    portraits = [r for r in records if (r.stage3 or {}).get("portrait")]
    close = [
        r for r in portraits
        if float((r.stage3 or {}).get("portrait", {}).get("primary_face_area_ratio") or 0) >= 0.10
    ]
    far = [
        r for r in portraits
        if 0 < float((r.stage3 or {}).get("portrait", {}).get("primary_face_area_ratio") or 0) < 0.04
    ]
    if len(portraits) >= MIN_FRAMES and len(far) >= len(close) and far:
        out.append(
            Observation(
                "You photograph people inside their surroundings rather than close in on "
                "the face -- the setting is doing as much work as the subject.",
                f"{len(far)} of {len(portraits)} portraits put the face under 4% of the frame",
                _names(far),
            )
        )
    elif close and len(close) > len(far):
        out.append(
            Observation(
                "You work close. The face fills the frame and the surroundings drop away.",
                f"{len(close)} of {len(portraits)} portraits fill 10% or more of the frame with the face",
                _names(close),
            )
        )

    dark = [
        r for r in records
        if getattr(measurements.get(r.asset_key), "mean_luma", 128) and
        float(getattr(measurements.get(r.asset_key), "mean_luma", 128)) < 100
    ]
    if _is_a_pattern(dark, total):
        out.append(
            Observation(
                "You expose for the highlights and let the rest go dark. That is a choice a "
                "RAW file supports well, and it is why several of these open up in the edit.",
                f"{len(dark)} of {total} frames sit below a mean luminance of 100 of 255",
                _names(dark),
            )
        )

    tall = [r for r in records if r.height > r.width]
    if _is_a_pattern(tall, total):
        out.append(
            Observation(
                "You turn the camera. A vertical frame is a decision, and you are making it often.",
                f"{len(tall)} of {total} frames are vertical",
                _names(tall),
            )
        )
    return out[:4]


def _technical_strengths(records, measurements) -> list[Observation]:
    out: list[Observation] = []
    total = len(records)

    sharp = [r for r in records if not (r.issues or {}).get("unrecoverable")]
    if len(sharp) / total >= 0.85:
        out.append(
            Observation(
                "Your focus and your shutter discipline are reliable. Almost nothing here "
                "was lost to a technical failure.",
                f"{len(sharp)} of {total} frames carry no unrecoverable fault",
                _names(sharp),
            )
        )

    recoverable = [r for r in records if (r.scores or {}).get("recoverability", 0) >= 80]
    if _is_a_pattern(recoverable, total):
        out.append(
            Observation(
                "You leave yourself room to work. These files hold the latitude an edit needs.",
                f"{len(recoverable)} of {total} frames score 80+ on recoverability",
                _names(recoverable),
            )
        )

    gains = [int(r.expected_gain or 0) for r in records]
    if gains and statistics.median(gains) >= 6:
        best = sorted(records, key=lambda r: -int(r.expected_gain or 0))[:3]
        out.append(
            Observation(
                "Your frames respond well to processing -- the gap between the file as shot "
                "and the file edited is real and consistent.",
                f"a median of {statistics.median(gains):.0f} points gained by a normal edit",
                [r.filename for r in best],
            )
        )
    return out[:3]


def _artistic_strengths(records) -> list[Observation]:
    """Where Stage 3 keeps saying the same good thing."""
    out: list[Observation] = []
    read = [r for r in records if (r.stage3 or {}).get("status") == "completed"]
    if not read:
        return out

    labels = {
        "emotional_resonance": "Your photographs produce a felt response rather than merely recording what was there.",
        "moment_specificity": "You catch specific moments -- gestures and coincidences that would not come again.",
        "formal_coherence": "Your frames hold together as whole pictures.",
        "distinctiveness": "Your work does not look like everyone else's photographs of the same places.",
        "documentary_significance": "You are preserving a place and a time, not just photographing it.",
        "visual_tension": "You leave things unresolved in the frame, which is what keeps a viewer in it.",
        "narrative_openness": "Your pictures raise questions instead of listing what was present.",
    }
    for dimension, sentence in labels.items():
        strong = [r for r in read if int((r.stage3 or {}).get(dimension) or 0) >= STRONG_DIMENSION]
        if _is_a_pattern(strong, len(read)):
            average = statistics.mean(
                int((r.stage3 or {}).get(dimension) or 0) for r in read
            )
            out.append(
                Observation(
                    sentence,
                    f"{len(strong)} of {len(read)} frames score {STRONG_DIMENSION}+ "
                    f"(collection average {average:.0f})",
                    _names(strong),
                )
            )
    out.sort(key=lambda o: -len(o.examples))
    return out[:3]


def _weaknesses(records, measurements) -> list[Observation]:
    out: list[Observation] = []
    total = len(records)

    duplicates = [r for r in records if not r.best_in_cluster]
    if _is_a_pattern(duplicates, total):
        out.append(
            Observation(
                "You shoot several near-identical frames of the same thing and keep them all. "
                "The extra takes are not adding a better version -- they are adding work later.",
                f"{len(duplicates)} of {total} frames lost to a sharper sibling",
                _names(duplicates),
            )
        )

    blinks = [
        r for r in records
        if (r.portrait_verdict or "keep") == "reject"
    ]
    if blinks:
        out.append(
            Observation(
                "Expressions are being lost -- blinks and mid-word moments. This is a shooting "
                "rhythm problem, not an editing one.",
                f"{len(blinks)} frame(s) where the face failed",
                _names(blinks),
            )
        )

    read = [r for r in records if (r.stage3 or {}).get("status") == "completed"]
    for dimension, sentence in (
        ("distinctiveness", "Many of these frames could have been taken by anyone standing "
                            "where you stood. The subjects are good; the angle on them is the usual one."),
        ("moment_specificity", "The moments are general rather than particular -- there is often "
                               "nothing happening that could only have happened then."),
        ("visual_tension", "The frames resolve too easily. Nothing in most of them is left "
                           "in conflict or unexplained."),
    ):
        low = [r for r in read if int((r.stage3 or {}).get(dimension) or 0) <= WEAK_DIMENSION]
        if read and len(low) / len(read) >= 0.4:
            average = statistics.mean(int((r.stage3 or {}).get(dimension) or 0) for r in read)
            out.append(
                Observation(
                    sentence,
                    f"{len(low)} of {len(read)} frames score {WEAK_DIMENSION} or below "
                    f"(average {average:.0f})",
                    _names(low),
                )
            )

    tilted = [
        r for r in records
        if any("Straighten" in str(step) for step in (r.edit_recipe or []))
    ]
    if _is_a_pattern(tilted, total):
        out.append(
            Observation(
                "Horizons are coming in tilted often enough that it is worth fixing in the "
                "camera rather than in the crop -- straightening costs resolution every time.",
                f"{len(tilted)} of {total} frames need straightening",
                _names(tilted),
            )
        )
    return out[:4]


def _improvements(records, insights: Insights) -> list[Observation]:
    """The three most useful things to change, each tied to a finding above.

    Ordered by how much of the collection each one would have affected, which
    is a defensible ordering and not a preference: the thing that cost the most
    frames is the thing worth changing first.
    """
    candidates: list[tuple[int, Observation]] = []
    total = len(records)

    duplicates = [r for r in records if not r.best_in_cluster]
    if duplicates:
        candidates.append((
            len(duplicates),
            Observation(
                "Shoot the frame, then change something before shooting again -- move, wait, "
                "or change the angle. A second identical take cannot be better than the first.",
                f"{len(duplicates)} of {total} frames were near-duplicates of a better one",
                _names(duplicates),
            ),
        ))

    rejected = [r for r in records if (r.portrait_verdict or "keep") == "reject"]
    if rejected:
        candidates.append((
            len(rejected) * 3,
            Observation(
                "With people, keep shooting through the moment rather than stopping at the "
                "frame you meant to take. The usable expression is usually one or two frames "
                "after the one you planned.",
                f"{len(rejected)} portrait(s) lost to a blink or a mid-word expression",
                _names(rejected),
            ),
        ))

    read = [r for r in records if (r.stage3 or {}).get("status") == "completed"]
    if read:
        distinct = statistics.mean(int((r.stage3 or {}).get("distinctiveness") or 0) for r in read)
        if distinct <= 55:
            candidates.append((
                len(read),
                Observation(
                    "Before you press the shutter on the obvious view, take two steps in any "
                    "direction and look again. The frames here that stand out are the ones "
                    "shot from somewhere other than where everyone stands.",
                    f"distinctiveness averages {distinct:.0f} of 100 across the collection",
                    _names(sorted(read, key=lambda r: -int((r.stage3 or {}).get("distinctiveness") or 0))[:3]),
                ),
            ))

        moment = statistics.mean(int((r.stage3 or {}).get("moment_specificity") or 0) for r in read)
        if moment <= 55:
            candidates.append((
                int(len(read) * 0.9),
                Observation(
                    "Wait longer in the good light. Most of these frames record a place "
                    "correctly; the strongest ones caught something happening in it.",
                    f"moment specificity averages {moment:.0f} of 100",
                    _names(sorted(read, key=lambda r: -int((r.stage3 or {}).get("moment_specificity") or 0))[:3]),
                ),
            ))

    weak_exposure = [
        r for r in records if int((r.scores or {}).get("current_quality", 100)) < 45
        and int(r.final_score or 0) >= 60
    ]
    if _is_a_pattern(weak_exposure, total):
        candidates.append((
            len(weak_exposure),
            Observation(
                "Several strong frames were underexposed far enough that the edit is doing "
                "real work to rescue them. Half a stop more at the time would cost nothing.",
                f"{len(weak_exposure)} frames scored under 45 as shot but 60+ after editing",
                _names(weak_exposure),
            ),
        ))

    candidates.sort(key=lambda row: -row[0])
    return [observation for _, observation in candidates[:3]]


def _inspiration(genres: list[tuple[str, int]]) -> list[dict]:
    out: list[dict] = []
    for genre, _count in genres[:2]:
        entries = INSPIRATION.get(genre)
        if entries:
            out.append({"genre": genre, "entries": entries[:3]})
    return out


def _is_a_pattern(subset, total: int) -> bool:
    """Enough frames, and enough of the collection, to be worth mentioning."""
    return len(subset) >= MIN_FRAMES and total and len(subset) / total >= MIN_SHARE


# --- the page -----------------------------------------------------------------


STYLE = """
:root { color-scheme: dark; }
body { background:#121212; color:#ececec; margin:0; padding:32px 24px 64px;
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:820px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 6px; font-weight:600; }
.lede { color:#9d9d9d; font-size:14px; margin:0 0 30px; }
h2 { font-size:16px; margin:34px 0 12px; font-weight:600;
     border-bottom:1px solid #2b2b2b; padding-bottom:8px; }
.item { margin:0 0 18px; }
.item p { margin:0 0 4px; }
.evidence { color:#8c8c8c; font-size:12.5px; }
.examples { color:#7fa8c8; font-size:12.5px; }
.genres { color:#c9c9c9; }
ol { padding-left:20px; }
ol li { margin:0 0 14px; }
.insp { color:#c9c9c9; font-size:14px; }
.insp b { color:#ececec; }
.foot { color:#767676; font-size:12px; margin-top:36px; }
a { color:#8fb8d8; }
"""


def write(
    insights: Insights,
    path: Path,
    *,
    language: str = "en",
    report_link: str | None = None,
    scope: str = "all",
    total_stored: int = 0,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    link = (
        f'<p class="lede"><a href="{html.escape(report_link)}">'
        f'{html.escape(t("insights.back", language))}</a></p>'
        if report_link
        else ""
    )
    sections = [
        _list_section(t("insights.genres", language), _genre_items(insights, language)),
        _list_section(t("insights.habits", language), insights.visual_habits),
        _list_section(t("insights.technical", language), insights.technical_strengths),
        _list_section(t("insights.artistic", language), insights.artistic_strengths),
        _list_section(t("insights.weaknesses", language), insights.weaknesses),
        _numbered_section(t("insights.improvements", language), insights.improvements),
        _inspiration_section(insights, language),
    ]

    document = f"""<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(t("insights.title", language))}</title>
<style>{STYLE}</style>
</head>
<body>
<main>
<h1>{html.escape(t("insights.title", language))}</h1>
<p class="lede">{html.escape(_scope_line(insights, language, scope, total_stored))}</p>
{link}
{"".join(s for s in sections if s)}
<p class="foot">{html.escape(t("insights.footer", language))}</p>
</main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def _scope_line(insights: Insights, language: str, scope: str, total_stored: int) -> str:
    """Say which photographs this page is about, always.

    A page that silently describes a different set of photographs from one run
    to the next is worse than one that describes the wrong set: the reader
    cannot tell which they are looking at. So the scope is stated in the first
    sentence, with the numbers.
    """
    if scope == "new" and total_stored and total_stored != insights.total:
        return t(
            "insights.lede_new", language, total=insights.total, stored=total_stored
        )
    return t("insights.lede", language, total=insights.total)


def _genre_items(insights: Insights, language: str) -> list[Observation]:
    if not insights.genres:
        return []
    named = ", ".join(f"{genre} ({count})" for genre, count in insights.genres[:5])
    return [
        Observation(
            t("insights.genres_lead", language, genres=named),
            t("insights.genres_evidence", language, genre=insights.genres[0][0]),
        )
    ]


def _list_section(title: str, items: list[Observation]) -> str:
    if not items:
        return ""
    body = "".join(
        f'<div class="item"><p>{html.escape(o.text)}</p>'
        + (f'<p class="evidence">{html.escape(o.evidence)}</p>' if o.evidence else "")
        + (
            f'<p class="examples">{html.escape(", ".join(o.examples))}</p>'
            if o.examples
            else ""
        )
        + "</div>"
        for o in items
    )
    return f"<h2>{html.escape(title)}</h2>{body}"


def _numbered_section(title: str, items: list[Observation]) -> str:
    if not items:
        return ""
    body = "".join(
        f"<li><p>{html.escape(o.text)}</p>"
        + (f'<p class="evidence">{html.escape(o.evidence)}</p>' if o.evidence else "")
        + (
            f'<p class="examples">{html.escape(", ".join(o.examples))}</p>'
            if o.examples
            else ""
        )
        + "</li>"
        for o in items
    )
    return f"<h2>{html.escape(title)}</h2><ol>{body}</ol>"


def _inspiration_section(insights: Insights, language: str) -> str:
    if not insights.inspiration:
        return ""
    blocks = []
    for group in insights.inspiration:
        entries = "".join(
            f'<li class="insp"><b>{html.escape(e["name"])}</b> — {html.escape(e["note"])}</li>'
            for e in group["entries"]
        )
        blocks.append(
            f'<p class="genres">{html.escape(group["genre"])}</p><ul>{entries}</ul>'
        )
    return (
        f'<h2>{html.escape(t("insights.inspiration", language))}</h2>'
        f'<p class="evidence">{html.escape(t("insights.inspiration_note", language))}</p>'
        + "".join(blocks)
    )
