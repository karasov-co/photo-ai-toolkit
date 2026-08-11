"""The run's output: one record per asset, three formats, one summary.

The record is deliberately fat. Every number that went into a decision is
stored beside the decision, including the ones that disagreed, because the
question asked of this tool six months from now will not be "what did it say"
but "why did it say that, and was it right". A report that stores only the final
class cannot answer either.

That is also what makes re-classification cheap: `analysis.json` holds enough to
re-run routing against different thresholds without decoding a pixel or paying
for a single token. The expensive half of the pipeline writes here once.

Three formats, for three readers: JSON for the tool and for re-classification,
CSV for a spreadsheet, HTML for a human who wants to look at the pictures. The
HTML one exists because looking at a contact sheet is what caught four good
photographs that an earlier version of the technical filter had marked for
deletion, and no amount of tabular output would have.
"""

from __future__ import annotations

import csv
import html
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from i18n import t

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 2

# --- credential hygiene -----------------------------------------------------

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b\s*[=:]\s*\S+"),
    re.compile(r"https://[^/\s:@]+:[^/\s@]+@"),
)

REDACTED = "[REDACTED]"


def redact(text: str) -> str:
    """Strip anything that looks like a credential.

    Applied to log records, to error strings stored in reports, and to any
    exception text that reaches disk. An API key in a traceback ends up in a
    report the user then attaches to a bug thread, and at that point the key is
    public.
    """
    if not text:
        return text
    out = str(text)
    for pattern in SECRET_PATTERNS:
        out = pattern.sub(REDACTED, out)
    return out


class RedactingFilter(logging.Filter):
    """Attach to every handler so nothing secret is written by accident."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            record.args = tuple(
                redact(a) if isinstance(a, str) else a for a in _as_tuple(record.args)
            )
        return True


def _as_tuple(args):
    return args if isinstance(args, tuple) else (args,)


# --- the record -------------------------------------------------------------


@dataclass
class AssetRecord:
    asset_id: str
    source_path: str
    filename: str
    media_type: str
    checksum: str

    # Unique within a run: the path relative to the scan root. `filename` is a
    # label and is NOT unique -- two memory cards both hold P1000001.RW2, and
    # keying anything by it merges two different photographs.
    asset_key: str = ""
    # The exact files belonging to this asset, captured at analysis time. Any
    # later filesystem operation uses this list verbatim -- never a glob over
    # `stem.*`, which would sweep up unrelated files that happen to share a
    # stem and would miss anything named differently.
    all_files: list[str] = field(default_factory=list)
    # size / mtime / checksum per file, so a move can prove the file is still
    # the one that was analysed.
    file_states: dict = field(default_factory=dict)
    # Machine-readable grounds, for the purge gate. Aesthetic judgements are
    # deliberately not representable here.
    evidence: str = ""

    width: int = 0
    height: int = 0
    megapixels: float = 0.0
    duration: float = 0.0

    schema_version: int = REPORT_SCHEMA_VERSION
    analyzer_version: str = ""
    calibration: str = ""
    model_versions: dict = field(default_factory=dict)
    analyzed_at: str = ""

    scores: dict = field(default_factory=dict)
    route_class: str = ""
    route: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: int = 0

    issues: dict = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    genre: str = ""
    concepts: list[str] = field(default_factory=list)
    description: str = ""
    stock_metadata: dict = field(default_factory=dict)

    edit_recipe: list[str] = field(default_factory=list)
    expected_gain: int = 0

    marketplaces: list[dict] = field(default_factory=list)
    provenance: str = "unknown"
    legal_warnings: list[str] = field(default_factory=list)

    cluster_id: str = ""
    cluster_size: int = 1
    best_in_cluster: bool = True
    cluster_similarity: float = 0.0
    cluster_margin: float = 0.0
    # Stored so that `reclassify` can redo the diversity-aware flagship pass
    # without decoding anything. Without it, re-routing a stored run could
    # never produce a flagship, because the selection needs to compare frames
    # against each other rather than against a threshold.
    phash: str = ""
    semantic_present: bool = False

    video: dict = field(default_factory=dict)
    preview_path: str = ""

    proposed_action: str = ""
    completed_action: str = ""
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["error"] = redact(payload.get("error", ""))
        return payload


CSV_FIELDS = [
    "asset_id", "filename", "source_path", "media_type", "route_class", "route",
    "routing_score", "current_quality", "post_edit_potential", "expected_gain",
    "recoverability", "aesthetic_potential", "stock_potential", "portfolio_potential",
    "legal_readiness", "uniqueness", "confidence",
    "genre", "concepts", "description",
    "unrecoverable", "partially_fixable", "fixable",
    "edit_recipe", "marketplaces", "legal_warnings",
    "cluster_id", "cluster_size", "best_in_cluster",
    "megapixels", "duration", "provenance",
    "proposed_action", "completed_action", "status", "error",
]


def _csv_row(record: AssetRecord) -> dict:
    scores = record.scores or {}
    return {
        "asset_id": record.asset_id,
        "filename": record.filename,
        "source_path": record.source_path,
        "media_type": record.media_type,
        "route_class": record.route_class,
        "route": record.route,
        "routing_score": scores.get("routing_score", ""),
        "current_quality": scores.get("current_quality", ""),
        "post_edit_potential": scores.get("post_edit_potential", ""),
        "expected_gain": record.expected_gain,
        "recoverability": scores.get("recoverability", ""),
        "aesthetic_potential": scores.get("aesthetic_potential", ""),
        "stock_potential": scores.get("stock_potential", ""),
        "portfolio_potential": scores.get("portfolio_potential", ""),
        "legal_readiness": scores.get("legal_readiness", ""),
        "uniqueness": scores.get("uniqueness", ""),
        "confidence": scores.get("confidence", ""),
        "genre": record.genre,
        "concepts": "; ".join(record.concepts),
        "description": record.description,
        "unrecoverable": "; ".join(record.issues.get("unrecoverable", [])),
        "partially_fixable": "; ".join(record.issues.get("partially_fixable", [])),
        "fixable": "; ".join(record.issues.get("fixable", [])),
        "edit_recipe": " | ".join(record.edit_recipe),
        "marketplaces": "; ".join(
            m.get("platform", "") for m in record.marketplaces if m.get("eligible")
        ),
        "legal_warnings": "; ".join(record.legal_warnings),
        "cluster_id": record.cluster_id,
        "cluster_size": record.cluster_size,
        "best_in_cluster": record.best_in_cluster,
        "megapixels": record.megapixels,
        "duration": record.duration,
        "provenance": record.provenance,
        "proposed_action": record.proposed_action,
        "completed_action": record.completed_action,
        "status": record.status,
        "error": redact(record.error),
    }


def write_json(records: list[AssetRecord], path: Path, *, summary: dict | None = None) -> Path:
    """Atomic: a temp file, then a rename.

    Writing in place truncates first, so an interrupt leaves a partial file that
    the next run cannot parse -- which on this codebase once silently discarded
    every result recorded so far.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "disclaimer": t("warn.disclaimer"),
        "summary": summary or {},
        "assets": [r.to_dict() for r in records],
    }
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def read_json(path: Path) -> tuple[list[dict], dict]:
    """Load a previous run so routing can be redone without re-analysing."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("assets") or []), dict(payload.get("summary") or {})


def write_csv(records: list[AssetRecord], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(_csv_row(record))
    return path


# --- collection summary -----------------------------------------------------


def summarise(records: list[AssetRecord], *, recoverable_bytes: int = 0) -> dict:
    ok = [r for r in records if r.status == "ok"]
    by_class: dict[str, int] = {}
    for record in ok:
        by_class[record.route_class] = by_class.get(record.route_class, 0) + 1

    genres: dict[str, int] = {}
    for record in ok:
        if record.genre:
            genres[record.genre] = genres.get(record.genre, 0) + 1

    clusters = {r.cluster_id for r in ok if r.cluster_id and r.cluster_size > 1}
    strongest = sorted(
        ok, key=lambda r: r.scores.get("routing_score", 0), reverse=True
    )[:10]

    return {
        "total": len(records),
        "photos": sum(1 for r in records if r.media_type == "photo"),
        "videos": sum(1 for r in records if r.media_type == "video"),
        "failed": sum(1 for r in records if r.status != "ok"),
        "low_confidence": sum(1 for r in ok if r.confidence < 55),
        "by_class": by_class,
        "duplicate_clusters": len(clusters),
        "recoverable_bytes": recoverable_bytes,
        "recoverable_mb": round(recoverable_bytes / 1_048_576, 1),
        "top_genres": sorted(genres.items(), key=lambda kv: -kv[1])[:8],
        "missing_releases": sum(1 for r in ok if "needs_model_release" in r.tags),
        "marketplace_ready": sum(
            1 for r in ok if any(m.get("export_ready") for m in r.marketplaces)
        ),
        "strongest": [
            {"filename": r.filename, "score": r.scores.get("routing_score", 0), "class": r.route_class}
            for r in strongest
        ],
    }


def format_summary(summary: dict, language: str = "en") -> str:
    """The block printed at the end of a run."""
    lines = ["", "=" * 62, t("summary.title", language), "=" * 62]
    lines.append(f"  {t('summary.total', language):<38}{summary.get('total', 0):>8}")
    lines.append(f"  {t('summary.photos', language):<38}{summary.get('photos', 0):>8}")
    lines.append(f"  {t('summary.videos', language):<38}{summary.get('videos', 0):>8}")
    lines.append("")
    for route_class, count in sorted(
        (summary.get("by_class") or {}).items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"  {t(f'class.{route_class}', language):<38}{count:>8}")
    lines.append("")
    lines.append(f"  {t('summary.clusters', language):<38}{summary.get('duplicate_clusters', 0):>8}")
    lines.append(f"  {t('summary.low_confidence', language):<38}{summary.get('low_confidence', 0):>8}")
    lines.append(f"  {t('summary.missing_releases', language):<38}{summary.get('missing_releases', 0):>8}")
    lines.append(f"  {t('summary.marketplace_ready', language):<38}{summary.get('marketplace_ready', 0):>8}")
    lines.append(f"  {t('summary.failed', language):<38}{summary.get('failed', 0):>8}")
    lines.append(
        f"  {t('summary.recoverable_space', language):<38}{summary.get('recoverable_mb', 0):>7} MB"
    )

    genres = summary.get("top_genres") or []
    if genres:
        lines.append("")
        lines.append(f"  {t('summary.top_genres', language)}: " + ", ".join(f"{g} ({n})" for g, n in genres))

    strongest = summary.get("strongest") or []
    if strongest:
        lines.append("")
        lines.append(f"  {t('summary.strongest', language)}:")
        for item in strongest[:8]:
            lines.append(f"    [{item['score']:>3}] {item['filename']}  ({item['class']})")

    lines.append("")
    lines.append("  " + t("warn.disclaimer", language))
    lines.append("=" * 62)
    return "\n".join(lines)


# --- HTML -------------------------------------------------------------------


CLASS_COLORS = {
    "trash": "#7a2e2e",
    "review": "#7a682e",
    "archive_only": "#4a4a4a",
    "stock_standard": "#2e5a7a",
    "stock_strong": "#2e7a55",
    "flagship": "#6b2e7a",
}


def write_html(
    records: list[AssetRecord],
    path: Path,
    *,
    summary: dict | None = None,
    language: str = "en",
) -> Path:
    """A single self-contained page. Previews are linked, not embedded.

    Embedding 400 base64 JPEGs produces a file no browser opens comfortably;
    linking keeps the page instant and the previews are written next to it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summary or summarise(records)

    cards = []
    for record in sorted(records, key=lambda r: -r.scores.get("routing_score", 0)):
        cards.append(_card(record, path.parent, language))

    document = f"""<!doctype html>
<html lang="{language}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>photo-ai-toolkit report</title>
<style>
:root {{ color-scheme: dark; }}
body {{ background:#141414; color:#e8e8e8; margin:0; padding:24px;
       font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.disclaimer {{ color:#9a9a9a; font-size:12px; max-width:70ch; margin-bottom:20px; }}
.summary {{ display:flex; flex-wrap:wrap; gap:12px; margin-bottom:24px; }}
.stat {{ background:#1e1e1e; border-radius:8px; padding:10px 14px; min-width:120px; }}
.stat b {{ display:block; font-size:20px; }}
.stat span {{ color:#9a9a9a; font-size:12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:16px; }}
.card {{ background:#1c1c1c; border-radius:10px; overflow:hidden; border:1px solid #2a2a2a; }}
.card img {{ width:100%; display:block; background:#000; aspect-ratio:3/2; object-fit:contain; }}
.card .body {{ padding:12px; }}
.name {{ font-weight:600; word-break:break-all; }}
.badge {{ display:inline-block; padding:2px 8px; border-radius:99px; font-size:11px; color:#fff; }}
.scores {{ display:grid; grid-template-columns:1fr auto; gap:2px 10px; font-size:12px; margin:8px 0; }}
.scores .k {{ color:#9a9a9a; }}
.gain {{ color:#6fcf97; font-weight:600; }}
ul {{ margin:6px 0; padding-left:18px; font-size:12px; }}
.bad {{ color:#e08585; }} .mid {{ color:#e0c185; }} .ok {{ color:#85c8e0; }}
details summary {{ cursor:pointer; color:#9a9a9a; font-size:12px; }}
</style>
</head>
<body>
<h1>photo-ai-toolkit &mdash; {len(records)} assets</h1>
<p class="disclaimer">{html.escape(t("warn.disclaimer", language))}</p>
<div class="summary">{_summary_html(summary, language)}</div>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def _summary_html(summary: dict, language: str) -> str:
    stats = [
        (t("summary.total", language), summary.get("total", 0)),
        (t("summary.photos", language), summary.get("photos", 0)),
        (t("summary.videos", language), summary.get("videos", 0)),
        (t("summary.clusters", language), summary.get("duplicate_clusters", 0)),
        (t("summary.low_confidence", language), summary.get("low_confidence", 0)),
        (t("summary.marketplace_ready", language), summary.get("marketplace_ready", 0)),
    ]
    by_class = sorted((summary.get("by_class") or {}).items(), key=lambda kv: -kv[1])
    for route_class, count in by_class:
        stats.append((t(f"class.{route_class}", language), count))
    return "".join(
        f'<div class="stat"><b>{value}</b><span>{html.escape(str(label))}</span></div>'
        for label, value in stats
    )


def _card(record: AssetRecord, base: Path, language: str) -> str:
    color = CLASS_COLORS.get(record.route_class, "#444")
    scores = record.scores or {}
    preview = _relative_preview(record.preview_path, base)

    def rows(keys):
        return "".join(
            f'<div class="k">{html.escape(t(f"score.{k}", language))}</div><div>{scores.get(k, 0)}</div>'
            for k in keys
        )

    gain = record.expected_gain
    gain_html = (
        f'<div class="k">{html.escape(t("recipe.expected_gain", language))}</div>'
        f'<div class="gain">+{gain}</div>'
        if gain > 0
        else ""
    )

    def issue_list(key: str, css: str) -> str:
        items = record.issues.get(key) or []
        if not items:
            return ""
        return (
            f'<div class="{css}">{html.escape(t(f"issues.{key}", language))}:</div><ul>'
            + "".join(f"<li>{html.escape(str(i))}</li>" for i in items)
            + "</ul>"
        )

    recipe = (
        "<details><summary>"
        + html.escape(t("recipe.title", language))
        + "</summary><ul>"
        + "".join(f"<li>{html.escape(str(s))}</li>" for s in record.edit_recipe)
        + "</ul></details>"
        if record.edit_recipe
        else ""
    )

    markets = [m.get("platform", "") for m in record.marketplaces if m.get("eligible")][:4]
    markets_html = (
        f'<div style="font-size:12px;color:#9a9a9a">{html.escape(t("misc.marketplaces", language))}: '
        f"{html.escape(', '.join(markets))}</div>"
        if markets
        else ""
    )

    warnings = (
        '<div class="bad" style="font-size:12px">'
        + "<br>".join(html.escape(str(w)) for w in record.legal_warnings)
        + "</div>"
        if record.legal_warnings
        else ""
    )

    img = f'<img src="{html.escape(preview)}" alt="" loading="lazy">' if preview else ""
    class_label = html.escape(t(f"class.{record.route_class}", language))
    route_label = html.escape(t(f"route.{record.route}", language) if record.route else "")
    score_rows = rows(
        [
            "routing_score", "current_quality", "post_edit_potential",
            "stock_potential", "portfolio_potential", "confidence",
        ]
    )

    return f"""<div class="card">
{img}
<div class="body">
<div class="name">{html.escape(record.filename)}</div>
<span class="badge" style="background:{color}">{class_label}</span>
<span class="badge" style="background:#333">{route_label}</span>
<div class="scores">{score_rows}{gain_html}</div>
{issue_list("unrecoverable", "bad")}{issue_list("partially_fixable", "mid")}{issue_list("fixable", "ok")}
{recipe}
{markets_html}
{warnings}
<details><summary>{html.escape(t("misc.reasons", language))}</summary><ul>
{"".join(f"<li>{html.escape(str(r))}</li>" for r in record.reasons)}
</ul></details>
</div></div>"""


def _relative_preview(preview_path: str, base: Path) -> str:
    if not preview_path:
        return ""
    try:
        return os.path.relpath(Path(preview_path), base)
    except ValueError:
        return preview_path
