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
    # xAI keys. Added after a provider quoted one back inside a 401 body that
    # went straight into processing.log -- the file people attach to bug
    # reports. `sk-` was covered; `xai-` was not.
    re.compile(r"xai-[A-Za-z0-9_\-]{16,}"),
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
    # The headline answer: one of TOP, GOOD_STOCK, GOOD_PERSONAL, WEAK,
    # NEEDS_DECISION. `route_class` still drives the filesystem layout and the
    # marketplace logic; this is what the photograph *is*.
    category: str = ""
    final_score: int = 0
    # How the final score was arrived at, including what Stage 3 moved it by and
    # every ceiling or floor that was applied. A capped score and a genuinely
    # low one are indistinguishable without this.
    final_score_detail: dict = field(default_factory=dict)
    category_reasons: list[str] = field(default_factory=list)
    # Where this photograph's XMP sidecar was written, if it earned one.
    recipe_path: str = ""
    # Optional readings of the frame, each with the measurement that earned it.
    # Never a style library: a season applied to an unrelated photograph is a
    # lie about the picture.
    creative_directions: list[dict] = field(default_factory=list)
    # Why it cannot be sold, kept strictly apart from why it is or is not good.
    commercial_blockers: list[str] = field(default_factory=list)
    route_class: str = ""
    route: str = ""
    tags: list[str] = field(default_factory=list)
    confidence: int = 0

    issues: dict = field(default_factory=dict)
    strengths: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # Stable keys plus params, so the console and the HTML can render the same
    # reason in the user's language while the JSON keeps the English text.
    reason_keys: list[dict] = field(default_factory=list)

    genre: str = ""
    # Normalised "Make Model". The personal model abstains on an unfamiliar
    # camera, and it can only do that if the camera actually reaches it.
    camera: str = ""
    concepts: list[str] = field(default_factory=list)
    description: str = ""
    stock_metadata: dict = field(default_factory=dict)

    edit_recipe: list[str] = field(default_factory=list)
    expected_gain: int = 0
    # Whether the uplift figure has been checked against a human ranking.
    # False everywhere until `bench-quality` says otherwise, because
    # `frame_quality` is both the objective the preview search optimises and
    # the ruler that reports the result -- so an unvalidated uplift measures
    # how far the search moved its own metric, not how much better the
    # photograph got.
    uplift_validated: bool = False

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
    analysis_mode: str = "local_only"
    semantic_requested: bool = False
    semantic_completed: bool = False
    semantic_model: str = ""
    semantic_error: str = ""

    video: dict = field(default_factory=dict)
    preview_path: str = ""

    # --- darkroom assistant ---
    edit_recipes: list[dict] = field(default_factory=list)
    rendered_variants: dict = field(default_factory=dict)
    recipe_confidence: dict = field(default_factory=dict)
    preserve_intent: list[str] = field(default_factory=list)
    # The files that were actually written, and why the other candidates were
    # refused. A rejection is the more useful half: it says what the tool
    # thought about doing and decided would have damaged the frame.
    suggested_sidecars: dict = field(default_factory=dict)
    darkroom_engine: str = ""
    darkroom_engine_version: str = ""
    darkroom_rejections: list[str] = field(default_factory=list)

    # --- artistic read and the learning loop ---
    artistic: dict = field(default_factory=dict)
    # The Stage 3 record: validated scores, or an explicit status saying why
    # there are none. Never a bare null pretending to be a low score.
    stage3: dict = field(default_factory=dict)
    portrait_verdict: str = "keep"
    personal_preference_probability: float | None = None
    curatorial_disagreement: bool = False
    out_of_distribution: bool = False
    abstained: bool = False
    decision_bucket: str = ""
    policy_evidence: list[str] = field(default_factory=list)

    proposed_action: str = ""
    completed_action: str = ""
    status: str = "ok"
    error: str = ""

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["error"] = redact(payload.get("error", ""))
        return payload


CSV_FIELDS = [
    "asset_id", "filename", "category", "final_score", "stage3_delta",
    "source_path", "media_type", "route_class", "route",
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
        "category": record.category,
        "final_score": record.final_score,
        "stage3_delta": (record.final_score_detail or {}).get("stage3_delta", ""),
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


def write_json_atomic(payload: dict, path: Path) -> Path:
    """Any JSON document, written the same safe way as the report is.

    A temp file and a rename, because an interrupted write leaves a file that
    parses as nothing and reads as "the run produced no insights".
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


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
    """Counts, with the two stock figures named so they cannot be confused.

    "Usable stock: 37" beside "Marketplace-ready: 0" was technically correct and
    read as a contradiction. They answer different questions: one is a technical
    screen, the other requires content, faces, logos, releases and metadata to
    have been checked. The names now say which is which.
    """
    ok = [r for r in records if r.status == "ok"]
    by_class: dict[str, int] = {}
    for record in ok:
        by_class[record.route_class] = by_class.get(record.route_class, 0) + 1

    genres: dict[str, int] = {}
    for record in ok:
        if record.genre:
            genres[record.genre] = genres.get(record.genre, 0) + 1

    clusters = {r.cluster_id for r in ok if r.cluster_id and r.cluster_size > 1}
    # Ranked by the photographic score, not by `routing_score`. The old ordering
    # put the most *saleable* frames at the top of a list headed "strongest",
    # which is a different question and frequently a different answer.
    strongest = sorted(ok, key=lambda r: r.final_score, reverse=True)[:10]

    import curation

    by_category = curation.counts([r.category for r in ok])
    top_photos = [r for r in ok if r.category == curation.PhotoCategory.TOP.value]
    stage3_moved = [
        r for r in ok if (r.final_score_detail or {}).get("stage3_delta")
    ]

    best = max((r.final_score for r in ok), default=0)
    return {
        "by_category": by_category,
        # So that an empty TOP pile explains itself. "0 top photos" alone leaves
        # the reader unable to tell a strict threshold from a broken one.
        "best_final_score": best,
        "top_threshold": curation.DEFAULT_THRESHOLDS.top,
        "top_photos": [
            {"filename": r.filename, "score": r.final_score, "genre": r.genre}
            for r in sorted(top_photos, key=lambda r: r.final_score, reverse=True)
        ],
        "stage3_completed": sum(
            1 for r in ok if (r.stage3 or {}).get("status") == "completed"
        ),
        "stage3_influenced": len(stage3_moved),
        "needs_decision_fraction": (
            round(by_category.get("NEEDS_DECISION", 0) / len(ok), 4) if ok else 0.0
        ),
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
        # Technically past the thresholds. Says nothing about content.
        "technically_usable": sum(
            1 for r in ok if r.route_class in ("stock_standard", "stock_strong", "flagship")
        ),
        # Everything checked and exportable. Requires the semantic pass.
        "fully_checked": sum(
            1 for r in ok if any(m.get("export_ready") for m in r.marketplaces)
        ),
        "marketplace_ready": sum(
            1 for r in ok if any(m.get("export_ready") for m in r.marketplaces)
        ),
        "not_semantically_checked": sum(1 for r in ok if not r.semantic_present),
        "semantic_ran": any(r.semantic_present for r in ok),
        # Partial coverage has to be visible. "local + semantic" was printed on
        # a run where a rejected group left eight of forty-seven photographs
        # with no content check at all, and nothing in the summary said so.
        "semantic_partial": any(r.semantic_present for r in ok)
        and not all(r.semantic_present for r in ok),
        "analysis_mode": next((r.analysis_mode for r in records if r.analysis_mode), "local_only"),
        "strongest": [
            {
                "filename": r.filename,
                "score": r.final_score,
                "class": r.category or r.route_class,
                "stage3_delta": (r.final_score_detail or {}).get("stage3_delta", 0),
            }
            for r in strongest
        ],
    }


# Column width for the summary. Russian labels are longer than English ones and
# were overflowing the old 38, which pushed the numbers out of alignment.
LABEL_WIDTH = 44


def _row(label: str, value, width: int = LABEL_WIDTH) -> str:
    """One aligned line. Padding is measured on the label, not guessed.

    A label longer than the column gets a single space rather than being butted
    straight against its number -- which is where `Готово к загрузке на сток0`
    came from.
    """
    text = str(label)
    padding = max(1, width - len(text))
    return f"  {text}{' ' * padding}{value:>7}"


def format_summary(summary: dict, language: str = "en", *, expert: bool = False) -> str:
    """The block printed at the end of a run.

    The default is the five piles and nothing else. What used to be here as well
    -- route classes, missing releases, technically-usable counts, marketplace
    readiness -- answered questions a photographer never asked, in vocabulary
    they should not have to learn. All of it is still computed, still in the
    JSON, and printed by `--expert`.
    """
    lines = ["", "=" * 66, t("summary.title", language), "=" * 66]

    mode = summary.get("analysis_mode", "local_only")
    lines.append(_row(t("mode.title", language), t(f"mode.{mode}", language)))
    if mode != "local_and_semantic":
        lines.append("")
        lines.append(f"  *** {t('mode.banner', language)} ***")
        lines.append(f"  {t('mode.banner_detail', language)}")
    if summary.get("semantic_partial"):
        lines.append(
            "  "
            + t(
                "summary.semantic_partial",
                language,
                count=summary.get("not_semantically_checked", 0),
                total=summary.get("total", 0),
            )
        )
    lines.append("")

    lines.append(_row(t("summary.total", language), summary.get("total", 0)))
    if summary.get("videos"):
        lines.append(_row(t("summary.photos", language), summary.get("photos", 0)))
        lines.append(_row(t("summary.videos", language), summary.get("videos", 0)))

    # Every pile is printed even at zero -- an absent line reads as "none", and
    # so does a zero, but only the zero is something the reader can rely on.
    by_category = summary.get("by_category") or {}
    if by_category:
        lines.append("")
        lines.append(f"  {t('summary.categories', language)}")
        for category, count in by_category.items():
            lines.append(_row("  " + t(f"category.{category}", language), count))
        if not by_category.get("TOP"):
            lines.append(
                "  "
                + t(
                    "summary.top_gap",
                    language,
                    best=summary.get("best_final_score", 0),
                    threshold=summary.get("top_threshold", 85),
                )
            )

    lines.append("")
    lines.append(_row(t("summary.clusters", language), summary.get("duplicate_clusters", 0)))
    if summary.get("failed"):
        lines.append(_row(t("summary.failed", language), summary.get("failed", 0)))

    top = summary.get("top_photos") or []
    strongest = summary.get("strongest") or []
    shown = top or strongest
    if shown:
        lines.append("")
        label = "summary.top_photos" if top else "summary.strongest"
        lines.append(f"  {t(label, language)}:")
        for item in shown[:8]:
            lines.append(f"    [{item['score']:>3}] {item['filename']}")

    if expert:
        lines.extend(_expert_summary(summary, language))

    lines.append("")
    lines.extend(_wrap(t("warn.disclaimer" if expert else "warn.disclaimer_simple", language)))
    lines.append("=" * 66)
    return "\n".join(lines)


def _expert_summary(summary: dict, language: str) -> list[str]:
    """The old block, behind a flag. Nothing was deleted, only moved."""
    lines = ["", f"  {t('summary.route_classes', language)}"]
    for route_class, count in sorted(
        (summary.get("by_class") or {}).items(), key=lambda kv: -kv[1]
    ):
        lines.append(_row("  " + t(f"class.{route_class}", language), count))
    lines.append("")
    lines.append(_row(t("summary.low_confidence", language), summary.get("low_confidence", 0)))

    if summary.get("semantic_ran"):
        lines.append(
            _row(t("summary.missing_releases", language), summary.get("missing_releases", 0))
        )
    else:
        lines.append(
            _row(t("summary.release_status", language), t("summary.release_unchecked", language))
        )
    lines.append(
        _row(t("summary.technically_usable", language), summary.get("technically_usable", 0))
    )
    lines.append(_row(t("summary.fully_checked", language), summary.get("fully_checked", 0)))
    lines.append(
        _row(t("summary.recoverable_space", language), f"{summary.get('recoverable_mb', 0)} MB")
    )

    genres = summary.get("top_genres") or []
    if genres:
        lines.append("")
        lines.append(
            f"  {t('summary.top_genres', language)}: "
            + ", ".join(f"{g} ({n})" for g, n in genres)
        )
    return lines


def _wrap(text: str, width: int = 64) -> list[str]:
    """Wrap on whitespace, so no join can ever fuse two words together."""
    import textwrap

    return ["  " + line for line in textwrap.wrap(" ".join(text.split()), width=width)]


# --- HTML -------------------------------------------------------------------


# The five categories, in the order the eye should meet them.
CATEGORY_COLORS = {
    "TOP": "#6b2e7a",
    "GOOD_STOCK": "#2e7a55",
    "GOOD_PERSONAL": "#2e5a7a",
    "NEEDS_DECISION": "#7a682e",
    "WEAK": "#5a5a5a",
}

CLASS_COLORS = {
    "trash": "#7a2e2e",
    "review": "#7a682e",
    "archive_only": "#4a4a4a",
    "duplicate_candidate": "#5a4a7a",
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

    # Ranked by the photographic score. The page used to open with whatever was
    # most saleable, which on a personal archive is not what anybody came to see.
    ordered = sorted(records, key=lambda r: -r.final_score)
    cards = [_card(record, path.parent, language) for record in ordered]
    top_section = _top_photos_html(ordered, path.parent, language)

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
.ab {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:6px; margin:8px 0; }}
.ab figure {{ margin:0; }}
.ab img {{ width:100%; aspect-ratio:3/2; object-fit:cover; border-radius:4px; background:#000; }}
.ab figcaption {{ font-size:11px; color:#9a9a9a; text-align:center; padding-top:2px; }}
.keep {{ border-left:3px solid #6fcf97; padding-left:8px; margin:6px 0; font-size:12px; }}
.veto {{ border-left:3px solid #e08585; padding-left:8px; margin:6px 0; font-size:12px; color:#e0a5a5; }}
.bucket {{ display:inline-block; padding:2px 8px; border-radius:99px; font-size:11px;
          background:#26323a; color:#9fd0e0; }}
.banner {{ background:#4a2020; border:1px solid #7a3030; border-radius:8px; padding:12px 16px;
          margin-bottom:16px; }}
.banner b {{ display:block; font-size:15px; letter-spacing:0.04em; }}
.banner span {{ color:#e0b5b5; font-size:13px; }}
.stat[title] {{ cursor:help; border-bottom:1px dotted #555; }}
.top {{ background:#1b1420; border:1px solid #46284f; border-radius:10px;
       padding:16px; margin-bottom:24px; }}
.top h2 {{ font-size:16px; margin:0 0 12px; color:#d9b8e6; }}
.top .grid {{ grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); }}
.top .card {{ border-color:#46284f; }}
.delta-up {{ color:#6fcf97; }} .delta-down {{ color:#e08585; }}
.blocker {{ color:#c8a06a; font-size:12px; margin-top:6px; }}
details summary {{ cursor:pointer; color:#9a9a9a; font-size:12px; }}
</style>
</head>
<body>
<h1>photo-ai-toolkit &mdash; {len(records)} assets</h1>
{_mode_banner(summary, language)}
<p class="disclaimer">{html.escape(t("warn.disclaimer", language))}</p>
<div class="summary">{_summary_html(summary, language)}</div>
{top_section}
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
    return path


def _top_photos_html(records: list[AssetRecord], base: Path, language: str) -> str:
    """The TOP pile, above everything else and visually separated.

    Empty is a legitimate result and says so out loud rather than vanishing: a
    shoot with no top photographs in it is a normal thing, and a missing section
    reads as a bug in the report.
    """
    top = [r for r in records if r.category == "TOP"]
    if not top:
        return (
            f'<div class="top"><h2>{html.escape(t("summary.top_photos", language))}</h2>'
            f'<p class="disclaimer" style="margin:0">'
            f"{html.escape(t('summary.no_top_photos', language))}</p></div>"
        )
    cards = "".join(_card(record, base, language) for record in top)
    return (
        f'<div class="top"><h2>{html.escape(t("summary.top_photos", language))}'
        f" &mdash; {len(top)}</h2>"
        f'<div class="grid">{cards}</div></div>'
    )


def _mode_banner(summary: dict, language: str) -> str:
    """A report from a run whose content check never happened must say so.

    Large, red and above the fold, because the failure this guards against is a
    user reading a local-only report as though the pictures had been looked at.
    """
    mode = summary.get("analysis_mode", "local_only")
    if mode == "local_and_semantic":
        return (
            f'<p class="disclaimer">{html.escape(t("mode.title", language))}: '
            f'{html.escape(t(f"mode.{mode}", language))}</p>'
        )
    return (
        f'<div class="banner"><b>{html.escape(t("mode.banner", language))}</b>'
        f'<span>{html.escape(t("mode.banner_detail", language))}</span>'
        f'<span>{html.escape(t("mode.title", language))}: '
        f'{html.escape(t(f"mode.{mode}", language))}</span></div>'
    )


def _summary_html(summary: dict, language: str) -> str:
    tooltips = {
        t("summary.technically_usable", language): t("summary.technically_usable_help", language),
        t("summary.fully_checked", language): t("summary.fully_checked_help", language),
    }
    stats = [
        (t("summary.total", language), summary.get("total", 0)),
        (t("summary.technically_usable", language), summary.get("technically_usable", 0)),
        (t("summary.fully_checked", language), summary.get("fully_checked", 0)),
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
        f'<div class="stat"{_title(tooltips.get(label))}><b>{value}</b>'
        f"<span>{html.escape(str(label))}</span></div>"
        for label, value in stats
    )


def _title(text: str | None) -> str:
    return f' title="{html.escape(text)}"' if text else ""


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
    darkroom_html = _darkroom_html(record, base, language)

    category_html = ""
    if record.category:
        delta = (record.final_score_detail or {}).get("stage3_delta") or 0
        # Signed, and only shown when it is non-zero -- "stage3 +0" on every card
        # of a local-only run says nothing and reads as though Stage 3 ran.
        moved = (
            f' <span class="{"delta-up" if delta > 0 else "delta-down"}">stage3 {delta:+d}</span>'
            if delta
            else ""
        )
        category_html = (
            f'<span class="badge" style="background:'
            f'{CATEGORY_COLORS.get(record.category, "#444")}">'
            f'{html.escape(t(f"category.{record.category}", language))} '
            f"{record.final_score}</span>{moved}"
        )

    blockers_html = (
        f'<div class="blocker">{html.escape(t("misc.not_for_stock", language))}: '
        + html.escape("; ".join(record.commercial_blockers))
        + "</div>"
        if record.commercial_blockers
        else ""
    )
    category_reasons_html = (
        "<ul>"
        + "".join(f"<li>{html.escape(str(r))}</li>" for r in record.category_reasons)
        + "</ul>"
        if record.category_reasons
        else ""
    )
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
{category_html}
{category_reasons_html}
<span class="badge" style="background:{color}">{class_label}</span>
<span class="badge" style="background:#333">{route_label}</span>
{blockers_html}
<div class="scores">{score_rows}{gain_html}</div>
{issue_list("unrecoverable", "bad")}{issue_list("partially_fixable", "mid")}{issue_list("fixable", "ok")}
{darkroom_html}
{recipe}
{markets_html}
{warnings}
<details><summary>{html.escape(t("misc.reasons", language))}</summary><ul>
{"".join(f"<li>{html.escape(str(r))}</li>" for r in record.reasons)}
</ul></details>
</div></div>"""


def _darkroom_html(record: AssetRecord, base: Path, language: str) -> str:
    """The A/B panel: what was proposed, what was protected, what was refused.

    A pilot cannot be run by reading JSON. The three things a photographer
    actually needs side by side are the original, the variants, and the reason a
    candidate was thrown away -- the last one especially, because it says what
    the tool nearly did to the frame.
    """
    if not (record.edit_recipes or record.darkroom_rejections or record.decision_bucket):
        return ""

    parts: list[str] = ['<details open><summary>Darkroom</summary>']

    if record.decision_bucket:
        parts.append(f'<div><span class="bucket">{html.escape(record.decision_bucket)}</span></div>')

    variants = record.rendered_variants or {}
    if variants:
        order = ["original", "faithful", "expressive", "monochrome"]
        cells = []
        for name in order:
            if name not in variants:
                continue
            src = _relative_preview(variants[name], base)
            cells.append(
                f'<figure><img src="{html.escape(src)}" alt="" loading="lazy">'
                f"<figcaption>{html.escape(name)}</figcaption></figure>"
            )
        if cells:
            parts.append('<div class="ab">' + "".join(cells) + "</div>")

    for item in record.preserve_intent or []:
        parts.append(f'<div class="keep">keep: {html.escape(str(item))}</div>')

    for recipe in record.edit_recipes or []:
        steps = recipe.get("human_readable") or []
        parts.append(
            f"<div style=\"font-size:12px;margin-top:6px\"><b>{html.escape(recipe.get('variant', ''))}</b>"
            f" &mdash; {html.escape(recipe.get('intent', ''))}</div><ul>"
            + "".join(f"<li>{html.escape(str(step))}</li>" for step in steps)
            + "</ul>"
        )
        for warning in recipe.get("warnings") or []:
            parts.append(f'<div class="mid" style="font-size:12px">{html.escape(str(warning))}</div>')

    for rejection in record.darkroom_rejections or []:
        parts.append(f'<div class="veto">refused: {html.escape(str(rejection))}</div>')

    confidence = record.recipe_confidence or {}
    if confidence:
        first = next(iter(confidence.values()))
        parts.append(
            '<div style="font-size:12px;color:#9a9a9a">confidence &mdash; '
            f"tone {first.get('tone', 0):.2f}, colour {first.get('color', 0):.2f}, "
            f"crop {first.get('crop', 0):.2f}</div>"
        )

    if record.darkroom_engine:
        parts.append(
            '<div style="font-size:11px;color:#777">engine: '
            f"{html.escape(record.darkroom_engine)} {html.escape(record.darkroom_engine_version)}</div>"
        )
    if record.suggested_sidecars:
        names = ", ".join(sorted(record.suggested_sidecars))
        parts.append(f'<div style="font-size:11px;color:#777">sidecars: {html.escape(names)}</div>')

    if record.policy_evidence:
        parts.append(
            '<details><summary>why the tool held back</summary><ul>'
            + "".join(f"<li>{html.escape(str(r))}</li>" for r in record.policy_evidence[:6])
            + "</ul></details>"
        )

    parts.append("</details>")
    return "".join(parts)


def _relative_preview(preview_path: str, base: Path) -> str:
    if not preview_path:
        return ""
    try:
        return os.path.relpath(Path(preview_path), base)
    except ValueError:
        return preview_path
