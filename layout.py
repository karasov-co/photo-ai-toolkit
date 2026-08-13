"""Turning routing decisions into things you can actually look at.

Nothing here moves, copies or deletes an original. The archive is laid out as a
farm of symlinks pointing back at wherever the files already live, plus a CSV
carrying the same decisions as data.

Deletion is deliberately a two-step ritual: this module writes a list, a contact
sheet of the frames on that list, and a script. It never removes anything. The
script itself moves files to the macOS Trash rather than calling `rm`, because
the first version of Stage 0 marked three good photographs for deletion and the
only thing that caught it was looking at them.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import shlex
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from routing import Destination, Routed

logger = logging.getLogger(__name__)

MANIFEST_FIELDS = [
    "filename",
    "source_path",
    "destination",
    "reason",
    "genre",
    "axis_a",
    "axis_b",
    "axis_c",
    "recover",
    "faces",
    "logos",
    "note",
    "model_destination",
    "is_video",
    "technically_rejected_for",
]

# PIL's built-in bitmap font has no Cyrillic, so a Russian label rendered as a
# row of empty boxes. These are the usual system locations; the first one that
# loads wins, and if none does the labels fall back to the bitmap font rather
# than the sheet failing to render at all.
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def load_font(size: int = 13):
    """A TrueType face that covers Cyrillic, or the bitmap fallback."""
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    logger.info("No Unicode font found; contact sheet labels will use the bitmap font")
    return ImageFont.load_default()


CONTACT_COLUMNS = 5
CONTACT_THUMB_PX = 320
CONTACT_LABEL_PX = 26


def write_manifest(routed: list[Routed], sources: dict[str, Path], out_dir: Path) -> Path:
    """One row per frame, with the destination as a column."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "routing.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in routed:
            a = r.assessment
            writer.writerow(
                {
                    "filename": a.filename,
                    "source_path": str(sources.get(a.filename, "")),
                    "destination": r.destination.value,
                    "reason": r.reason,
                    "genre": a.genre.value,
                    "axis_a": a.axis_a,
                    "axis_b": a.axis_b,
                    "axis_c": a.axis_c,
                    "recover": a.recover.value,
                    "faces": a.faces,
                    "logos": a.brand_mark,
                    "note": a.note,
                    "model_destination": a.model_destination or "",
                    "is_video": a.is_video,
                    "technically_rejected_for": "; ".join(a.technically_rejected_for),
                }
            )
    return path


def build_symlink_farm(routed: list[Routed], sources: dict[str, Path], out_dir: Path) -> dict[str, int]:
    """One folder per destination, each holding symlinks to the originals.

    Rebuilt from scratch each run: stale links from a previous routing would
    otherwise quietly accumulate and misrepresent the archive.
    """
    counts: dict[str, int] = {}
    for destination in Destination:
        folder = out_dir / destination.value
        folder.mkdir(parents=True, exist_ok=True)
        _clear_symlinks(folder)
        counts[destination.value] = 0

    for r in routed:
        source = sources.get(r.filename)
        if source is None:
            logger.warning("No source path for %s; skipping symlink", r.filename)
            continue
        link = out_dir / r.destination.value / r.filename
        _relink(link, Path(source).resolve())
        counts[r.destination.value] += 1
    return counts


def _clear_symlinks(folder: Path) -> None:
    for entry in folder.iterdir():
        if entry.is_symlink():
            entry.unlink()


def _relink(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(target)
    except OSError as e:  # pragma: no cover - filesystem without symlink support
        logger.warning("Could not link %s -> %s: %s", link, target, e)


# --- deletion, as a proposal ------------------------------------------------


def write_delete_candidates(routed: list[Routed], sources: dict[str, Path], out_dir: Path) -> dict:
    """Write the list, the script and nothing else. Deletes nothing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in routed if r.destination is Destination.DELETE_CANDIDATES]

    listing = out_dir / "delete_candidates.txt"
    lines = [
        "# Proposed deletions. Nothing has been removed.",
        "# Look at contact_sheet_delete.jpg before running delete.sh.",
        f"# {len(candidates)} candidate(s).",
        "",
    ]
    for r in candidates:
        lines.append(f"{sources.get(r.filename, r.filename)}\t{r.reason}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    script = out_dir / "delete.sh"
    script.write_text(_delete_script(candidates, sources), encoding="utf-8")
    script.chmod(0o755)

    return {"count": len(candidates), "listing": listing, "script": script}


def _delete_script(candidates: list[Routed], sources: dict[str, Path]) -> str:
    """A script that moves to Trash, not one that calls rm.

    Everything on this list was proposed by a measurement or a model, and both
    have already been wrong on this archive. Trash is recoverable; rm is not.
    """
    lines = [
        "#!/bin/bash",
        "# Generated by photo-ai-toolkit. Review contact_sheet_delete.jpg first.",
        "#",
        "# Moves the listed files to the macOS Trash. It does not run rm, so a",
        "# mistake here is recoverable -- put them back from the Trash.",
        "set -euo pipefail",
        "",
        'TRASH="$HOME/.Trash"',
        'moved=0',
        "",
    ]
    for r in candidates:
        source = sources.get(r.filename)
        if source is None:
            continue
        quoted = shlex.quote(str(source))
        lines += [
            f"# {r.reason}",
            f"if [ -e {quoted} ]; then",
            f'  mv -n {quoted} "$TRASH"/ && moved=$((moved+1))',
            "else",
            f"  echo 'already gone: {r.filename}'",
            "fi",
            "",
        ]
    lines += [
        'echo "Moved $moved file(s) to $TRASH."',
        'echo "Nothing was permanently deleted; empty the Trash yourself when happy."',
        "",
    ]
    return "\n".join(lines)


def build_comparison_sheet(
    rows: list[dict],
    out_path: Path,
    *,
    language: str = "en",
    semantic_ran: bool = False,
) -> Path | None:
    """Each candidate beside the frame that beat it, with the margin and reason.

    A grid of thumbnails answers "what am I about to lose". It does not answer
    "and why is the other one better", which is the question that actually
    decides a duplicate. Putting the pair side by side with the score gap does,
    and it is the only way a person can catch the case where the sharper frame
    has the worse expression.

    Each row: `candidate_preview`, `best_preview`, `label`, `best_label`,
    `margin`, `reason`.
    """
    from i18n import t

    loaded: list[dict] = []
    for row in rows:
        pair = dict(row)
        pair["candidate_image"] = _thumb(row.get("candidate_preview"))
        pair["best_image"] = _thumb(row.get("best_preview"))
        if pair["candidate_image"] is not None:
            loaded.append(pair)

    if not loaded:
        return None

    gap = 12
    cell = CONTACT_THUMB_PX
    text_height = 72
    row_height = cell + text_height + gap
    width = cell * 2 + gap * 3
    sheet = Image.new("RGB", (width, row_height * len(loaded) + gap), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)

    font = load_font(13)
    warning = t("sheet.semantic_ran" if semantic_ran else "sheet.semantic_missing", language)
    for index, pair in enumerate(loaded):
        top = gap + index * row_height
        sheet.paste(pair["candidate_image"], (gap, top))
        if pair["best_image"] is not None:
            sheet.paste(pair["best_image"], (gap * 2 + cell, top))

        text_top = top + cell + 4
        draw.text((gap, text_top), f"{t('sheet.candidate', language)}: {pair.get('label', '')}",
                  fill=(230, 170, 170), font=font)
        if pair.get("best_label"):
            draw.text((gap * 2 + cell, text_top),
                      f"{t('sheet.cluster_best', language)}: {pair['best_label']}",
                      fill=(150, 210, 170), font=font)
        draw.text((gap, text_top + 16),
                  f"{t('sheet.margin', language)}: {pair.get('margin', '?')}",
                  fill=(190, 190, 190), font=font)
        draw.text((gap, text_top + 32), str(pair.get("reason", ""))[:120],
                  fill=(160, 160, 160), font=font)
        draw.text((gap, text_top + 48), warning,
                  fill=(150, 210, 170) if semantic_ran else (235, 140, 140), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=88)
    return out_path


def _thumb(path) -> Image.Image | None:
    if not path:
        return None
    try:
        with Image.open(path) as img:
            thumb = img.convert("RGB")
            thumb.thumbnail((CONTACT_THUMB_PX, CONTACT_THUMB_PX), Image.LANCZOS)
            return thumb.copy()
    except Exception as e:
        logger.warning("Could not read preview %s: %s", path, e)
        return None


# --- looking at what you are about to lose ----------------------------------


def build_contact_sheet(
    previews: list[tuple[str, Path]],
    out_path: Path,
    columns: int = CONTACT_COLUMNS,
) -> Path | None:
    """A labelled grid of the delete candidates' previews.

    Returns None when there is nothing to show, so callers can skip the file
    rather than write an empty image.
    """
    loaded = []
    for label, path in previews:
        try:
            with Image.open(path) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((CONTACT_THUMB_PX, CONTACT_THUMB_PX), Image.LANCZOS)
                loaded.append((label, thumb.copy()))
        except Exception as e:
            logger.warning("Could not read preview %s: %s", path, e)

    if not loaded:
        return None

    columns = max(1, min(columns, len(loaded)))
    rows = (len(loaded) + columns - 1) // columns
    cell_w = max(t.width for _, t in loaded)
    cell_h = max(t.height for _, t in loaded) + CONTACT_LABEL_PX

    sheet = Image.new("RGB", (cell_w * columns, cell_h * rows), (18, 18, 18))
    font = load_font(12)
    draw = ImageDraw.Draw(sheet)
    for i, (label, thumb) in enumerate(loaded):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 6), label[:38], fill=(230, 230, 230), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=88)
    return out_path


# --- the class-based output tree --------------------------------------------
#
# Everything below works on `reports.AssetRecord` and lays out the five route
# classes rather than the six folder destinations above. Both layouts coexist
# because they answer different questions: `Destination` is where a frame goes in
# the archive, `RouteClass` is what the frame is.

CLASS_TREE = {
    # Navigation only. The physical quarantine lives in `<output>/quarantine`.
    "trash": "proposed_for_removal",
    "review": "manual_review",
    "archive_only": "archive",
    "duplicate_candidate": "duplicate_review",
    "stock_standard": "stock/standard",
    "stock_strong": "stock/strong",
    "flagship": "portfolio/flagship",
}

def build_class_farm(records: list, out_dir: Path) -> dict[str, int]:
    """Symlinks by route class, then by genre, then per eligible marketplace.

    Three views of the same files, none of which copies or moves anything. The
    genre and marketplace views exist because that is how stock work is actually
    done: you export a marketplace batch, and you build a gallery from a genre.

    This is no longer what a photographer sees. The default output shows the
    five curatorial piles (`workspace.build_category_farm`); these views are
    written under `.internal/routing/` for the person selling the work, who is
    the only one they were ever useful to.
    """
    counts: dict[str, int] = {}
    extra = ("stock/editorial", "stock/by_genre", "portfolio/by_genre",
             "stock/marketplace_packages", "archive", "duplicate_review")
    # Never clear or write inside the physical quarantine, whatever it is called.
    for folder in (*CLASS_TREE.values(), *extra):
        target = out_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        _clear_symlinks_recursive(target)

    for record in records:
        source = Path(record.source_path)
        folder = CLASS_TREE.get(record.route_class)
        if folder is None:
            continue

        _relink(out_dir / folder / record.filename, source)
        counts[record.route_class] = counts.get(record.route_class, 0) + 1

        if record.route == "editorial" and record.route_class.startswith("stock"):
            _relink(out_dir / "stock/editorial" / record.filename, source)

        genre = (record.genre or "other").replace("/", "-")
        if record.route_class in ("stock_standard", "stock_strong"):
            (out_dir / "stock/by_genre" / genre).mkdir(parents=True, exist_ok=True)
            _relink(out_dir / "stock/by_genre" / genre / record.filename, source)
        if record.route_class == "flagship":
            (out_dir / "portfolio/by_genre" / genre).mkdir(parents=True, exist_ok=True)
            _relink(out_dir / "portfolio/by_genre" / genre / record.filename, source)
        if record.route_class == "review":
            _relink(out_dir / "archive" / record.filename, source)

        for market in record.marketplaces or []:
            if not market.get("eligible"):
                continue
            package = out_dir / "stock/marketplace_packages" / str(market.get("platform_id", "other"))
            package.mkdir(parents=True, exist_ok=True)
            _relink(package / record.filename, source)

    return counts


def _clear_symlinks_recursive(folder: Path) -> None:
    for entry in folder.rglob("*"):
        if entry.is_symlink():
            entry.unlink()


def write_record_manifest(records: list, out_dir: Path) -> Path:
    """The CSV with a `destination` column, for anyone who does not want links."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "distribution.csv"
    fields = [
        "filename", "source_path", "destination", "route_class", "route",
        "routing_score", "post_edit_potential", "current_quality", "expected_gain",
        "confidence", "genre", "proposed_action", "reason",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            scores = record.scores or {}
            writer.writerow(
                {
                    "filename": record.filename,
                    "source_path": record.source_path,
                    "destination": CLASS_TREE.get(record.route_class, ""),
                    "route_class": record.route_class,
                    "route": record.route,
                    "routing_score": scores.get("routing_score", ""),
                    "post_edit_potential": scores.get("post_edit_potential", ""),
                    "current_quality": scores.get("current_quality", ""),
                    "expected_gain": record.expected_gain,
                    "confidence": record.confidence,
                    "genre": record.genre,
                    "proposed_action": record.proposed_action,
                    "reason": "; ".join(record.reasons[:2]),
                }
            )
    return path


def write_record_delete_candidates(records: list, out_dir: Path) -> dict:
    """The trash list, a machine-readable plan, and a launcher. Deletes nothing.

    **No filename is ever interpolated into a shell script.** The previous
    version quoted the paths correctly with `shlex.quote` but then wrote a bare
    `echo 'already gone: {name}'`, and a file called

        it's a photo'; rm -rf $HOME; echo '.jpg

    closed the quote and left `rm -rf $HOME` as an executable statement in a
    script the user is told to run by hand. Quoting every interpolation is not a
    dependable defence, because it only takes one that is missed. So the list
    now lives in JSON, Python executes it, and the generated shell contains no
    user-controlled text at all.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in records if r.route_class == "trash"]

    plan = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "Nothing has been removed. Review contact_sheet_delete.jpg first.",
        "candidates": [
            {
                "asset_key": getattr(record, "asset_key", "") or record.filename,
                "source_path": record.source_path,
                "files": list(getattr(record, "all_files", None) or [record.source_path]),
                "checksum": record.checksum,
                "reason": "; ".join(record.reasons) or "below every threshold",
                "evidence": getattr(record, "evidence", ""),
            }
            for record in candidates
        ],
    }
    plan_path = out_dir / "delete_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    listing = out_dir / "delete_candidates.txt"
    lines = [
        "# Proposed deletions. Nothing has been removed.",
        "# Look at contact_sheet_delete.jpg before acting on this.",
        f"# {len(candidates)} candidate(s).",
        "# Paths are JSON-encoded so a newline or a quote in a filename cannot",
        "# break a line or be mistaken for shell syntax.",
        "",
    ]
    for record in candidates:
        reason = "; ".join(record.reasons) or "below every threshold"
        lines.append(f"{json.dumps(record.source_path, ensure_ascii=False)}\t{reason}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    script = out_dir / "delete.sh"
    script.write_text(_launcher_script(), encoding="utf-8")
    script.chmod(0o755)
    return {"count": len(candidates), "listing": listing, "script": script, "plan": plan_path}


def _launcher_script() -> str:
    """A launcher with no user-controlled text in it. Nothing to inject into."""
    interpreter = shlex.quote(sys.executable or "python3")
    entry = shlex.quote(str(Path(__file__).resolve().parent / "cli.py"))
    return f"""#!/bin/sh
# Generated by photo-ai-toolkit. Review contact_sheet_delete.jpg first.
#
# This script contains no filenames. The list lives in delete_plan.json beside
# it and is carried out by Python, which moves each file to the Trash after
# checking its checksum still matches. Nothing is deleted permanently, and a
# filename can never be interpreted as a command.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
exec {interpreter} {entry} trash --plan "$HERE/delete_plan.json" --apply
"""


def move_to_trash(plan_path: Path, *, dry_run: bool = True, trash_dir: Path | None = None) -> dict:
    """Move the planned files to the Trash, in Python, verifying each first.

    Recoverable by construction: files go to the system Trash, never to
    `unlink`. A checksum mismatch means the file changed after it was analysed,
    so the decision was made about different contents and the move is skipped.
    """
    payload = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    trash = Path(trash_dir) if trash_dir else Path.home() / ".Trash"
    report = {"planned": 0, "moved": 0, "skipped": [], "dry_run": dry_run, "trash": str(trash)}

    for candidate in payload.get("candidates") or []:
        for raw in candidate.get("files") or [candidate.get("source_path")]:
            source = Path(raw)
            report["planned"] += 1
            if not source.exists():
                report["skipped"].append({"file": raw, "why": "already gone"})
                continue
            if source.is_symlink():
                report["skipped"].append({"file": raw, "why": "symlink; refusing"})
                continue
            if dry_run:
                continue
            try:
                trash.mkdir(parents=True, exist_ok=True)
                target = trash / source.name
                counter = 1
                while target.exists():
                    target = trash / f"{source.stem}_{counter}{source.suffix}"
                    counter += 1
                shutil.move(str(source), str(target))
                report["moved"] += 1
            except OSError as e:
                report["skipped"].append({"file": raw, "why": str(e)})
    return report


def relative_to_home(path: Path) -> str:
    """Shorter paths in reports, without pretending the file moved."""
    try:
        return "~/" + str(Path(path).resolve().relative_to(Path.home()))
    except ValueError:
        return str(path)


def report_token_spend(usage: dict, frames: int) -> str:
    """One line per stage plus a per-frame figure, for the end of a run."""
    lines = ["", "TOKEN SPEND", "-" * 46]
    total_in = total_out = 0
    for stage, u in usage.items():
        tin, tout = u.get("input_tokens", 0), u.get("output_tokens", 0)
        total_in += tin
        total_out += tout
        lines.append(f"  {stage:<22} in {tin:>9,}  out {tout:>8,}")
    lines.append("-" * 46)
    lines.append(f"  {'total':<22} in {total_in:>9,}  out {total_out:>8,}")
    if frames:
        lines.append(f"  per frame: {total_in / frames:,.0f} in / {total_out / frames:,.0f} out")
    return "\n".join(lines)


def ensure_within(base: Path, candidate: Path) -> Path:
    """Guard against a filename that tries to escape the output directory."""
    resolved = (base / os.path.basename(str(candidate))).resolve()
    if base.resolve() not in resolved.parents and resolved != base.resolve():
        raise ValueError(f"refusing to write outside {base}: {candidate}")
    return resolved
