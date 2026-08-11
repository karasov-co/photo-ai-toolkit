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
import logging
import os
import shlex
from pathlib import Path

from PIL import Image, ImageDraw

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
                    "logos": a.logos,
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
    draw = ImageDraw.Draw(sheet)
    for i, (label, thumb) in enumerate(loaded):
        x = (i % columns) * cell_w
        y = (i // columns) * cell_h
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.height + 6), label[:38], fill=(230, 230, 230))

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
    "trash": "trash_quarantine",
    "review": "manual_review",
    "stock_standard": "stock/standard",
    "stock_strong": "stock/strong",
    "flagship": "portfolio/flagship",
}


def build_class_farm(records: list, out_dir: Path) -> dict[str, int]:
    """Symlinks by class, then by genre, then per eligible marketplace.

    Three views of the same files, none of which copies or moves anything. The
    genre and marketplace views exist because that is how the work is actually
    used: you export a marketplace batch, and you build a portfolio page from a
    genre.
    """
    counts: dict[str, int] = {}
    extra = ("stock/editorial", "stock/by_genre", "portfolio/by_genre",
             "stock/marketplace_packages", "archive")
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
    """The trash list, the script, and nothing else. Deletes nothing.

    The script moves to the Trash rather than calling rm. Everything on this
    list was proposed by a measurement or a model, and both have already been
    wrong on this archive.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [r for r in records if r.route_class == "trash"]

    listing = out_dir / "delete_candidates.txt"
    lines = [
        "# Proposed deletions. Nothing has been removed.",
        "# Look at contact_sheet_delete.jpg before running delete.sh.",
        f"# {len(candidates)} candidate(s).",
        "",
    ]
    for record in candidates:
        reason = "; ".join(record.reasons) or "below every threshold"
        lines.append(f"{record.source_path}\t{reason}")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    script = out_dir / "delete.sh"
    pairs = [(r.source_path, "; ".join(r.reasons) or "below every threshold") for r in candidates]
    script.write_text(_delete_script_for_paths(pairs), encoding="utf-8")
    script.chmod(0o755)
    return {"count": len(candidates), "listing": listing, "script": script}


def _delete_script_for_paths(pairs: list[tuple[str, str]]) -> str:
    lines = [
        "#!/bin/bash",
        "# Generated by photo-ai-toolkit. Review contact_sheet_delete.jpg first.",
        "#",
        "# Moves the listed files to the macOS Trash. It does not run rm, so a",
        "# mistake here is recoverable -- put them back from the Trash.",
        "set -euo pipefail",
        "",
        'TRASH="$HOME/.Trash"',
        "moved=0",
        "",
    ]
    for source, reason in pairs:
        quoted = shlex.quote(str(source))
        lines += [
            f"# {reason}",
            f"if [ -e {quoted} ]; then",
            f'  mv -n {quoted} "$TRASH"/ && moved=$((moved+1))',
            "else",
            f"  echo 'already gone: {Path(source).name}'",
            "fi",
            "",
        ]
    lines += [
        'echo "Moved $moved file(s) to $TRASH."',
        'echo "Nothing was permanently deleted; empty the Trash yourself when happy."',
        "",
    ]
    return "\n".join(lines)


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
