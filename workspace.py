"""Where everything a run produces is written, and what a photographer sees.

The output directory used to be the tool's filing cabinet: caches, previews,
manifests, monitoring state, four parallel symlink views, contact sheets,
marketplace packages and three report formats, all at the top level. Opening it
told you what the software does rather than what it found in your photographs.

So the root now holds exactly what a person came for:

    report.html                 the five piles, ranked
    photographer_insights.html  what the collection says about the photography
    top/  good_stock/  good_personal/  needs_decision/  weak/
    edit_recipes/               XMP sidecars for the frames worth editing
    .internal/                  everything else

Nothing is deleted by moving it. `.internal/` holds the full JSON and CSV, the
previews, the log, the analysis cache, the routing views, the quarantine and the
diagnostics -- all of it still written, still readable, and still what the
`report`, `reclassify` and `quarantine` commands read from. The leading dot is
the whole trick: Finder and `ls` hide it, and a photographer never has to decide
whether `model_monitoring.json` is something they need.

The category folders hold symlinks. Analysis never copies, moves or removes an
original, and the categories are not permitted to: a symlink farm can be
rebuilt from the JSON at any time, which is exactly why it is safe to rebuild it
from scratch on every run.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

INTERNAL = ".internal"
RECIPES = "edit_recipes"
REPORT_NAME = "report.html"
INSIGHTS_NAME = "photographer_insights.html"

# The five piles, in the order they are shown. Keyed by `AssetRecord.category`.
CATEGORY_DIRS: dict[str, str] = {
    "TOP": "top",
    "GOOD_STOCK": "good_stock",
    "GOOD_EDITORIAL": "good_editorial",
    "GOOD_PERSONAL": "good_personal",
    "NEEDS_DECISION": "needs_decision",
    "WEAK": "weak",
}

# Everything the old layout put at the root. Each is moved under `.internal/`
# when an existing output directory is re-used, so a second run tidies up after
# the first rather than leaving two conventions side by side.
LEGACY_ROOT_ENTRIES = (
    "reports",
    "previews",
    "analysis_cache.json",
    "processing.log",
    "model_monitoring.json",
    "overrides.json",
    "archive",
    "duplicate_review",
    "manual_review",
    "portfolio",
    "stock",
    "proposed_for_removal",
    "trash_quarantine",
    "suggestions",
    "test_results",
    "good",
)


class Workspace:
    """The paths for one run. Created once and passed around."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.internal = self.root / INTERNAL

    # --- what a photographer opens ---------------------------------------

    @property
    def report(self) -> Path:
        return self.root / REPORT_NAME

    @property
    def insights(self) -> Path:
        return self.root / INSIGHTS_NAME

    @property
    def recipes(self) -> Path:
        return self.root / RECIPES

    def category(self, name: str) -> Path:
        return self.root / CATEGORY_DIRS[name]

    # --- what the tool keeps ----------------------------------------------

    @property
    def previews(self) -> Path:
        return self.internal / "previews"

    @property
    def reports(self) -> Path:
        return self.internal / "reports"

    @property
    def cache(self) -> Path:
        return self.internal / "analysis_cache.json"

    @property
    def log(self) -> Path:
        return self.internal / "processing.log"

    @property
    def monitoring(self) -> Path:
        return self.internal / "model_monitoring.json"

    @property
    def overrides(self) -> Path:
        return self.internal / "overrides.json"

    @property
    def quarantine(self) -> Path:
        return self.internal / "quarantine"

    @property
    def routing_views(self) -> Path:
        """The class-based symlink farm. Still built, no longer at the root."""
        return self.internal / "routing"

    def create(self) -> Workspace:
        self.root.mkdir(parents=True, exist_ok=True)
        self.internal.mkdir(parents=True, exist_ok=True)
        for name in CATEGORY_DIRS.values():
            (self.root / name).mkdir(exist_ok=True)
        self.recipes.mkdir(exist_ok=True)
        self.previews.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        return self


def migrate(root: Path) -> list[str]:
    """Move an older run's top-level clutter under `.internal/`.

    Returns what was moved, for the log. Anything already present in
    `.internal/` is left alone and the stale copy removed only if it is a
    symlink directory the farm rebuilds anyway -- a real file is never
    destroyed to make room, it is renamed out of the way instead.
    """
    root = Path(root)
    if not root.is_dir():
        return []

    internal = root / INTERNAL
    moved: list[str] = []
    for name in LEGACY_ROOT_ENTRIES:
        source = root / name
        if not source.exists() and not source.is_symlink():
            continue
        internal.mkdir(parents=True, exist_ok=True)
        destination = internal / name
        if destination.exists() or destination.is_symlink():
            destination = _free_name(internal, name)
        try:
            shutil.move(str(source), str(destination))
        except OSError as e:  # pragma: no cover - permissions, or a live mount
            logger.warning("Could not move %s into %s: %s", source, INTERNAL, e)
            continue
        moved.append(name)

    if moved:
        logger.info("Moved %d legacy output entries into %s/", len(moved), INTERNAL)
    return moved


def _free_name(parent: Path, name: str) -> Path:
    """`name`, `name-1`, `name-2`... The first that does not exist."""
    for suffix in range(1, 1000):
        candidate = parent / f"{name}-{suffix}"
        if not candidate.exists() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"no free name for {name} in {parent}")


def build_category_farm(records, workspace: Workspace) -> dict[str, int]:
    """Symlinks per category, rebuilt from scratch. Originals are never touched.

    Rebuilding wholesale is safe precisely because these are links: the worst a
    bug here can do is leave a folder that points at nothing, and the folder is
    regenerated from `analysis.json` on the next run.
    """
    from layout import _relink

    counts: dict[str, int] = dict.fromkeys(CATEGORY_DIRS, 0)
    for name in CATEGORY_DIRS.values():
        folder = workspace.root / name
        folder.mkdir(parents=True, exist_ok=True)
        for entry in folder.iterdir():
            if entry.is_symlink():
                entry.unlink()

    for record in records:
        folder = CATEGORY_DIRS.get(record.category)
        if not folder:
            continue
        _relink(workspace.root / folder / record.filename, Path(record.source_path))
        counts[record.category] += 1
    return counts


# --- transactional publication ------------------------------------------------
#
# A run that fails must leave the previous run intact. That is not a nicety: the
# outputs are what a photographer works from, and half-replacing them is worse
# than not running at all -- a report listing 47 photographs beside category
# folders holding 299 is not recoverable by looking at it.
#
# So everything is built in a staging directory and moved into place only once
# it is complete. The move is per top-level artefact, each one backed up first,
# and any failure rolls the whole set back.

STAGED = "staging"

# What a complete run must have produced. Validated before anything is swapped,
# because "the staging directory exists" is not the same as "the run worked".
REQUIRED_ARTEFACTS = (REPORT_NAME, INSIGHTS_NAME)


class PublishError(RuntimeError):
    """Staging did not produce a complete run, so nothing was replaced."""


def staging_dir(space: Workspace, run_id: str) -> Path:
    path = space.internal / STAGED / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_staging(staged: Path) -> list[str]:
    """Everything missing from a supposedly finished run."""
    missing = [name for name in REQUIRED_ARTEFACTS if not (staged / name).is_file()]
    if not any((staged / folder).is_dir() for folder in CATEGORY_DIRS.values()):
        missing.append("category folders")
    return missing


def publish(space: Workspace, staged: Path) -> list[str]:
    """Move a validated staging directory into place, or change nothing.

    Each artefact is swapped by rename, which is atomic per entry; the set of
    them is not, so a failure part-way through restores what was already moved.
    That is the difference between a run that fails and a run that destroys the
    last good one.
    """
    missing = validate_staging(staged)
    if missing:
        raise PublishError(f"the run did not produce: {', '.join(missing)}")

    entries = [name for name in (*REQUIRED_ARTEFACTS, RECIPES, *CATEGORY_DIRS.values())
               if (staged / name).exists()]
    backups: list[tuple[Path, Path]] = []
    published: list[str] = []

    try:
        for name in entries:
            live = space.root / name
            incoming = staged / name
            if live.exists() or live.is_symlink():
                backup = space.internal / f"{STAGED}-backup-{name.replace('/', '_')}"
                _remove(backup)
                live.rename(backup)
                backups.append((backup, live))
            incoming.rename(live)
            published.append(name)
    except OSError as e:  # pragma: no cover - filesystem failure mid-swap
        logger.error("Publication failed after %d artefact(s); rolling back: %s", len(published), e)
        for name in published:
            _remove(space.root / name)
        for backup, live in backups:
            if not live.exists():
                backup.rename(live)
        raise PublishError(str(e)) from e

    for backup, _ in backups:
        _remove(backup)
    _remove(staged)
    return published


def _remove(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
