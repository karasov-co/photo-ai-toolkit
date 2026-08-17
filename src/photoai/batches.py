"""What each run actually looked at, recorded so the next one can tell.

Adding 252 photographs to a folder that already holds 47 is the normal way this
tool gets used, and it creates a question nothing else in the pipeline can
answer: *which of these did I just analyse?* Without a record, the insights page
re-describes the whole archive every time, so a photographer who imports a new
shoot is told again what their photography was like six months ago.

So every run writes a manifest: what was new, what had changed on disk, what was
answered from store, what failed, and which model and prompt versions produced
it. Append-only, one JSON object per line, because a corrupted last line costs
one run's provenance rather than all of it.

The manifest is also what makes an incremental run auditable. "47 reused" is a
claim; a list of checksums that were reused, beside the checksums the previous
run recorded, is a fact you can check.

One thing deliberately absent: anything derived from where the files came from.
A folder called `Japan` or `incoming` or `250` is a filesystem location, not
evidence about the photographs, and treating a path as context is how a tool
starts telling somebody their pictures are about Japan because of a directory
name.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "batches.jsonl"
SCHEMA_VERSION = 1


@dataclass
class BatchManifest:
    """One run's account of itself."""

    run_id: str = ""
    started_at: str = ""
    finished_at: str = ""
    schema_version: int = SCHEMA_VERSION

    new: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    # Every asset key this run saw. Recorded because "modified" cannot be
    # detected from checksums alone: a file whose contents changed is a
    # checksum nobody has seen, and only its *path* connects it to the analysis
    # it invalidates.
    keys: list[str] = field(default_factory=list)

    model: str = ""
    stage2_prompt_version: str = ""
    stage3_prompt_version: str = ""
    analyzer_version: str = ""

    llm_calls: dict = field(default_factory=dict)
    stage2_completed: int = 0
    stage3_completed: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> BatchManifest:
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (payload or {}).items() if k in known})

    @property
    def analysed(self) -> set[str]:
        """Checksums this run actually spent work on."""
        return set(self.new) | set(self.modified)


def new_run_id() -> str:
    """Sortable, and unique without coordination."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def classify_assets(assets, cache, model: str, *, seen_keys: set[str] | None = None):
    """Split what was found into new, modified and reused.

    "Modified" means a path that was analysed before and now hashes to
    something else -- the photograph was edited or replaced, so the stored
    analysis describes a file that no longer exists. It is distinguished from
    "new" because the two mean different things to the person reading the
    summary: one is work they added, the other is work they changed.
    """
    seen_keys = seen_keys or set()

    new, modified, reused = [], [], []
    for asset in assets:
        if cache.has_full_result(asset.checksum, model):
            reused.append(asset)
        elif asset.key in seen_keys:
            modified.append(asset)
        else:
            new.append(asset)
    return new, modified, reused


def seen_keys(manifests: list[BatchManifest]) -> set[str]:
    """Every asset key any previous run recorded."""
    return {key for manifest in manifests for key in manifest.keys}


def append(manifest: BatchManifest, path: Path) -> Path:
    """One line per run. Append-only on purpose."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(manifest.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_all(path: Path) -> list[BatchManifest]:
    """Every run recorded here, oldest first. A bad line is skipped, not fatal."""
    if not Path(path).is_file():
        return []
    out: list[BatchManifest] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(BatchManifest.from_dict(json.loads(line)))
        except (ValueError, TypeError) as e:
            logger.warning("Skipping an unreadable batch manifest line: %s", e)
    return out


def latest(path: Path) -> BatchManifest | None:
    manifests = read_all(path)
    return manifests[-1] if manifests else None


def scope_records(records, manifest: BatchManifest | None, scope: str = "new"):
    """The records the insights page should describe.

    `new` means this run's work. Falling back to everything when the manifest
    records nothing is deliberate: a re-run that reused every result would
    otherwise produce an insights page about nothing at all, which is a worse
    answer than one about the archive.
    """
    if scope == "all" or manifest is None:
        return list(records), "all"
    wanted = manifest.analysed
    if not wanted:
        return list(records), "all"
    selected = [r for r in records if r.checksum in wanted]
    if not selected:
        return list(records), "all"
    return selected, "new"
