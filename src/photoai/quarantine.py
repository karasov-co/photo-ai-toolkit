"""Moving files, reversibly, or not moving them at all.

This module never deletes anything. The strongest thing it does is *move* a file
into a quarantine directory, and every move is recorded with enough information
to put it back exactly where it came from. Permanent removal is a separate
operation with its own confirmation, its own age requirement, and its own
refusal to run while anything else is in flight.

The design assumptions are pessimistic on purpose, because each of them has a
real failure behind it:

- **Dry run is the default.** `plan()` returns what would happen; `apply()` is a
  second, explicit call. A tool that reorganises 5000 files as a side effect of
  being run with the wrong flag is not recoverable by apology.

- **A group moves together or not at all.** `P1042675.RW2`, its JPEG twin and
  its `.xmp` are one photograph. Quarantining the RAW and leaving the sidecar
  produces an orphan that no software can interpret and no user can explain.

- **Idempotent by checksum.** Re-running a half-finished operation is normal --
  a terminal was closed, a disk filled. An already-moved file whose checksum
  matches the manifest is a no-op, not a collision and not a duplicate.

- **Both ends are fenced.** Sources must be inside a configured root and
  destinations inside the quarantine root, checked after `resolve()` so that
  `../../..` and a symlink pointing outside are the same rejected case.

- **Symlinks are not followed.** The tool's own output is a farm of symlinks
  into the archive. Following one during a move would relocate the original
  while the run believed it was touching a link.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from photoai.media import checksum_file

logger = logging.getLogger(__name__)

MANIFEST_NAME = "quarantine_manifest.jsonl"
LOCK_NAME = ".operation.lock"
LOCK_STALE_SECONDS = 3600
DEFAULT_PURGE_AGE_DAYS = 30
PURGE_CONFIRMATION = "PERMANENTLY DELETE"

# The only grounds on which a file may ever be permanently destroyed. Each is a
# demonstrable property of the file rather than an opinion about the picture: it
# does not decode, it holds no image, there is no usable segment in it, or an
# identical copy is being kept. `unusable_duration` is deliberately absent: a
# short clip is short, which is a marketplace fact and not damage. Everything else -- low score, weak composition,
# unappealing subject, a model's aesthetic judgement -- reaches quarantine at
# most, and stays there.
PURGEABLE_EVIDENCE = frozenset(
    {
        "corrupt_file",
        "encoding_corruption",
        "empty_frame",
        "no_usable_segment",
        "insufficient_resolution",
        "exact_duplicate",
    }
)


def is_purgeable_evidence(evidence: str) -> bool:
    """Whether these grounds justify destroying the file, not merely filing it.

    Deliberately a whitelist. A new issue code added later is not purgeable
    until somebody decides it is, which is the safe direction for a list whose
    failure mode is permanent.
    """
    if not evidence:
        return False
    return all(part.strip() in PURGEABLE_EVIDENCE for part in evidence.split(",") if part.strip())


class OperationStatus(StrEnum):
    PLANNED = "planned"
    MOVED = "moved"
    SKIPPED = "skipped"
    FAILED = "failed"
    RESTORED = "restored"
    PURGED = "purged"


class UnsafePath(ValueError):
    """A path that escapes its configured root. Always a refusal, never a warning."""


class OperationLocked(RuntimeError):
    pass


@dataclass
class FileOperation:
    op_id: str
    asset_id: str
    source: str
    destination: str
    checksum: str
    size_bytes: int
    timestamp: str
    reason: str
    route_class: str = ""
    status: str = OperationStatus.PLANNED.value
    is_sidecar: bool = False
    scores: dict = field(default_factory=dict)
    error: str = ""
    # Everything below exists so a move can be verified and undone as a group.
    group_id: str = ""
    expected: dict = field(default_factory=dict)
    evidence: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def expected_state(self):
        from photoai.media import FileState

        return FileState.from_dict(self.expected) if self.expected else None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def assert_within(root: Path, candidate: Path) -> Path:
    """Prove containment without following the destination itself.

    The **parent** is resolved and the leaf name appended, rather than resolving
    the whole path. That distinction is the fix for a real failure: the symlink
    farm wrote a navigation link into the physical quarantine directory, and
    that link pointed back at the source file. `candidate.resolve()` then
    returned the *source* path, which is of course outside the quarantine root,
    and planning refused two perfectly ordinary videos with

        .../trash_quarantine/P1019374.MP4 resolves outside .../trash_quarantine

    Resolving only the parent still catches everything containment is for --
    `../../etc/passwd`, a symlinked parent directory, an absolute path
    elsewhere -- because those are all properties of the directory chain. What
    it no longer does is treat a pre-existing file at the destination as though
    it were the destination.
    """
    resolved_root = root.resolve()
    # Always resolve the parent, existing or not. `Path.resolve()` normalises
    # `..` even for a path that is not on disk; skipping it when the parent was
    # missing let `<root>/../../etc/passwd` pass a purely lexical containment
    # check, because Path does not collapse `..` on its own.
    resolved = candidate.parent.resolve() / candidate.name

    if resolved != resolved_root and not resolved.is_relative_to(resolved_root):
        raise UnsafePath(f"{candidate} resolves outside {resolved_root}")
    return resolved


def is_generated_link(path: Path, source_roots: list[Path] | None = None) -> bool:
    """Whether this is one of our own navigation symlinks rather than a file.

    The farm links point back into the archive. A link at a quarantine
    destination is therefore ours and stale -- it cannot be a quarantined file,
    because quarantined files are real bytes that were moved here.
    """
    if not path.is_symlink():
        return False
    try:
        target = path.readlink()
        target = target if target.is_absolute() else (path.parent / target)
        target = target.resolve()
    except OSError:
        return True
    if not source_roots:
        return True
    return any(_contains(root, target) for root in source_roots)


def _contains(root: Path, target: Path) -> bool:
    """Whether `target` sits under `root`, on a filesystem of either persuasion.

    `is_relative_to` on resolved paths is not enough on Windows. A temporary
    directory reaches this function as `C:\\Users\\RUNNER~1\\...` and resolves to
    `C:\\Users\\runneradmin\\...`, and NTFS compares case-insensitively besides,
    so a link that genuinely was ours came back as somebody else's file. Both
    sides go through realpath and normcase before they are compared.
    """
    import os

    try:
        a = os.path.normcase(os.path.realpath(target))
        b = os.path.normcase(os.path.realpath(Path(root)))
    except OSError:  # pragma: no cover - an unreadable root is not containment
        return False
    return a == b or a.startswith(b + os.sep)


class Manifest:
    """Append-only JSONL. One line per operation, flushed as it goes.

    Append-only rather than a rewritten document because the file is the only
    record of where an original went. A rewrite that is interrupted halfway
    loses the ability to restore everything written before it.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, operation: FileOperation) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(operation.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load(self) -> list[FileOperation]:
        if not self.path.exists():
            return []
        operations: list[FileOperation] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                operations.append(FileOperation(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping unreadable manifest line: %s", e)
        return operations

    def latest_by_destination(self) -> dict[str, FileOperation]:
        """The current state of each destination, last write winning."""
        state: dict[str, FileOperation] = {}
        for op in self.load():
            state[op.destination] = op
        return state

    def by_operation(self, op_id: str) -> list[FileOperation]:
        return [op for op in self.load() if op.op_id == op_id]


class Lock:
    """A crude PID lock, so a purge cannot run beside a move.

    Stale locks expire rather than requiring manual cleanup: a process killed
    mid-run would otherwise wedge the directory permanently, and a user in that
    position will delete the lock file by hand anyway.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def acquire(self) -> None:
        if self.is_held():
            raise OperationLocked(f"another operation is running (lock: {self.path})")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"pid": os.getpid(), "at": time.time()}), encoding="utf-8")

    def release(self) -> None:
        self.path.unlink(missing_ok=True)

    def is_held(self) -> bool:
        if not self.path.exists():
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if time.time() - float(payload.get("at", 0)) > LOCK_STALE_SECONDS:
            logger.warning("Ignoring stale lock at %s", self.path)
            return False
        return True

    def __enter__(self) -> Lock:
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


@dataclass
class PlannedMove:
    asset_id: str
    files: list[Path]
    destination_dir: Path
    reason: str
    route_class: str = ""
    scores: dict = field(default_factory=dict)
    # What each file looked like when it was analysed. A move is refused if the
    # file on disk no longer matches: between analysis and quarantine the user
    # may have edited, replaced or re-exported it, and the decision was made
    # about the old contents.
    states: dict[str, dict] = field(default_factory=dict)
    # Why this asset is proposed for removal, in a form the purge gate can read.
    # An aesthetic judgement is never sufficient grounds for permanent deletion.
    evidence: str = ""


class Quarantine:
    """Reversible relocation, fenced at both ends."""

    def __init__(
        self,
        quarantine_dir: Path,
        *,
        source_roots: list[Path] | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.quarantine_dir = Path(quarantine_dir).resolve()
        self.source_roots = [Path(r).resolve() for r in (source_roots or [])]
        self.manifest = Manifest(manifest_path or self.quarantine_dir / MANIFEST_NAME)
        self.lock = Lock(self.quarantine_dir / LOCK_NAME)

    # --- planning -----------------------------------------------------------

    def plan(self, moves: list[PlannedMove]) -> list[FileOperation]:
        """What would happen. Touches nothing on disk."""
        op_id = uuid.uuid4().hex[:12]
        planned: list[FileOperation] = []
        for move in moves:
            for i, source in enumerate(move.files):
                try:
                    destination = self._destination_for(source, move.destination_dir)
                except UnsafePath as e:
                    logger.error("Refusing to plan %s: %s", source, e)
                    continue
                planned.append(
                    FileOperation(
                        op_id=op_id,
                        asset_id=move.asset_id,
                        group_id=f"{op_id}:{move.asset_id}",
                        source=str(source),
                        destination=str(destination),
                        checksum="",
                        size_bytes=source.stat().st_size if source.exists() else 0,
                        timestamp=_now(),
                        reason=move.reason,
                        route_class=move.route_class,
                        status=OperationStatus.PLANNED.value,
                        is_sidecar=i > 0,
                        scores=move.scores,
                        expected=move.states.get(str(source), {}),
                        evidence=move.evidence,
                    )
                )
        return planned

    def _destination_for(self, source: Path, destination_dir: Path) -> Path:
        """Mirror the source's layout under the destination.

        Preserving the relative structure is what makes a restore possible even
        if the manifest is lost -- the quarantine tree alone tells you where
        each file belonged.

        A stale navigation symlink sitting at the destination is stepped around
        with a collision-safe name rather than followed, removed, or treated as
        a containment violation.
        """
        self._assert_source_allowed(source)
        relative = self._relative_to_root(source)
        candidate = destination_dir / relative
        destination = assert_within(self.quarantine_dir, candidate)

        if destination.is_symlink():
            logger.info(
                "A stale link occupies %s; choosing a collision-safe name instead",
                destination.name,
            )
            destination = assert_within(self.quarantine_dir, _next_free_name(destination))
        return destination

    def _assert_source_allowed(self, source: Path) -> None:
        if not self.source_roots:
            return
        resolved = source.resolve() if source.exists() else source
        if not any(resolved.is_relative_to(root) for root in self.source_roots):
            raise UnsafePath(f"{source} is outside every configured source root")

    def _relative_to_root(self, source: Path) -> Path:
        resolved = source.resolve() if source.exists() else source
        for root in self.source_roots:
            if resolved.is_relative_to(root):
                return resolved.relative_to(root)
        return Path(resolved.name)

    # --- applying -----------------------------------------------------------

    def apply(self, planned: list[FileOperation], *, dry_run: bool = True) -> list[FileOperation]:
        """Carry out a plan, one asset group at a time, all-or-nothing.

        A RAW, its JPEG twin and its `.xmp` are one photograph. Moving them
        independently means a failure partway -- a full disk, a permission
        error, a file that vanished -- leaves the RAW in quarantine and the
        sidecar in the archive. That orphan is not something the user can
        interpret or the tool can restore, so a group either moves completely
        or is put back exactly as it was.
        """
        if dry_run:
            return planned

        results: list[FileOperation] = []
        with self.lock:
            for group in self._grouped(planned):
                results.extend(self._move_group(group))
        return results

    @staticmethod
    def _grouped(planned: list[FileOperation]) -> list[list[FileOperation]]:
        groups: dict[str, list[FileOperation]] = {}
        for op in planned:
            groups.setdefault(op.group_id or op.asset_id or op.source, []).append(op)
        return list(groups.values())

    def _move_group(self, ops: list[FileOperation]) -> list[FileOperation]:
        """Verify the whole group, move it, and roll back if anything fails."""
        refusal = self._verify_group(ops)
        if refusal is not None:
            return refusal

        moved: list[FileOperation] = []
        for op in ops:
            outcome = self._move_one(op, record=False)
            if outcome.status == OperationStatus.FAILED.value:
                restored, failures = self._rollback(moved)
                for done in moved:
                    done.status = OperationStatus.SKIPPED.value
                    done.error = (
                        f"rolled back: {outcome.source} failed ({outcome.error})"
                        if restored
                        else f"ROLLBACK INCOMPLETE after {outcome.error}: {failures}"
                    )
                results = [*moved, outcome, *ops[len(moved) + 1 :]]
                for remaining in ops[len(moved) + 1 :]:
                    remaining.status = OperationStatus.SKIPPED.value
                    remaining.error = "group aborted before this file was touched"
                for row in results:
                    self.manifest.append(row)
                return results
            moved.append(outcome)

        for row in moved:
            self.manifest.append(row)
        return moved

    def _verify_group(self, ops: list[FileOperation]) -> list[FileOperation] | None:
        """Refuse the whole group if any member changed since it was analysed."""
        for op in ops:
            source = Path(op.source)
            expected = op.expected_state
            if expected is None or not source.exists():
                continue
            unchanged, why = expected.matches(source)
            if unchanged:
                continue
            for row in ops:
                row.status = OperationStatus.SKIPPED.value
                row.error = (
                    f"{Path(op.source).name} changed since analysis ({why}); "
                    "re-analyse this asset before moving it"
                )
                self.manifest.append(row)
            logger.warning("Refusing to move %s: %s", op.source, why)
            return ops
        return None

    def _rollback(self, moved: list[FileOperation]) -> tuple[bool, list[str]]:
        """Put back everything this group has already moved."""
        failures: list[str] = []
        for op in reversed(moved):
            destination, source = Path(op.destination), Path(op.source)
            if not destination.exists() or source.exists():
                continue
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination), str(source))
            except OSError as e:
                failures.append(f"{destination} -> {source}: {e}")
                logger.error("Rollback failed for %s: %s", destination, e)
        return not failures, failures

    def _move_one(self, op: FileOperation, *, record: bool = True) -> FileOperation:
        source = Path(op.source)
        destination = Path(op.destination)

        def done(status: OperationStatus, error: str = "") -> FileOperation:
            op.status = status.value
            op.error = error
            if record:
                self.manifest.append(op)
            return op

        if source.is_symlink():
            return done(
                OperationStatus.SKIPPED,
                "source is a symlink; refusing to move what it points at",
            )

        if not source.exists():
            # Already moved by an interrupted earlier run, or moved by hand.
            # Neither aborts the group: a resumed run must be a no-op, not a
            # failure that rolls back the files it correctly moved last time.
            if destination.exists():
                return done(OperationStatus.SKIPPED, "already at the destination")
            return done(OperationStatus.SKIPPED, "source no longer exists")

        try:
            op.checksum = checksum_file(source)
        except OSError as e:
            return done(OperationStatus.FAILED, f"could not checksum: {e}")

        if destination.exists():
            if _same_file(destination, op.checksum):
                return done(OperationStatus.SKIPPED, "identical file already quarantined")
            destination = _next_free_name(destination)
            op.destination = str(destination)

        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            op.timestamp = _now()
            return done(OperationStatus.MOVED)
        except OSError as e:
            logger.error("Move failed for %s: %s", source, e)
            return done(OperationStatus.FAILED, str(e))

    # --- undo ---------------------------------------------------------------

    def restore(self, op_id: str | None = None, *, dry_run: bool = True) -> list[FileOperation]:
        """Put quarantined files back where they came from.

        Restores only what is currently `moved`: replaying a restore is a no-op
        rather than an error, and a file the user has already put back by hand
        is left alone.
        """
        candidates = [
            op
            for op in (self.manifest.by_operation(op_id) if op_id else self.manifest.load())
            if op.status == OperationStatus.MOVED.value
        ]
        still_moved = {
            op.destination: op
            for op in candidates
            if self.manifest.latest_by_destination().get(op.destination, op).status
            == OperationStatus.MOVED.value
        }

        results: list[FileOperation] = []
        if dry_run:
            return list(still_moved.values())

        with self.lock:
            for op in still_moved.values():
                results.append(self._restore_one(op))
        return results

    def _restore_one(self, op: FileOperation) -> FileOperation:
        destination = Path(op.destination)
        source = Path(op.source)
        restored = FileOperation(**{**op.to_dict(), "timestamp": _now()})

        if not destination.exists():
            restored.status = OperationStatus.FAILED.value
            restored.error = "quarantined file is gone"
            self.manifest.append(restored)
            return restored

        if source.exists():
            restored.status = OperationStatus.SKIPPED.value
            restored.error = "a file already occupies the original path"
            self.manifest.append(restored)
            return restored

        try:
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
            restored.status = OperationStatus.RESTORED.value
        except OSError as e:
            restored.status = OperationStatus.FAILED.value
            restored.error = str(e)

        self.manifest.append(restored)
        return restored

    # --- the only destructive operation -------------------------------------

    def purge(
        self,
        *,
        confirmation: str,
        older_than_days: int = DEFAULT_PURGE_AGE_DAYS,
        dry_run: bool = True,
    ) -> dict:
        """Permanently remove quarantined files. Every gate must be passed.

        Deliberately awkward. Requires a typed phrase, an age, an unlocked
        directory, and a manifest entry proving the file was quarantined by this
        tool rather than merely present in the folder.
        """
        if confirmation != PURGE_CONFIRMATION:
            raise ValueError(
                f"purge requires the exact confirmation phrase {PURGE_CONFIRMATION!r}"
            )
        if self.lock.is_held():
            raise OperationLocked("cannot purge while another operation is running")

        cutoff = time.time() - older_than_days * 86400
        eligible: list[FileOperation] = []
        refused: list[dict] = []
        for op in self.manifest.latest_by_destination().values():
            if op.status != OperationStatus.MOVED.value:
                continue
            destination = Path(op.destination)
            if not destination.exists():
                continue
            if destination.stat().st_mtime > cutoff:
                continue
            if not is_purgeable_evidence(op.evidence):
                # A judgement about how a photograph *looks* is not grounds for
                # destroying it, however old the file is and however confident
                # the score was. Only demonstrable technical failure qualifies,
                # and an unrecorded reason counts as not demonstrated.
                refused.append({"file": op.destination, "why": op.evidence or "grounds not recorded"})
                continue
            if op.checksum and not _same_file(destination, op.checksum):
                refused.append({"file": op.destination, "why": "checksum no longer matches manifest"})
                continue
            eligible.append(op)

        report = {
            "eligible": len(eligible),
            "bytes": sum(Path(op.destination).stat().st_size for op in eligible),
            "files": [op.destination for op in eligible],
            "refused": refused,
            "dry_run": dry_run,
            "purged": 0,
        }
        if dry_run:
            return report

        with self.lock:
            for op in eligible:
                try:
                    Path(op.destination).unlink()
                    purged = FileOperation(**{**op.to_dict(), "timestamp": _now()})
                    purged.status = OperationStatus.PURGED.value
                    self.manifest.append(purged)
                    report["purged"] += 1
                except OSError as e:
                    logger.error("Could not purge %s: %s", op.destination, e)
        return report

    # --- reporting ----------------------------------------------------------

    def state(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.manifest.latest_by_destination().values():
            counts[op.status] = counts.get(op.status, 0) + 1
        return counts

    def recoverable_bytes(self) -> int:
        return sum(
            op.size_bytes
            for op in self.manifest.latest_by_destination().values()
            if op.status == OperationStatus.MOVED.value
        )


def _same_file(path: Path, checksum: str) -> bool:
    try:
        return bool(checksum) and checksum_file(path, full=True) == checksum
    except OSError:
        return False


def _next_free_name(path: Path) -> Path:
    """`name.jpg` -> `name_1.jpg`. Never overwrites, never gives up silently."""
    stem, suffix, parent = path.stem, path.suffix, path.parent
    for i in range(1, 10_000):
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
    raise UnsafePath(f"could not find a free filename beside {path}")


def summarise_plan(planned: list[FileOperation], language: str = "en") -> str:
    """What a user sees before approving anything, in their language."""
    from photoai.i18n import t

    if not planned:
        return t("plan.nothing_to_move", language)

    by_reason: dict[str, int] = {}
    total = 0
    for op in planned:
        by_reason[op.reason] = by_reason.get(op.reason, 0) + 1
        total += op.size_bytes

    lines = [
        t("plan.would_move", language, count=len(planned), mb=f"{total / 1_048_576:.1f}"),
        *(f"  {count:>5}  {reason}" for reason, count in sorted(by_reason.items())),
        "",
        t("plan.dry_run", language),
    ]
    return "\n".join(lines)
