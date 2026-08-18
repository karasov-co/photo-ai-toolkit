"""Everything that can go wrong before a single cent is spent.

A stranger's first run fails for one of six reasons, and every one of them is
knowable in under a second: no key, a key the provider rejects, no FFmpeg, a RAW
format LibRaw was not built for, nowhere to write, or nowhere to write *enough*.
Finding out on frame 1,700 of 2,000 is the difference between a bug report and a
deleted repository.

Nothing here calls a paid endpoint. The key is checked for presence and shape,
not for balance -- `analyze` already runs a real preflight, and this command
exists to be free.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Rough, and deliberately generous: previews, thumbnails and derivatives for a
# large shoot. Better to warn at 2 GB free than to fill somebody's disk.
BYTES_PER_PHOTOGRAPH = 1_200_000
MIN_FREE_BYTES = 2 * 1024**3


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = False
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(c.fatal and not c.ok for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {"name": c.name, "ok": c.ok, "detail": c.detail, "fatal": c.fatal, "fix": c.fix}
                for c in self.checks
            ],
        }


def run(*, input_dir: Path | None = None, output_dir: Path | None = None) -> Report:
    report = Report()
    report.checks.append(_python())
    report.checks.append(_pillow())
    report.checks.append(_rawpy())
    report.checks.append(_ffmpeg())
    report.checks.append(_embeddings())
    report.checks.append(_key())
    if input_dir:
        report.checks.append(_readable(input_dir))
    if output_dir:
        report.checks.append(_writable(output_dir))
        report.checks.append(_space(output_dir, input_dir))
    return report


def _python() -> Check:
    version = ".".join(str(p) for p in sys.version_info[:3])
    ok = sys.version_info >= (3, 12)
    return Check(
        "Python", ok, version, fatal=True,
        fix="" if ok else "This needs Python 3.12 or newer.",
    )


def _pillow() -> Check:
    try:
        import PIL

        return Check("Pillow", True, getattr(PIL, "__version__", "?"), fatal=True)
    except Exception as e:
        return Check("Pillow", False, str(e), fatal=True, fix="pip install -r requirements.txt")


def _rawpy() -> Check:
    """LibRaw, and which formats it will actually open on this machine.

    The commonest first issue on Windows: rawpy ships wheels for most platforms
    and builds from source where it does not, and that build needs a compiler.
    """
    try:
        import rawpy

        return Check(
            "LibRaw (rawpy)", True,
            f"{getattr(rawpy, '__version__', '?')} -- RAW files can be read",
        )
    except Exception as e:
        return Check(
            "LibRaw (rawpy)", False, str(e).splitlines()[0],
            fix=(
                "RAW files will be skipped; JPEG still works. "
                "pip install rawpy, and on Windows install the Visual C++ build "
                "tools first if it tries to compile."
            ),
        )


def _ffmpeg() -> Check:
    binary = shutil.which("ffmpeg")
    return Check(
        "FFmpeg", bool(binary), binary or "not on PATH",
        fix="" if binary else (
            "Video is skipped without it. macOS: brew install ffmpeg. "
            "Debian/Ubuntu: apt-get install ffmpeg. Windows: choco install ffmpeg."
        ),
    )


def _embeddings() -> Check:
    """The semantic similarity encoder: installed, downloaded, verified, or not.

    Never fatal and never a download. An unticked line here means the diversity
    pass groups by palette and framing instead of by subject -- a real
    difference in what reaches the top pile, and one worth seeing before the run
    rather than deducing from the report afterwards.
    """
    from photoai import embeddings

    state = embeddings.status()
    if state.ok:
        return Check(
            "Semantic similarity", True,
            f"{state.model_id} ({embeddings.MODEL.licence}), verified at {state.path}",
        )
    return Check(
        "Semantic similarity", False, state.reason,
        fix=(
            f"Optional. Without it, near-duplicates are grouped by perceptual hash, "
            f"which merges two different subjects that share a palette. "
            f"{state.fix}"
        ),
    )


def _key() -> Check:
    """Presence and shape only. Balance is `analyze`'s preflight, which costs."""
    from photoai import bootstrap

    key = bootstrap.api_key()
    if not key:
        return Check(
            "API key", False, "not set",
            fix=(
                f"Without one, `analyze` runs the local pass and writes a full report; "
                f"it just cannot look at the pictures. Set {bootstrap.XAI_KEY_VAR} in "
                ".env for the content and artistic read."
            ),
        )
    source = "environment" if os.environ.get(bootstrap.XAI_KEY_VAR) else ".env"
    return Check("API key", True, f"found in the {source} ({len(key)} characters)")


def _readable(path: Path) -> Check:
    path = Path(path)
    if not path.is_dir():
        return Check("Input folder", False, f"{path} is not a directory", fatal=True)
    try:
        count = sum(1 for _ in path.rglob("*") if _.is_file())
    except OSError as e:
        return Check("Input folder", False, str(e), fatal=True)
    detail = f"{count} file(s)"
    if count > 20000:
        detail += " -- large; consider --limit for a first run"
    return Check("Input folder", True, detail)


def _writable(path: Path) -> Check:
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".photoai-write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return Check(
            "Output folder", False, str(e), fatal=True,
            fix="Choose a folder you can write to, or fix the permissions.",
        )
    return Check("Output folder", True, f"{path} is writable")


def _space(output_dir: Path, input_dir: Path | None) -> Check:
    try:
        free = shutil.disk_usage(Path(output_dir)).free
    except OSError as e:
        return Check("Free space", False, str(e))
    needed = MIN_FREE_BYTES
    if input_dir and Path(input_dir).is_dir():
        photographs = sum(1 for _ in Path(input_dir).rglob("*") if _.is_file())
        needed = max(needed, photographs * BYTES_PER_PHOTOGRAPH)
    ok = free >= needed
    return Check(
        "Free space", ok,
        f"{free / 1024**3:.1f} GB free, about {needed / 1024**3:.1f} GB wanted",
        fix="" if ok else "Previews and derivatives will not fit. Free some space first.",
    )


def format_report(report: Report) -> str:
    lines = ["", "=" * 60, "READINESS", "=" * 60]
    for check in report.checks:
        mark = "ok  " if check.ok else ("FAIL" if check.fatal else "warn")
        lines.append(f"  [{mark}] {check.name:<16} {check.detail}")
        if check.fix:
            for part in check.fix.split(". "):
                if part.strip():
                    lines.append(f"           {part.strip().rstrip('.')}.")
    lines.append("=" * 60)
    lines.append(
        "  Ready to run." if report.ok
        else "  Something above has to be fixed before a run can work."
    )
    return "\n".join(lines) + "\n"
