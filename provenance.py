"""How the pixels came to exist, and what that costs at each marketplace.

The rule this module enforces is that provenance is **declared, never guessed**.
There is no reliable way to look at an image and know whether a generative tool
touched it, and a wrong guess is expensive in both directions: labelling a real
photograph as AI-generated bars it from Alamy and Shutterstock, while failing to
label a generated one is a policy breach that carries account strikes. So the
default is `UNKNOWN` plus whatever the file's own metadata actually says, and
`UNKNOWN` is treated as "ask the user", not as "probably fine".

What *is* detectable is a declaration somebody already made: C2PA content
credentials, `IPTC DigitalSourceType`, XMP written by a known generator, and the
EXIF Software tag. Those are read as evidence. Their absence is not evidence of
anything -- metadata is trivially stripped.

The second job is the edit-side warning. The tool's own suggested recipes are
non-generative by construction (see `edit_recipe.py`), so applying one never
changes an asset's marketplace eligibility. If a user applies a generative edit
of their own, that is a state change with consequences, and `warn_for_edit` is
what says so before it happens rather than after a rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class Provenance(StrEnum):
    CAMERA_ORIGINAL = "camera_original"
    SCANNED_FILM = "scanned_film"
    TRADITIONAL_EDIT = "traditional_edit"
    AI_ASSISTED_RETOUCH = "ai_assisted_retouch"
    GENERATIVE_FILL_USED = "generative_fill_used"
    PARTIALLY_GENERATED = "partially_generated"
    FULLY_GENERATED = "fully_generated"
    UNKNOWN = "unknown"


# Anything at or past this point in the list is generative for policy purposes.
GENERATIVE = frozenset(
    {
        Provenance.GENERATIVE_FILL_USED,
        Provenance.PARTIALLY_GENERATED,
        Provenance.FULLY_GENERATED,
    }
)

# Tools that mimic a conventional darkroom operation. Alamy names denoise
# explicitly as acceptable on an original photograph; the distinction the
# platforms draw is between enhancing captured detail and inventing new detail.
CONVENTIONAL_AI = frozenset({Provenance.AI_ASSISTED_RETOUCH})

GENERATOR_HINTS = (
    "midjourney",
    "dall-e",
    "dalle",
    "stable diffusion",
    "stablediffusion",
    "firefly",
    "flux",
    "imagen",
    "ideogram",
    "leonardo.ai",
    "generative fill",
)

RETOUCH_HINTS = ("denoise", "topaz", "lightroom", "camera raw", "capture one", "dxo", "photoshop")


@dataclass
class ProvenanceRecord:
    value: Provenance = Provenance.UNKNOWN
    declared_by: str = "default"
    evidence: list[str] = field(default_factory=list)

    @property
    def is_generative(self) -> bool:
        return self.value in GENERATIVE

    @property
    def needs_label(self) -> bool:
        """Whether a submission must carry an AI declaration."""
        return self.is_generative

    @property
    def is_uncertain(self) -> bool:
        return self.value is Provenance.UNKNOWN


def from_metadata(metadata: dict | None) -> ProvenanceRecord:
    """Read a declaration out of the file. Absence proves nothing.

    Only positive signals are acted on. A file with no AI metadata is left
    `UNKNOWN` rather than promoted to `CAMERA_ORIGINAL`, because stripping
    metadata is one command and the promotion would launder exactly the files
    the platforms care about.
    """
    if not metadata:
        return ProvenanceRecord()

    evidence: list[str] = []
    haystack = " ".join(
        str(metadata.get(k, "")).lower()
        for k in ("software", "Software", "creator_tool", "CreatorTool", "digital_source_type", "c2pa")
    )

    digital_source = str(metadata.get("digital_source_type", "")).lower()
    if "trainedalgorithmicmedia" in digital_source or "compositewithtrainedalgorithmicmedia" in digital_source:
        evidence.append(f"IPTC DigitalSourceType = {metadata.get('digital_source_type')}")
        value = (
            Provenance.FULLY_GENERATED
            if "trainedalgorithmicmedia" in digital_source
            and "composite" not in digital_source
            else Provenance.PARTIALLY_GENERATED
        )
        return ProvenanceRecord(value=value, declared_by="metadata", evidence=evidence)

    for hint in GENERATOR_HINTS:
        if hint in haystack:
            evidence.append(f"generator signature in metadata: {hint}")
            return ProvenanceRecord(
                value=Provenance.PARTIALLY_GENERATED, declared_by="metadata", evidence=evidence
            )

    if metadata.get("c2pa"):
        evidence.append("C2PA content credentials present")

    for hint in RETOUCH_HINTS:
        if hint in haystack:
            evidence.append(f"conventional editing software: {hint}")
            return ProvenanceRecord(
                value=Provenance.TRADITIONAL_EDIT, declared_by="metadata", evidence=evidence
            )

    if metadata.get("camera_make") or metadata.get("camera_model"):
        evidence.append(
            f"camera EXIF present ({metadata.get('camera_make')} {metadata.get('camera_model')})".strip()
        )
        return ProvenanceRecord(
            value=Provenance.CAMERA_ORIGINAL, declared_by="metadata", evidence=evidence
        )

    return ProvenanceRecord(evidence=evidence)


@dataclass
class PolicyConflict:
    platform: str
    severity: str
    message: str


def conflicts_for(record: ProvenanceRecord, platforms: list[dict]) -> list[PolicyConflict]:
    """Where this provenance is a problem, and how much of one."""
    found: list[PolicyConflict] = []
    for platform in platforms:
        policy = platform.get("generative_ai", "accepted_with_label")
        name = platform.get("name", platform.get("id", "?"))

        if record.is_generative and policy == "rejected":
            found.append(
                PolicyConflict(
                    platform=platform.get("id", ""),
                    severity="blocking",
                    message=(
                        f"{name} does not accept generative content: "
                        f"{platform.get('generative_ai_note', '')}"
                    ).strip(),
                )
            )
        elif record.is_generative and policy == "accepted_with_label":
            found.append(
                PolicyConflict(
                    platform=platform.get("id", ""),
                    severity="action_required",
                    message=f"{name} accepts this only when declared as generative at submission.",
                )
            )
        elif record.is_uncertain and policy == "rejected":
            found.append(
                PolicyConflict(
                    platform=platform.get("id", ""),
                    severity="advisory",
                    message=(
                        f"{name} rejects generative content and provenance is undeclared. "
                        "Confirm this is a camera original before submitting."
                    ),
                )
            )
    return found


def warn_for_edit(record: ProvenanceRecord, uses_generative: bool) -> list[str]:
    """What an edit would cost, said before it is applied rather than after."""
    if not uses_generative:
        return []
    warnings = [
        "This edit uses generative tools. Applying it changes the asset's provenance to "
        "'partially generated' and bars it from Alamy, Shutterstock and Getty/iStock.",
        "It also breaks editorial authenticity: a generatively altered frame cannot be "
        "submitted as documentary or editorial content anywhere.",
    ]
    if record.value is Provenance.CAMERA_ORIGINAL:
        warnings.append(
            "The file is currently a camera original, which is its most valuable state. "
            "Consider exporting the generative version as a separate asset."
        )
    return warnings


def label_for_submission(record: ProvenanceRecord) -> str:
    """The string a contributor form wants."""
    if record.value is Provenance.FULLY_GENERATED:
        return "Generative AI"
    if record.value in GENERATIVE:
        return "Contains generative AI elements"
    if record.value in CONVENTIONAL_AI:
        return "AI-assisted retouching only (no generative content)"
    if record.value is Provenance.UNKNOWN:
        return "Undeclared -- confirm before submitting"
    return "Not generative AI"
