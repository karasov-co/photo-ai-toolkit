"""Titles, descriptions and keywords -- built from evidence, capped on purpose.

The temptation with generated stock metadata is volume: platforms allow fifty
keywords, so fill fifty. That is a mistake with a real cost. Keywords that do
not describe what is visible degrade search relevance for the buyer, and every
platform covered here treats systematic irrelevant keywording as spam, which is
an account-level problem rather than a per-file one.

So keywords come only from things something actually observed -- what the vision
model described, the genre it assigned, the concepts it named, and hard facts
from EXIF -- deduplicated, ordered by how central they are, and capped well
below the platform maximum. `keyword_confidence` reports how much of the list is
model-derived versus inferred, so a thin result is visible rather than padded.

Everything written goes to a **sidecar or a derived export copy**. Originals are
never modified: an original with metadata burned into it is no longer the file
the camera produced, and that is not a state you can undo.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

MAX_KEYWORDS = 40
MAX_TITLE_CHARS = 120
MAX_DESCRIPTION_CHARS = 200
MIN_KEYWORD_CHARS = 3

# The practical stock taxonomy. Primary genre plus any number of secondaries.
GENRES = (
    "business and finance",
    "workplace",
    "technology",
    "lifestyle",
    "people",
    "family",
    "relationships",
    "education",
    "healthcare and wellness",
    "food and drink",
    "travel",
    "nature",
    "landscape",
    "wildlife",
    "architecture",
    "interiors",
    "real estate",
    "city and urban",
    "transportation",
    "industry",
    "agriculture",
    "sports and fitness",
    "events",
    "documentary",
    "product and still life",
    "beauty and fashion",
    "abstract",
    "backgrounds and textures",
    "seasonal and holiday",
    "environment and sustainability",
    "aerial and drone",
    "macro",
    "social media and creator",
    "cinematic footage",
)

# The internal ranking vocabulary maps onto the buyer-facing taxonomy. These are
# not the same vocabulary and should not be conflated: "detail" is a way of
# shooting, "product and still life" is a way of selling.
GENRE_ALIASES = {
    "street": "city and urban",
    "landscape": "landscape",
    "portrait": "people",
    "detail": "abstract",
    "reportage": "documentary",
    "night": "city and urban",
    "architecture": "architecture",
    "other": "lifestyle",
}

CONCEPT_VOCABULARY = (
    "remote work",
    "teamwork",
    "sustainability",
    "healthy lifestyle",
    "aging",
    "cybersecurity",
    "financial planning",
    "tourism",
    "logistics",
    "family connection",
    "mental health",
    "renewable energy",
    "urban life",
    "craftsmanship",
    "tradition",
    "solitude",
    "exploration",
)

STOPWORDS = frozenset(
    {
        "a", "an", "and", "the", "with", "of", "in", "on", "at", "to", "for", "from",
        "by", "is", "are", "was", "were", "this", "that", "it", "its", "as", "into",
        "over", "under", "very", "some", "any", "photo", "photograph", "image", "shot",
    }
)


@dataclass
class StockMetadata:
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    primary_category: str = ""
    secondary_category: str = ""
    genre: str = ""
    concepts: list[str] = field(default_factory=list)
    location: str = ""
    route: str = "commercial"
    people_count: int = 0
    ai_label: str = ""
    suggested_marketplaces: list[str] = field(default_factory=list)
    copyright: str = ""
    keyword_confidence: float = 0.0
    edited_by_user: bool = False

    @property
    def is_complete(self) -> bool:
        """What every platform asks for before a file can be submitted."""
        return bool(self.title and self.description and len(self.keywords) >= 5 and self.primary_category)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "keywords": self.keywords,
            "primary_category": self.primary_category,
            "secondary_category": self.secondary_category,
            "genre": self.genre,
            "concepts": self.concepts,
            "location": self.location,
            "route": self.route,
            "people_count": self.people_count,
            "ai_label": self.ai_label,
            "suggested_marketplaces": self.suggested_marketplaces,
            "copyright": self.copyright,
            "keyword_confidence": self.keyword_confidence,
            "edited_by_user": self.edited_by_user,
        }


def to_taxonomy(internal_genre: str) -> str:
    return GENRE_ALIASES.get(str(internal_genre).lower(), "lifestyle")


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z\-']+", str(text).lower()) if len(w) >= MIN_KEYWORD_CHARS]


def build_keywords(
    *,
    description: str,
    genre: str,
    concepts: list[str],
    secondary_genres: list[str],
    location: str = "",
    camera_keywords: list[str] | None = None,
    limit: int = MAX_KEYWORDS,
) -> tuple[list[str], float]:
    """Ordered keywords, most central first, plus how evidenced the list is.

    Order matters more than length at every platform covered here -- the first
    ten carry the search weight. So the sequence is deliberate: what it is,
    then what it means, then where it is, then how it was made.
    """
    ordered: list[str] = []
    evidenced = 0

    def push(term: str, *, from_model: bool) -> None:
        nonlocal evidenced
        clean = term.strip().lower()
        if len(clean) < MIN_KEYWORD_CHARS or clean in STOPWORDS or clean in ordered:
            return
        ordered.append(clean)
        if from_model:
            evidenced += 1

    taxonomy_genre = to_taxonomy(genre)
    push(taxonomy_genre, from_model=True)
    for secondary in secondary_genres or []:
        push(secondary, from_model=True)

    # Nouns from the description carry the literal content of the frame.
    for word in _words(description):
        push(word, from_model=True)

    for concept in concepts or []:
        push(concept, from_model=True)

    if location:
        for part in re.split(r"[,/]", location):
            push(part, from_model=True)

    for extra in camera_keywords or []:
        push(extra, from_model=False)

    trimmed = ordered[:limit]
    confidence = round(min(1.0, evidenced / max(len(trimmed), 1)), 3)
    return trimmed, confidence


def build_title(description: str, genre: str, location: str = "") -> str:
    """A caption a buyer would search for, not a filename."""
    text = (description or "").strip().rstrip(".")
    if not text:
        text = to_taxonomy(genre).title()
    if location and location.lower() not in text.lower():
        text = f"{text}, {location}"
    text = text[0].upper() + text[1:] if text else text
    return text[:MAX_TITLE_CHARS].rstrip(" ,;")


def build_description(description: str, concepts: list[str], route: str) -> str:
    parts = [(description or "").strip().rstrip(".")]
    if concepts:
        parts.append("Concepts: " + ", ".join(concepts[:4]))
    text = ". ".join(p for p in parts if p)
    return text[:MAX_DESCRIPTION_CHARS].rstrip(" ,;") + ("." if text else "")


def generate(
    *,
    semantic,
    route: str,
    exif: dict | None = None,
    provenance_label: str = "",
    marketplaces: list[str] | None = None,
    copyright_holder: str = "",
) -> StockMetadata:
    """Assemble the submission metadata from everything already known."""
    exif = exif or {}
    location = _location_from_exif(exif)
    camera_keywords = _camera_keywords(exif)

    keywords, confidence = build_keywords(
        description=semantic.description,
        genre=semantic.genre,
        concepts=semantic.concepts,
        secondary_genres=semantic.secondary_genres,
        location=location,
        camera_keywords=camera_keywords,
    )

    taxonomy = to_taxonomy(semantic.genre)
    secondary = next(
        (g for g in (semantic.secondary_genres or []) if g in GENRES and g != taxonomy),
        "",
    )

    return StockMetadata(
        title=build_title(semantic.description, semantic.genre, location),
        description=build_description(semantic.description, semantic.concepts, route),
        keywords=keywords,
        primary_category=taxonomy,
        secondary_category=secondary,
        genre=str(semantic.genre),
        concepts=list(semantic.concepts or []),
        location=location,
        route=route,
        people_count=int(semantic.people_count or 0),
        ai_label=provenance_label,
        suggested_marketplaces=list(marketplaces or []),
        copyright=copyright_holder,
        keyword_confidence=confidence,
    )


def _location_from_exif(exif: dict) -> str:
    """Only a location somebody recorded. Coordinates are not a place name."""
    for key in ("location", "city", "sublocation"):
        if exif.get(key):
            return str(exif[key])
    return ""


def _camera_keywords(exif: dict) -> list[str]:
    out: list[str] = []
    if exif.get("camera_make"):
        out.append(str(exif["camera_make"]).lower())
    focal = exif.get("focal_length")
    if isinstance(focal, int | float) and focal:
        if focal <= 24:
            out.append("wide angle")
        elif focal >= 85:
            out.append("telephoto")
    return out


# --- writing it out ---------------------------------------------------------


XMP_TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"
    xmlns:pat="https://github.com/karasov-co/photo-ai-toolkit/ns/1.0/">
   <dc:title><rdf:Alt><rdf:li xml:lang="x-default">{title}</rdf:li></rdf:Alt></dc:title>
   <dc:description><rdf:Alt><rdf:li xml:lang="x-default">{description}</rdf:li></rdf:Alt></dc:description>
   <dc:rights><rdf:Alt><rdf:li xml:lang="x-default">{copyright}</rdf:li></rdf:Alt></dc:rights>
   <dc:subject><rdf:Bag>
{keywords}
   </rdf:Bag></dc:subject>
   <photoshop:Category>{primary_category}</photoshop:Category>
   <Iptc4xmpCore:Location>{location}</Iptc4xmpCore:Location>
   <pat:route>{route}</pat:route>
   <pat:aiLabel>{ai_label}</pat:aiLabel>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def write_xmp_sidecar(metadata: StockMetadata, target: Path) -> Path:
    """Write `<name>.xmp` beside a derived export. Never touches an original."""
    keywords = "\n".join(f"    <rdf:li>{escape(k)}</rdf:li>" for k in metadata.keywords)
    body = XMP_TEMPLATE.format(
        title=escape(metadata.title),
        description=escape(metadata.description),
        copyright=escape(metadata.copyright),
        keywords=keywords,
        primary_category=escape(metadata.primary_category),
        location=escape(metadata.location),
        route=escape(metadata.route),
        ai_label=escape(metadata.ai_label),
    )
    path = target.with_suffix(".xmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


SUBMISSION_CSV_FIELDS = [
    "filename",
    "title",
    "description",
    "keywords",
    "category",
    "secondary_category",
    "route",
    "ai_label",
    "location",
    "suggested_marketplaces",
]


def write_submission_csv(rows: list[tuple[str, StockMetadata]], path: Path) -> Path:
    """The manifest a contributor uploads alongside a batch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_CSV_FIELDS)
        writer.writeheader()
        for filename, meta in rows:
            writer.writerow(
                {
                    "filename": filename,
                    "title": meta.title,
                    "description": meta.description,
                    "keywords": ", ".join(meta.keywords),
                    "category": meta.primary_category,
                    "secondary_category": meta.secondary_category,
                    "route": meta.route,
                    "ai_label": meta.ai_label,
                    "location": meta.location,
                    "suggested_marketplaces": ", ".join(meta.suggested_marketplaces),
                }
            )
    return path


def load_edits(path: Path) -> dict[str, dict]:
    """Read metadata a user edited by hand before export."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def apply_edits(metadata: StockMetadata, edits: dict) -> StockMetadata:
    """A user's edit wins over anything generated, and is marked as theirs."""
    if not edits:
        return metadata
    for key, value in edits.items():
        if hasattr(metadata, key) and key != "edited_by_user":
            setattr(metadata, key, value)
    metadata.edited_by_user = True
    return metadata


def strip_gps(exif: dict) -> dict:
    """Remove coordinates from what goes into an export copy.

    Location is private by default: a home, a school, a route walked daily. It
    is stripped from derived copies unless the user asks otherwise, and the
    original on disk is untouched either way.
    """
    return {k: v for k, v in exif.items() if k not in {"gps_lat", "gps_lon", "gps_altitude"}}
