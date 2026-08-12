"""darktable sidecar. A different schema that happens to share an extension.

darktable stores its history stack under the `darktable:` namespace, in its own
encoding, and reads nothing from Adobe's `crs:` keys. Emitting one file with
both namespaces would produce a document each program silently half-ignores, so
the two exporters stay separate and a test asserts they never mix.

**Not verified against a real darktable installation** -- none is present on the
machine this was written on. The structure follows the documented XMP workflow;
the parameter blobs darktable uses for individual modules are version-specific
binary encodings and are deliberately not forged here. What is written is the
history-free metadata plus the recipe in the toolkit's own namespace, which
darktable preserves and a human can read.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

TEMPLATE = """<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="photo-ai-toolkit">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:darktable="http://darktable.sf.net/"
    xmlns:pat="https://github.com/karasov-co/photo-ai-toolkit/ns/1.0/"
    darktable:import_timestamp="-1"
    darktable:history_end="0"
    pat:suggestedBy="photo-ai-toolkit"
    pat:variant="{variant}"
    pat:intent="{intent}"
    pat:sourceChecksum="{checksum}"
    pat:exposureEV="{exposure}"
    pat:highlights="{highlights}"
    pat:shadows="{shadows}"
    pat:contrast="{contrast}"
    pat:blacks="{blacks}"
    pat:whites="{whites}"
    pat:temperatureDeltaK="{temperature}"
    pat:denoiseLuminance="{denoise}"
    pat:sharpening="{sharpening}"
    pat:monochrome="{monochrome}"
    pat:note="Advisory values. darktable module parameters are version-specific and are not forged here.">
   <pat:preserve><rdf:Bag>
{preserve}
   </rdf:Bag></pat:preserve>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>
"""


def to_darktable_xmp(recipe) -> str:
    g = recipe.global_adjustments
    return TEMPLATE.format(
        variant=escape(recipe.variant),
        intent=escape(recipe.intent),
        checksum=escape(recipe.source_checksum),
        exposure=f"{g.exposure_ev:+.2f}",
        highlights=g.highlights,
        shadows=g.shadows,
        contrast=g.contrast,
        blacks=g.blacks,
        whites=g.whites,
        temperature=g.temperature_delta_k,
        denoise=recipe.detail.denoise_luminance,
        sharpening=recipe.detail.sharpening,
        monochrome=str(recipe.color.monochrome).lower(),
        preserve="\n".join(f"    <rdf:li>{escape(p)}</rdf:li>" for p in recipe.preserve),
    )


def write_suggestion(recipe, root: Path) -> Path:
    from edit_schema import suggestion_path

    path = suggestion_path(root, recipe.asset_id or "unknown", recipe.variant, ".dt.xmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_darktable_xmp(recipe), encoding="utf-8")
    return path
