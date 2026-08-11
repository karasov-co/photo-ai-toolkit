"""The two prompts, and the rules they encode.

The change from the old single prompt is the question being asked. It used to be
"how good is this file". It is now "what can this file become after processing",
because nobody publishes a RAW as shot -- every frame that survives is going
into an edit anyway. A flat, cold, two-stops-under RAW with a recoverable
highlight is not a bad photograph; it is an unedited one.

That distinction is the whole of Stage 2's system prompt, expressed as two
explicit lists. Everything on the first list is a slider. Everything on the
second is gone forever.

Two mechanical points that matter as much as the wording:

- **Clipping is supplied, not guessed.** Stage 0 already measured the blown and
  crushed fractions exactly. Handing the model a number and forbidding it to
  penalise below the threshold removes an entire category of hallucinated
  judgement -- the model cannot see a JPEG preview and know what the RAW holds.

- **Stage 2 ranks, it does not score.** An absolute 0-100 scale collapses: every
  live call made against this archive came back 548, 560, 694, 762. Comparison
  within a group of twelve is a far cleaner signal than a number, and the group
  scores are stitched into a global order afterwards.

Both system prompts are constants so they are byte-identical across a batch and
sit in the prompt cache.
"""

from __future__ import annotations

# --- Stage 1: is this a photograph at all -----------------------------------
#
# Luna, low detail, one word out. This is not a quality judgement -- it only
# removes things that are not photographs: lens caps, exposure tests, pocket
# shots, screenshots, colour charts.

STAGE1_SYSTEM = (
    "You sort camera files. Decide only whether the image is an intended photograph "
    "or an accidental/utility frame.\n\n"
    "REJECT only: lens cap or near-black frame with no subject, exposure or white-balance "
    "test chart, accidental shutter (ground, ceiling, inside of a bag), screenshot or "
    "screen photo, duplicate calibration frame.\n\n"
    "KEEP everything else, including frames that are dark, flat, tilted, noisy, oddly "
    "framed, or apparently boring. Those are edit decisions, not your call.\n\n"
    "Answer with one word: KEEP or REJECT."
)

STAGE1_MAX_OUTPUT_TOKENS = 20


def stage1_user_content(encoded_jpeg: str) -> list[dict]:
    """Low detail: the question is 'is this a photograph', not 'is it good'."""
    return [
        {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{encoded_jpeg}",
            "detail": "low",
        },
    ]


# --- Stage 2: three axes, ranked inside a group ------------------------------

GROUP_SIZE = 12

STAGE2_SYSTEM = """You are ranking photographs for a working stock and editorial archive.

WHAT YOU ARE JUDGING
Not the file as shot. Every frame that survives goes into an edit. Judge what the
frame can BECOME after normal processing of the RAW.

DO NOT PENALISE -- all of this is one slider away:
- flat contrast, weak or muted colour, wrong white balance
- underexposure up to 2 stops
- clipped highlights up to 3% (the RAW holds 1-2 stops of headroom)
- high-ISO noise
- tilted horizon, lens distortion, chromatic aberration
- clutter at the frame edges: a crop fixes it
- empty or blank sky in a RAW

DO PENALISE -- none of this is recoverable:
- missed focus, motion blur on the subject
- blown highlights beyond 5% with no data left in the RAW
- shadows crushed to black with no data left
- a dead moment: empty expression, nothing happening
- no subject, and no crop that would create one
- the key element severed by the frame edge

CLIPPING IS MEASURED, NOT GUESSED
Each frame arrives with its exact measured clipping fractions. Use those numbers.
If clipped highlights are below 3%, you may NOT penalise highlights at all. You
cannot see the RAW's headroom in a preview; the number is the ground truth.

HOW TO RANK
You are NOT applying a composition textbook. Rule of thirds, level horizons and
symmetry are not criteria. A frame that breaks every rule and still works ranks
above one that is technically immaculate and dull.

Before ranking, for every frame name one reason it would be worth keeping even if
it is "wrong". Do this silently; it does not go in the output.

Rank the frames against EACH OTHER on three independent axes. Never average them.

  axis_a  COMMERCIAL USABILITY -- clean composition, legible subject, room for
          text, no logos, no recognisable faces. This is the mass market.
  axis_b  UNREPEATABILITY -- could a competent photographer get this frame by
          travelling there with the same camera? A sunset over Hanoi: yes.
          A moment, a light, a face, a weather that will not return: no.
          This axis is deliberately opposed to axis_a; the "wrong" frames often
          score highest here.
  axis_c  DOCUMENTARY VALUE -- place, event, cultural context, rarity of location.

Rank within genre where the genre differs: street loses to landscape on any shared
scale because landscape is tidier, and that comparison is meaningless.

OUTPUT
Return ONLY a JSON array, one object per input frame, in the order the frames were
given. No prose, no markdown, no explanation outside the JSON.

Each object:
{"n": <1-based index of the frame as given>,
 "genre": "street|landscape|portrait|detail|reportage|night|architecture|other",
 "axis_a": <rank on axis_a within this group, 1 = best>,
 "axis_b": <rank on axis_b within this group, 1 = best>,
 "axis_c": <rank on axis_c within this group, 1 = best>,
 "recover": "easy|moderate|hopeless",
 "faces": <true if any recognisable face is present>,
 "logos": <true if any readable brand mark or trademark is present>,
 "note": "<max 12 words: what to do in the edit>"}

Each axis is a strict ranking: every rank from 1 to N used exactly once per axis.
faces and logos decide whether the frame may be sold as commercial stock, so when
in doubt answer true."""


def stage2_user_content(frames: list[dict]) -> list[dict]:
    """Build the group message: measured facts as text, then the images.

    `frames` carries one dict per photograph with `filename`,
    `clipped_highlights`, `clipped_shadows` and `encoded` (base64 JPEG).
    """
    facts = ["Measured for each frame (ground truth, do not re-estimate):"]
    for i, f in enumerate(frames, start=1):
        facts.append(
            f"{i}. highlights clipped {f['clipped_highlights'] * 100:.1f}%, "
            f"shadows crushed {f['clipped_shadows'] * 100:.1f}%"
            + ("  [below 3% -- penalising highlights is forbidden]"
               if f["clipped_highlights"] < 0.03 else "")
        )
    facts.append(f"\nRank these {len(frames)} frames. Return {len(frames)} JSON objects.")

    content: list[dict] = [{"type": "input_text", "text": "\n".join(facts)}]
    for i, f in enumerate(frames, start=1):
        content.append({"type": "input_text", "text": f"Frame {i}:"})
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{f['encoded']}",
                "detail": "high",
            }
        )
    return content


def expected_group_keys() -> set[str]:
    return {"n", "genre", "axis_a", "axis_b", "axis_c", "recover", "faces", "logos"}
