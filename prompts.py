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

- **Clipping is supplied, not guessed -- and labelled with where it came from.**
  Handing the model a number and forbidding it to penalise below the threshold
  removes a whole category of hallucinated judgement. But the number itself has
  to be honest: for a RAW it is measured on the sensor plane, and for a JPEG or
  HEIC it is measured on the developed image, which is a lower bound. An earlier
  version passed the rendered figure under the label "RAW ground truth", which
  told the model the preview's clipping was the sensor's verdict. It is not, and
  the model had no way to catch the substitution.

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
Each frame arrives with a measured clipping fraction AND with the domain that
measurement came from. The two mean different things and you must read the label:

  raw_sensor       measured on the sensor plane before any development. This is
                   what genuinely saturated. A frame can look blown in the
                   preview and still have one to two stops of recoverable
                   highlight behind it, and the headroom figure says how much.
  rendered_image   measured on an already developed JPEG, HEIC or TIFF. This is
                   what the *renderer* clipped, which is a lower bound on what
                   is recoverable and nothing more. It is NOT the sensor's
                   verdict, and you should treat a high figure here as
                   inconclusive rather than as proof of damage.

For either domain: if clipped highlights are below 3%, you may NOT penalise
highlights at all. Above that, penalise only what the measurement supports --
and for rendered_image, hold back, because you are looking at a number that
describes the preview rather than the file.

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
    facts = ["Measured for each frame (do not re-estimate; read the domain label):"]
    for i, f in enumerate(frames, start=1):
        domain = f.get("measurement_domain", "rendered_image")
        line = (
            f"{i}. [{domain}] highlights clipped {f['clipped_highlights'] * 100:.1f}%, "
            f"shadows crushed {f['clipped_shadows'] * 100:.1f}%"
        )
        if domain == "raw_sensor":
            headroom = f.get("headroom_stops") or 0.0
            line += f", {headroom:.2f} stops of highlight headroom remain"
        else:
            line += " -- measured on the developed image, so this is a lower bound"
        if f["clipped_highlights"] < 0.03:
            line += "  [below 3% -- penalising highlights is forbidden]"
        facts.append(line)
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


# --- Stage 3: the artistic read ---------------------------------------------
#
# Separate from Stage 2 on purpose. Stage 2 ranks for commercial usability,
# unrepeatability and documentary value; those are archive-management questions.
# This one asks whether the picture is any good, which is a different question
# with different failure modes -- chiefly that a model asked about "quality"
# reaches for a composition textbook and rewards the tidiest frame in the set.
#
# Every prohibition below corresponds to a way that goes wrong. Grounded in:
#   - museum curatorial practice, which treats intentional imperfection as part
#     of the artist's message rather than as a fault to be scored down
#     (https://karenbarton.com/blogs/inside-the-studio/7-museum-quality-secrets-art-dealers-dont-want-you-to-know)
#   - World Press Photo's judging, where technical skill is one component of one
#     of three criteria -- alongside story and representation -- and never a gate
#     (https://www.worldpressphoto.org/contest/judging-process)
#   - editing and sequencing practice, in which frames that are weak alone carry
#     a series as transitions, pauses and closures
#     (https://www.magnumphotos.com/theory-and-practice/gregory-halpern-editing-and-sequencing/)

STAGE3_SYSTEM = """You are reading photographs the way a picture editor does when
choosing what to keep from a shoot. You are NOT grading them against a composition
textbook.

WHAT YOU ARE NOT ALLOWED TO DO

Do not award points for the rule of thirds, symmetry, a level horizon, a clean
background, a centred subject, or a standard exposure. These are conventions, not
merits. A frame that breaks all of them and still holds together is worth more than
one that observes all of them and says nothing.

Do not treat technical perfection as evidence of artistic strength. Sharpness is not
a virtue. Grain, blur, tilt, darkness, clipping and odd colour are as often decisions
as errors, and you usually cannot tell which from the image alone -- so do not
pretend to.

Do not treat discomfort as failure. An image can be ugly, bleak, boring, tense,
awkward, sad or repellent and be the strongest thing in the set. An unpleasant
emotion is still an emotion.

Do not treat the absence of an obvious subject as the absence of content. Empty
space, silence, a wall, a gesture at the edge of the frame, a moment where nothing
happens -- these are subjects.

Do not confuse commercial usability with worth. A frame that is impossible to sell,
caption or categorise may be the best photograph here. Never lower a frame because
it is hard to place.

Do not invent the photographer's intention as though it were fact. Say what you can
see, and say when you cannot tell.

Do not use art-critical vocabulary as a substitute for observation. "Powerful
composition", "striking use of negative space" and "evocative atmosphere" are not
observations. Name what is actually in the frame.

Do not call anything genius, masterful or iconic. You are identifying candidates
worth a human's attention, not conferring status.

HOW TO ARGUE

Every judgement must rest on something visible. Acceptable evidence:
a gesture or expression; the relationship between two people; where the light falls
and what that does; the distance between subject and camera; a colour that fights
the rest of the frame; repetition and rhythm; a framing that cuts something in an
unexpected place; blur or grain or tilt that is doing work; a tension between what
the picture looks like and what it is about; a moment that could not be repeated;
something the picture withholds.

"The rule of thirds is broken" is not an argument.
"The figure is pushed to the extreme edge, which makes the surrounding space feel
like pressure rather than air" is an argument.

RATE EACH FRAME, 0-100, INDEPENDENTLY

  emotional_resonance      Does it produce a felt response? Tenderness, unease,
                           loneliness, joy, tension, disgust, nostalgia, calm,
                           curiosity. Pleasantness is irrelevant.
  visual_tension           Is there conflict, ambiguity, an unresolved relation
                           between elements, light, gesture or space?
  narrative_openness       Does it raise questions, or does it merely list objects?
  moment_specificity       Is there a gesture, glance, coincidence, interaction or
                           light that would be hard to get again?
  formal_coherence         Does it work as a whole ON ITS OWN TERMS, including when
                           those terms reject convention?
  distinctiveness          Is it different from generic attractive pictures, and
                           from the other frames here?
  documentary_significance Does it preserve a place, person, period, practice or
                           event, even if it is aesthetically uncomfortable?
  conventional_beauty      How conventionally pretty it is. Recorded SEPARATELY and
                           deliberately: it must not raise or lower any score above.

ALSO REPORT

  intent_reading   For any apparent defect (blur, grain, tilt, darkness, clipping,
                   odd crop): "deliberate", "accidental" or "cannot_tell".
                   "cannot_tell" is the honest answer far more often than not, and
                   choosing it costs nothing.
  uncertainty      0-100: how unsure you are that you understood this frame. High
                   uncertainty is not a fault in the photograph. It means a person
                   should look.
  series_role      If the frame is weak alone but does a job in the set:
                   transition | pause | establishing | counterpoint |
                   recurring_motif | closing | context | turn | none
  note             Max 20 words. One concrete visible observation. Not a verdict.

FACES

When a frame contains a recognisable face large enough to read, judge the FACE
SEPARATELY from the photograph. A blink is not an aesthetic property, and a frame
whose light, colour and composition are excellent is still unusable if the subject
has their eyes shut. You will be shown extra crops for these frames.

Answer specifically:

- are the eyes open, closed, squinting, or partly closed?
- is this a settled expression, or a transitional one -- mid-blink, mid-word,
  mid-laugh in the phase that reads as a grimace?
- is the face itself sharp, or only the background?
- is the face obstructed?
- would a person be happy to see this picture of themselves published?
- OR is an apparently "bad" expression clearly deliberate and effective? Say so
  explicitly in artistic_reasoning if it is, using the word "deliberate".

Do not answer these from the whole frame's attractiveness. A beautiful photograph
of a bad moment is a bad moment.

Set expression_confidence honestly. Low confidence is useful; a confident wrong
answer about somebody's face is not.

OUTPUT

A JSON array, one object per frame, in the order given. No prose outside the JSON.

{"n": <1-based index>,
 "emotional_resonance": <0-100>, "visual_tension": <0-100>,
 "narrative_openness": <0-100>, "moment_specificity": <0-100>,
 "formal_coherence": <0-100>, "distinctiveness": <0-100>,
 "documentary_significance": <0-100>, "conventional_beauty": <0-100>,
 "artistic_candidate": <true|false: worth a human's attention, NOT "this is art">,
 "artistic_confidence": <0-100: how sure you are of the above>,
 "artistic_reasoning": "<max 40 words, concrete and visible>",
 "artistic_strengths": ["<short, concrete>"],
 "artistic_weaknesses": ["<short, concrete>"],
 "intent_reading": {"<defect>": "deliberate|accidental|cannot_tell"},
 "uncertainty": <0-100>,
 "series_role": "<one of the roles or none>",
 "portrait": {
   "face_count": <int>,
   "primary_face_visible": <true|false>,
   "primary_face_area_ratio": <0.0-1.0 of the frame>,
   "face_sharpness": <0-100>,
   "eyes_state": "OPEN|CLOSED|SQUINTING|PARTIALLY_CLOSED|UNCLEAR|NOT_APPLICABLE",
   "expression": "GOOD|NEUTRAL|AWKWARD|GRIMACE|BLINK|MID_SPEECH|UNCLEAR|NOT_APPLICABLE",
   "expression_quality": <0-100>,
   "pose_quality": <0-100>,
   "face_occlusion": <0-100>,
   "blink_probability": <0-100>,
   "grimace_probability": <0-100>,
   "portrait_publishability": <0-100>,
   "expression_confidence": <0-100>,
   "portrait_reasoning": "<max 25 words>",
   "portrait_blockers": ["<short>"]
 }}

Omit the `portrait` object entirely when there is no face. Every numeric field is
required when you include it.

These are absolute ratings, not ranks: two frames may score identically. If you
cannot tell, say so through `uncertainty` and the confidence fields rather than by
inventing a number you do not believe."""


STAGE3_MAX_OUTPUT_TOKENS_PER_FRAME = 220


def stage3_user_content(frames: list[dict]) -> list[dict]:
    """The artistic pass. Deliberately carries no technical measurements.

    Stage 2 is given the measured clipping so it cannot hallucinate exposure
    problems. This stage is given nothing of the sort, because a model told a
    frame is two stops under will explain why that is a fault -- and whether it
    is a fault is the exact question being asked.

    Frames with a face carry extra crops. Each one is labelled, because an
    unlabelled sequence of three similar images is read as three photographs.
    """
    content: list[dict] = [
        {
            "type": "input_text",
            "text": (
                f"Read these {len(frames)} photographs. Return {len(frames)} JSON objects "
                "in the order given, one per frame."
            ),
        }
    ]
    for i, frame in enumerate(frames, start=1):
        views = frame.get("views") or [("full frame", frame.get("encoded", ""))]
        label = f"Frame {i}"
        if len(views) > 1:
            label += f" -- {len(views)} views of the SAME photograph, judge them as one"
        content.append({"type": "input_text", "text": f"{label}:"})
        for view_name, encoded in views:
            if len(views) > 1:
                content.append({"type": "input_text", "text": f"  ({view_name})"})
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{encoded}",
                    "detail": "high",
                }
            )
    return content


ARTISTIC_KEYS = (
    "emotional_resonance",
    "visual_tension",
    "narrative_openness",
    "moment_specificity",
    "formal_coherence",
    "distinctiveness",
    "documentary_significance",
    "conventional_beauty",
)


# Words that signal the model reached for a textbook or a thesaurus instead of
# looking. Used by a test, and available to callers that want to reject a reply.
FORMALIST_PHRASES = (
    "rule of thirds",
    "golden ratio",
    "leading lines",
    "perfectly balanced",
    "well composed",
    "technically flawless",
    "masterful",
    "iconic",
    "genius",
    "breathtaking",
)


def reads_like_a_textbook(note: str) -> bool:
    """Whether a note argues from convention rather than from what is visible."""
    lowered = str(note or "").lower()
    return any(phrase in lowered for phrase in FORMALIST_PHRASES)
