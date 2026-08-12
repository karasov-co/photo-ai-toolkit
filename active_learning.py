"""Which question to ask, so that ten minutes beats labelling the whole archive.

Asking about random frames wastes the only resource that matters, which is the
photographer's patience. The useful questions are the ones whose answer changes
the model most: pairs it cannot separate, frames where two models disagree,
genres it has never seen, defects it could not classify.

The answer format is four buttons, not a 1-10 rating. A rating is slow, drifts
between sessions, and is the wrong shape for the fitter anyway.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Question:
    kind: str
    prompt: str
    options: list[str]
    assets: list[str] = field(default_factory=list)
    value: float = 0.0
    why: str = ""


KEEP = ["A", "B", "both", "neither"]
STRONGER = ["A", "B", "the same", "do not know"]
INTENT = ["deliberate", "accidental", "cannot tell"]
ROLE = ["key frame", "transition", "context", "not needed"]
VARIANT = ["original", "faithful", "expressive"]


def propose(records: list, model, *, limit: int = 12) -> list[Question]:
    """Rank candidate questions by how much the answer would teach."""
    questions: list[Question] = []

    for record in records:
        genre = getattr(record, "genre", "") or "other"
        asset_id = getattr(record, "asset_id", "")
        artistic = getattr(record, "artistic", {}) or {}

        # An unclassifiable defect: the single most useful thing to be told,
        # because no measurement can settle it and it gates the rescue path.
        for signal in artistic.get("intent_signals") or []:
            if signal.get("verdict") == "cannot_tell":
                questions.append(
                    Question(
                        kind="intent",
                        prompt=f"{record.filename}: is the {signal.get('defect')} deliberate?",
                        options=INTENT,
                        assets=[asset_id],
                        value=1.0,
                        why="no measurement can decide this, and it gates the rescue path",
                    )
                )

        # An unseen genre: the model abstains here and will keep abstaining.
        if not model.knows_genre(genre):
            questions.append(
                Question(
                    kind="keep",
                    prompt=f"{record.filename}: keep this?",
                    options=KEEP,
                    assets=[asset_id],
                    value=0.9,
                    why=f"no decisions recorded for {genre} yet",
                )
            )

        # Near the decision boundary: where one answer moves the threshold most.
        prediction = model.predict(asset_id, genre=genre)
        if not prediction.abstained and abs(prediction.probability - 0.5) < 0.2:
            questions.append(
                Question(
                    kind="keep",
                    prompt=f"{record.filename}: keep this?",
                    options=KEEP,
                    assets=[asset_id],
                    value=0.85,
                    why=f"the model is at {prediction.probability:.2f} and cannot decide",
                )
            )

    questions.extend(_burst_pairs(records))
    questions.sort(key=lambda q: -q.value)
    return _deduplicate(questions)[:limit]


def _burst_pairs(records: list) -> list[Question]:
    """Two frames of one moment: the cleanest pairwise signal available."""
    by_cluster: dict[str, list] = {}
    for record in records:
        cluster = getattr(record, "cluster_id", "")
        if cluster and getattr(record, "cluster_size", 1) > 1:
            by_cluster.setdefault(cluster, []).append(record)

    questions: list[Question] = []
    for members in by_cluster.values():
        if len(members) < 2:
            continue
        a, b = members[0], members[1]
        margin = abs(
            (a.scores or {}).get("current_quality", 0) - (b.scores or {}).get("current_quality", 0)
        )
        questions.append(
            Question(
                kind="pairwise_emotion",
                prompt=f"{a.filename} or {b.filename}: which is stronger?",
                options=STRONGER,
                assets=[a.asset_id, b.asset_id],
                # A near-tie is worth more: the measurements already separate
                # the rest, and only a person can separate these.
                value=1.0 if margin < 5 else 0.5,
                why=f"technically {margin:.0f} points apart; only a person can separate them",
            )
        )
    return questions


def _deduplicate(questions: list[Question]) -> list[Question]:
    seen: set[tuple] = set()
    out: list[Question] = []
    for question in questions:
        key = (question.kind, tuple(sorted(question.assets)))
        if key in seen:
            continue
        seen.add(key)
        out.append(question)
    return out


def format_session(questions: list[Question]) -> str:
    """Five to ten minutes after a shoot, printed."""
    if not questions:
        return "Nothing worth asking: the model is not uncertain anywhere useful."
    lines = [f"{len(questions)} question(s), most informative first.", ""]
    for i, question in enumerate(questions, start=1):
        lines.append(f"{i}. {question.prompt}")
        lines.append(f"   {' / '.join(question.options)}")
        lines.append(f"   why: {question.why}")
        lines.append("")
    return "\n".join(lines)
