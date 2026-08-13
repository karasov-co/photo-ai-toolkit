"""A personal ranking fitted from comparisons, and a hard limit on its authority.

Bradley-Terry over weighted pairwise wins, reusing the fitter already in
`aggregate.py`. Comparisons rather than absolute ratings, because that is the
form the judgement is actually stable in.

The important part is not the model. It is `PersonalModel.may_decide`, which is
false far more often than it is true. A preference model fitted on one
photographer's choices is, by construction, a model of what that photographer
has already understood. Letting it act outside the region it was fitted in is
how a tool starts quietly removing the kind of picture its owner has not yet
learned to like.

So it abstains on anything it has not seen enough of: an unfamiliar genre, an
unfamiliar camera, too few decisions overall, or a prediction too close to the
threshold to be meaningful.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from aggregate import bradley_terry
from preference_store import PreferenceStore

logger = logging.getLogger(__name__)

# Below this the model has seen too little to say anything about anybody.
MIN_DECISIONS_TO_RANK = 50
# Below this it may rank, but may not influence any filesystem decision.
MIN_DECISIONS_TO_DECIDE = 1000
# A genre needs its own evidence; being good at landscapes says nothing about
# how this person edits portraits.
MIN_DECISIONS_PER_GENRE = 40
# Predictions inside this band of the threshold are indistinguishable from a
# coin toss and are abstained on.
DECISION_MARGIN = 0.15


@dataclass
class Prediction:
    asset_id: str
    probability: float = 0.5
    abstained: bool = True
    reason: str = ""
    in_distribution: bool = True

    @property
    def keeps(self) -> bool:
        return self.probability >= 0.5


@dataclass
class PersonalModel:
    strengths: dict[str, float] = field(default_factory=dict)
    decisions: int = 0
    genres: dict[str, int] = field(default_factory=dict)
    cameras: set[str] = field(default_factory=set)

    @property
    def can_rank(self) -> bool:
        return self.decisions >= MIN_DECISIONS_TO_RANK

    @property
    def can_decide(self) -> bool:
        """Whether this model may influence a move. Deliberately hard to reach."""
        return self.decisions >= MIN_DECISIONS_TO_DECIDE

    def knows_genre(self, genre: str) -> bool:
        return self.genres.get(genre, 0) >= MIN_DECISIONS_PER_GENRE

    def knows_camera(self, camera: str) -> bool:
        return not camera or camera in self.cameras

    def score(self, asset_id: str) -> float | None:
        strength = self.strengths.get(asset_id)
        return None if strength is None else round(strength, 5)

    def probability_kept(self, asset_id: str) -> float:
        """Logistic on the fitted strength, against the population's own centre.

        Bradley-Terry strengths are identified only up to a common factor:
        multiply every strength by three and the model describes exactly the
        same preferences. So an absolute threshold on one strength is
        meaningless -- 0.5 and DECISION_MARGIN were drifting with whatever
        scale the fit happened to land on, which made "confident" mean
        something different after every refit.

        Normalising to a mean log-strength of zero fixes the scale to the
        population: 0.5 is now "average for this photographer", which is what
        the threshold was always meant to say.
        """
        strength = self.strengths.get(asset_id)
        if strength is None:
            return 0.5
        centred = math.log(max(strength, 1e-6)) - self._mean_log_strength()
        return round(1.0 / (1.0 + math.exp(-centred)), 5)

    def _mean_log_strength(self) -> float:
        """The scale the fit happened to produce, so it can be divided out."""
        values = [math.log(max(v, 1e-6)) for v in self.strengths.values()]
        return sum(values) / len(values) if values else 0.0

    def predict(self, asset_id: str, *, genre: str = "", camera: str = "") -> Prediction:
        """Predict, or say why not. Abstaining is the common case by design."""
        if not self.can_rank:
            return Prediction(asset_id, reason=f"only {self.decisions} decisions recorded")

        in_distribution = self.knows_genre(genre) and self.knows_camera(camera)
        if not in_distribution:
            missing = "genre" if not self.knows_genre(genre) else "camera"
            return Prediction(
                asset_id,
                reason=f"unfamiliar {missing}: nothing comparable has been judged yet",
                in_distribution=False,
            )

        probability = self.probability_kept(asset_id)
        if abs(probability - 0.5) < DECISION_MARGIN:
            return Prediction(
                asset_id,
                probability=probability,
                reason=f"probability {probability:.2f} is too close to a coin toss",
            )
        return Prediction(asset_id, probability=probability, abstained=False)


def fit(store: PreferenceStore) -> PersonalModel:
    """Weighted Bradley-Terry over the recorded comparisons."""
    decisions = store.all()
    pairs: list[tuple[str, str]] = []
    for decision in decisions:
        if not decision.is_pairwise:
            continue
        # Weight is expressed as repetition, which is what the MM fitter reads.
        for _ in range(max(1, int(round(decision.weight)))):
            pairs.append((decision.winner, decision.loser))

    genres: dict[str, int] = {}
    for decision in decisions:
        if decision.genre:
            genres[decision.genre] = genres.get(decision.genre, 0) + 1

    return PersonalModel(
        strengths=bradley_terry(pairs) if pairs else {},
        decisions=len(decisions),
        genres=genres,
        cameras={d.camera for d in decisions if d.camera},
    )


def disagreement(personal: Prediction, curator_says_keep: bool) -> bool:
    """Whether the personal model and the general prior point opposite ways.

    Any disagreement routes to review. The personal model is not more right than
    the curatorial prior; it is differently informed, and where they conflict
    the honest answer is that nobody knows.
    """
    if personal.abstained:
        return False
    return personal.keeps != curator_says_keep
