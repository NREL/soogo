"""Transition search acquisition function for hidden constraints."""

# Copyright (c) 2025 Alliance for Energy Innovation, LLC

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Byron Selvage"]

import numpy as np
from scipy.spatial.distance import cdist

from .base import Acquisition
from .utils import select_weighted_candidates
from ..model import Surrogate
from ..sampling import NormalSampler, Sampler


class TransitionSearch(Acquisition):
    """
    Transition search acquisition function as described in [#]_.

    This acquisition function is used to find new sample points by perturbing
    the current best sample point and uniformly selecting points from the
    domain. The scoreWeight parameter can be used to control the transition
    from local to global search. A scoreWeight close to 1.0 will favor
    the predicted function value (local search), while a scoreWeight close to
    0.0 will favor the distance to previously sampled points (global search).

    The evaluability of candidate points is predicted using the candidate
    surrogate model. If the evaluability probaility of a candidate point
    is below a threshold, the point is discarded. If no candidate points
    remain after this filtering, all candidate points are considered
    evaluable.

    The candidate points are scored using a weighted value of their predicted
    function value and the distance to previously sampled points. The candidate
    with the best total score is selected as the new sample point.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def generate_candidates(
        self,
        surrogateModel: Surrogate,
        bounds,
        nCand: int,
        xbest: np.ndarray = None,
    ) -> np.ndarray:
        """
        Generate candidate points by perturbing the current best point and
        uniformly sampling from the bounds. A total of 2* nCand candidate
        points are generated, where the first nCand points are perturbations of
        the current best point and the second nCand points are uniformly sampled
        from the bounds.

        :param surrogateModel: Surrogate model for the objective function.
        :param bounds: List with the limits [x_min, x_max] of each direction.
        :param nCand: Number of candidate points to be generated.
        :param xbest: Current best point. If None, use the best point from the
            surrogate model.
        :return: Array of candidate points.
        """
        dim = len(bounds)
        bounds = np.asarray(bounds)

        if xbest is None:
            best_idx = np.argmin(surrogateModel.Y)
            xbest = surrogateModel.X[best_idx]

        ## Create nCand points by perturbing the current best point
        # Set perturbation probability
        if dim <= 10:
            perturbProbability = 1.0
        else:
            perturbProbability = np.random.uniform(0, 1)

        # Generate perturbation candidates
        perturbationSampler = NormalSampler(n=nCand, sigma=0.02)
        perturbationCandidates = perturbationSampler.get_dds_sample(
            bounds, mu=xbest, probability=perturbProbability
        )

        ## Generate nCand points uniformly from the bounds
        uniformSampler = Sampler(nCand)
        uniformCandidates = uniformSampler.get_uniform_sample(bounds)

        # Combine perturbation and uniform candidates
        candidates = np.vstack((perturbationCandidates, uniformCandidates))
        return candidates

    def select_candidates(
        self,
        surrogateModel: Surrogate,
        candidates: np.ndarray,
        bounds,
        n: int = 1,
        scoreWeight: float = 0.5,
        evaluabilityThreshold: float = 0.25,
        evaluabilitySurrogate: Surrogate = None,
    ) -> np.ndarray:
        """
        Select the best candidate points based on the predicted function
        value and distance to previously sampled points. The candidates are
        scored using a weighted score that combines the predicted function
        value and the distance to previously sampled points.

        :param surrogateModel: Surrogate model for the objective function.
        :param candidates: Array of candidate points.
        :param bounds: List with the limits [x_min, x_max] of each direction.
        :param n: Number of best candidates to return.
        :param scoreWeight: Weight for the predicted function value and distance
            scores in the total score.
        :param evaluabilityThreshold: Threshold for the evaluability
            probability. Candidates with evaluability probability below this
            threshold are discarded.
        :param evaluabilitySurrogate: Surrogate model for the evaluability
            probability of the candidate points. If provided, candidates with
            evaluability probability below the threshold are discarded.
        :return: The n best candidate points.
        """
        # Filter candidates based on evaluability
        if evaluabilitySurrogate is not None:
            evaluability = evaluabilitySurrogate(candidates)
            # Keep candidates above the evaluability threshold
            if len(candidates[evaluability > evaluabilityThreshold]) > 0:
                candidates = candidates[evaluability > evaluabilityThreshold]
            else:
                # If no candidates are above the evaluability threshold, keep
                # candidates with positive evaluability
                candidates = candidates[evaluability > 0]

        # Get the predicted function values for the candidates
        predictedValues = surrogateModel(candidates)

        # Compute distances to previously evaluated points
        if evaluabilitySurrogate is not None:
            distx = cdist(candidates, evaluabilitySurrogate.X)
        else:
            distx = cdist(candidates, surrogateModel.X)

        # Select points with weighted sum
        weightpattern = [scoreWeight]
        xselected, _ = select_weighted_candidates(
            candidates,
            distx,
            predictedValues,
            n,
            self.tol(bounds),
            weightpattern,
        )

        return xselected

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        *,
        evaluabilitySurrogate: Surrogate = None,
        evaluabilityThreshold: float = 0.25,
        scoreWeight: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        """
        This acquisition function generates candidate points by perturbing the
        current best point and uniformly sampling from the bounds. It then
        selects the best candidate point based on a weighted score that combines
        the predicted function value and the distance to previously sampled
        points.

        :param surrogateModel: Surrogate model for the objective function.
        :param bounds: List with the limits [x_min, x_max] of each direction.
        :param n: Number of points to be acquired.
        :param evaluabilitySurrogate: Surrogate model for the evaluability
            probability of the candidate points. If provided, candidates with
            evaluability probability below the threshold are discarded.
        :param evaluabilityThreshold: Threshold for the evaluability
            probability.
        :param scoreWeight: Weight for the predicted function value and distance
            scores in the total score. The total score is computed as:
            `scoreWeight * valueScore + (1 - scoreWeight) * distanceScore`.
        :param kwargs: Additional keyword arguments (unused).
        """
        # Set Ncand = 500*dim
        dim = len(bounds)
        nCand = 500 * dim

        bounds = np.asarray(bounds)

        # Get current best point
        best_idx = np.argmin(surrogateModel.Y)
        xbest = surrogateModel.X[best_idx]

        # Generate candidate points
        candidates = self.generate_candidates(
            surrogateModel, bounds, nCand, xbest=xbest
        )

        # Select n best candidates
        bestCandidates = self.select_candidates(
            surrogateModel,
            candidates,
            bounds,
            n=n,
            scoreWeight=scoreWeight,
            evaluabilityThreshold=evaluabilityThreshold,
            evaluabilitySurrogate=evaluabilitySurrogate,
        )

        return bestCandidates
