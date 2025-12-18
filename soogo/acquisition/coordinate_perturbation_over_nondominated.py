"""Coordinate perturbation acquisition over nondominated points."""

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

__authors__ = ["Weslley S. Pereira"]

import numpy as np
from scipy.spatial.distance import cdist

from .base import Acquisition
from .weighted_acquisition import WeightedAcquisition
from ..model import Surrogate
from ..sampling import NormalSampler
from ..utils import find_pareto_front


class CoordinatePerturbationOverNondominated(Acquisition):
    """Coordinate perturbation acquisition function over the nondominated set.

    This acquisition method was proposed in [#]_. It perturbs locally each of
    the non-dominated sample points to find new sample points. The perturbation
    is performed by :attr:`acquisitionFunc`.

    :param acquisitionFunc: Weighted acquisition function with a normal sampler.
        Stored in :attr:`acquisitionFunc`.

    .. attribute:: acquisitionFunc

        Weighted acquisition function with a normal sampler.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, acquisitionFunc: WeightedAcquisition, **kwargs) -> None:
        self.acquisitionFunc = acquisitionFunc
        assert isinstance(self.acquisitionFunc.sampler, NormalSampler)
        super().__init__(**kwargs)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        *,
        nondominated=(),
        paretoFront=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :param nondominated: Nondominated set in the objective space.
        :param paretoFront: Pareto front in the objective space.
        :param kwargs: Additional keyword arguments passed to the acquisition
            function.
        """
        dim = len(bounds)
        atol = self.acquisitionFunc.tol(bounds)

        # Find a collection of points that are close to the Pareto front
        bestCandidates = np.empty((0, dim))
        for ndpoint in nondominated:
            x = self.acquisitionFunc.optimize(
                surrogateModel, bounds, 1, xbest=ndpoint
            )
            # Choose points that are not too close to previously selected points
            if bestCandidates.size == 0:
                if x.size > 0:
                    bestCandidates = x.reshape(1, -1)
            else:
                distNeighborOfx = cdist(x, bestCandidates).min()
                if distNeighborOfx >= atol:
                    bestCandidates = np.concatenate(
                        (bestCandidates, x), axis=0
                    )

        # Return if no point was found
        if bestCandidates.size == 0:
            return bestCandidates

        # Eliminate points predicted to be dominated
        fnondominatedAndBestCandidates = np.concatenate(
            (paretoFront, surrogateModel(bestCandidates)), axis=0
        )
        idxPredictedPareto = find_pareto_front(
            fnondominatedAndBestCandidates,
            iStart=len(nondominated),
        )
        idxPredictedBest = [
            i - len(nondominated)
            for i in idxPredictedPareto
            if i >= len(nondominated)
        ]
        bestCandidates = bestCandidates[idxPredictedBest, :]

        return bestCandidates[:n, :]
