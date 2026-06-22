"""Coordinate perturbation acquisition over nondominated points."""

# Copyright (c) 2024-2026 Alliance for Energy Innovation, LLC

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

from .base import Acquisition
from .utils import FarEnoughSampleFilter
from .coordinate_perturbation import CoordinatePerturbation
from ..model import Surrogate
from ..utils import find_pareto_front


class CoordinatePerturbationOverNondominated(Acquisition):
    """Coordinate perturbation acquisition function over the nondominated set.

    This acquisition method was proposed in [#]_. It perturbs locally each of
    the non-dominated sample points to find new sample points. The perturbation
    is performed by :attr:`acquisitionFunc`.

    :param acquisitionFunc: A :class:`.CoordinatePerturbation` instance used to
        perform local perturbations around nondominated points. Stored in
        :attr:`acquisitionFunc`.

    .. attribute:: acquisitionFunc

        Coordinate-perturbation acquisition used for local exploration of the
        nondominated set.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(
        self, acquisitionFunc: CoordinatePerturbation, seed=None, **kwargs
    ) -> None:
        self.acquisitionFunc = acquisitionFunc
        self.rng = np.random.default_rng(seed)
        super().__init__(**kwargs)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        xbest=None,
        ybest=None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :param xbest: Nondominated set in the objective space. If not
            provided, use the surrogate to compute it.
        :param ybest: Pareto front in the objective space. If not
            provided, use the surrogate to compute it.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        atol = self.acquisitionFunc.tol(bounds)

        # Compute nondominated set and Pareto front if not provided
        if xbest is None or ybest is None:
            paretoFrontIdx = find_pareto_front(surrogateModel.Y)
            ybest = surrogateModel.Y[paretoFrontIdx]
            xbest = surrogateModel.X[paretoFrontIdx]

        # Find a collection of points that are close to the Pareto front
        bestCandidates = np.empty((len(xbest), dim))
        for i, ndpoint in enumerate(xbest):
            bestCandidates[i] = self.acquisitionFunc.optimize(
                surrogateModel, bounds, n=1, xbest=ndpoint, **kwargs
            )

        # Eliminate points predicted to be dominated
        fnondominatedAndBestCandidates = np.concatenate(
            (ybest, surrogateModel(bestCandidates)), axis=0
        )
        idxPredictedPareto = find_pareto_front(
            fnondominatedAndBestCandidates,
            iStart=len(xbest),
        )
        idxPredictedBest = [
            i - len(xbest) for i in idxPredictedPareto if i >= len(xbest)
        ]
        bestCandidates = bestCandidates[idxPredictedBest, :]

        # Eliminate points that are too close to one another. No need to use
        # known points since the CoordinatePerturbation acquisition already
        # guarantees points are far enough from them
        bestCandidates = FarEnoughSampleFilter(np.empty((0, dim)), atol)(
            bestCandidates
        )

        # Scramble candidates
        bestCandidates = bestCandidates[
            self.rng.permutation(len(bestCandidates)), :
        ]

        # Return at most n points
        return bestCandidates[: min(n, len(bestCandidates))]
