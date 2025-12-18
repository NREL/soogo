"""Minimize multi-objective surrogate acquisition function."""

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

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize as pymoo_minimize

from .base import Acquisition
from ..model import Surrogate
from ..integrations.pymoo import PymooProblem, ListDuplicateElimination


class MinimizeMOSurrogate(Acquisition):
    """Obtain pareto-optimal sample points for the multi-objective surrogate
    model.

    :param optimizer: Continuous multi-objective optimizer. If None, use
        NSGA2 from pymoo.
    :param mi_optimizer: Mixed-integer multi-objective optimizer. If None, use
        MixedVariableGA from pymoo with RankAndCrowding survival strategy.

    """

    def __init__(self, **kwargs) -> None:
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = NSGA2()
        if "mi_optimizer" not in kwargs:
            kwargs["mi_optimizer"] = MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
                survival=RankAndCrowding(),
            )
        super().__init__(**kwargs)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired. If n is zero, use all
            points in the Pareto front.
        :param kwargs: Additional keyword arguments (unused).
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Solve the surrogate multiobjective problem
        multiobjSurrogateProblem = PymooProblem(
            surrogateModel, bounds, iindex, n_obj=surrogateModel.ntarget
        )
        res = pymoo_minimize(
            multiobjSurrogateProblem,
            optimizer,
            seed=surrogateModel.ntrain,
            verbose=False,
        )

        # If the Pareto-optimal solution set exists, randomly select n
        # points from the Pareto front
        if res.X is not None:
            bestCandidates = np.array(
                [[x[i] for i in range(dim)] for x in res.X]
            )

            # Create tolerance based on smallest variable length
            atol = self.tol(bounds)

            # Discard points that are too close to previously sampled points.
            distNeighbor = cdist(bestCandidates, surrogateModel.X).min(axis=1)
            bestCandidates = bestCandidates[distNeighbor >= atol, :]

            # Return if no point was left
            nMax = len(bestCandidates)
            if nMax == 0:
                return np.empty((0, dim))

            # Randomly select points in the Pareto front
            idxs = (
                np.random.choice(nMax, size=min(n, nMax))
                if n > 0
                else np.arange(nMax)
            )
            bestCandidates = bestCandidates[idxs]

            # Discard points that are too close to eachother
            selectedIdx = [0]
            for i in range(1, len(bestCandidates)):
                if (
                    cdist(
                        bestCandidates[i].reshape(1, -1),
                        bestCandidates[selectedIdx],
                    ).min()
                    >= atol
                ):
                    selectedIdx.append(i)
            bestCandidates = bestCandidates[selectedIdx]

            return bestCandidates
        else:
            return np.empty((0, dim))
