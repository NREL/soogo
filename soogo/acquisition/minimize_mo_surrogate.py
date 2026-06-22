"""Minimize multi-objective surrogate acquisition function."""

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
from typing import Optional

from pymoo.optimize import minimize as pymoo_minimize

from .base import Acquisition
from ..model import Surrogate
from ..integrations.pymoo import PymooProblem
from .utils import FarEnoughSampleFilter


class MinimizeMOSurrogate(Acquisition):
    """Obtain pareto-optimal sample points for the multi-objective surrogate
    model.

    :param seed: Seed for random number generator.

    .. attribute:: rng

        Random number generator.

    """

    def __init__(self, seed=None, **kwargs) -> None:
        super().__init__(multi_objective=True, **kwargs)
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        exclusion_set: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :param exclusion_set: Known points, if any, in addition to the ones
            used to train the surrogate.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)

        if surrogateModel.ntarget < 2:
            raise ValueError(
                "The surrogate model must have at least two targets "
                "to perform multi-objective optimization."
            )

        # Report unused kwargs
        super().report_unused_optimize_kwargs(kwargs)

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Solve the surrogate multiobjective problem
        multiobjSurrogateProblem = PymooProblem(
            surrogateModel, bounds, iindex, n_obj=surrogateModel.ntarget
        )
        res = pymoo_minimize(
            multiobjSurrogateProblem,
            optimizer,
            seed=self.rng.integers(np.iinfo(np.int32).max).item(),
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

            # Discard points that are too close to previously sampled points and
            # to each other.
            exclusion_set = (
                np.vstack((exclusion_set, surrogateModel.X))
                if exclusion_set is not None
                else surrogateModel.X
            )
            bestCandidates = FarEnoughSampleFilter(exclusion_set, atol)(
                bestCandidates
            )

            # Return if no point was left
            nMax = len(bestCandidates)
            if nMax == 0:
                return np.empty((0, dim))

            # Randomly select points in the Pareto front
            idxs = self.rng.choice(nMax, size=min(n, nMax))
            bestCandidates = bestCandidates[idxs]

            return bestCandidates
        else:
            return np.empty((0, dim))
