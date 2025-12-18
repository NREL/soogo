"""Maximize distance acquisition function."""

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

__authors__ = ["Weslley S. Pereira", "Byron Selvage"]

import numpy as np
from typing import Optional

from pymoo.optimize import minimize as pymoo_minimize

from .base import Acquisition
from ..model import Surrogate
from ..integrations.pymoo import PymooProblem
from .utils import FarEnoughSampleFilter


class MaximizeDistance(Acquisition):
    """
    Maximizing distance acquisition function as described in [#]_.

    This acquisition function is used to find new sample points that maximize
    the minimum distance to previously sampled points.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        points: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Acquire 1 point that maximize the minimum distance to previously
        sampled points.

        :param surrogateModel: The surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param points: Points to consider for distance maximization. If None,
            use all previously sampled points in the surrogate model.
        :return: Array of acquired points that maximize minimum distance.
        """
        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        if points is None:
            currentPoints = surrogateModel.X.copy()
        else:
            currentPoints = points.copy()

        filter = FarEnoughSampleFilter(currentPoints, self.tol(bounds))

        problem = PymooProblem(
            lambda x: -filter.tree.query(x)[0],
            bounds,
            iindex,
        )
        res = pymoo_minimize(
            problem,
            optimizer,
            seed=surrogateModel.ntrain + 1,
            verbose=False,
        )
        if res.X is not None:
            return filter(np.array([[res.X[j] for j in range(len(bounds))]]))
        else:
            return np.empty((0, len(bounds)))
