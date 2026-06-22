"""Maximize distance acquisition function."""

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

__authors__ = ["Weslley S. Pereira", "Byron Selvage"]

import numpy as np
from typing import Optional
import functools

from pymoo.optimize import minimize as pymoo_minimize

from .base import Acquisition
from ..model import Surrogate
from ..integrations.pymoo import PymooProblem
from .utils import FarEnoughSampleFilter


def _negative_distance(tree, x):
    """Compute negative distance to nearest neighbor.

    :param tree: KDTree of previously sampled points.
    :param x: Point to evaluate.
    :return: Negative distance to nearest neighbor.
    """
    return -tree.query(x)[0]


class MaximizeDistance(Acquisition):
    """
    Maximizing distance acquisition function as described in [#]_.

    This acquisition function is used to find new sample points that maximize
    the minimum distance to previously sampled points.

    :param seed: Seed for random number generator.

    .. attribute:: rng

        Random number generator.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, seed=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        constr=None,
        exclusion_set: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Acquire 1 point that maximize the minimum distance to previously
        sampled points.

        :param surrogateModel: The surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param constr: Constraint function for candidate points.
            Feasible candidates should satisfy constr(x) <= 0.
        :param exclusion_set: Known points, if any, in addition to the ones
            used to train the surrogate.
        :return: Array of acquired points that maximize minimum distance.
        """
        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer
        n_ieq_constr = (
            (constr(surrogateModel.X[0:1])).size if constr is not None else 0
        )

        # Report unused kwargs
        super().report_unused_optimize_kwargs(kwargs)

        exclusion_set = (
            np.vstack((exclusion_set, surrogateModel.X))
            if exclusion_set is not None
            else surrogateModel.X
        )
        filter = FarEnoughSampleFilter(exclusion_set, self.tol(bounds))

        problem = PymooProblem(
            functools.partial(_negative_distance, filter.tree),
            bounds,
            iindex,
            gfunc=constr,
            n_ieq_constr=n_ieq_constr,
        )
        res = pymoo_minimize(
            problem,
            optimizer,
            seed=self.rng.integers(np.iinfo(np.int32).max).item(),
            verbose=False,
        )
        if res.X is not None:
            return filter(np.array([[res.X[j] for j in range(len(bounds))]]))
        else:
            return np.empty((0, len(bounds)))
