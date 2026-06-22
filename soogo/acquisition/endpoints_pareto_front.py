"""Endpoint Pareto front acquisition function for multi-objective
optimization.
"""

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
import functools

from pymoo.optimize import minimize as pymoo_minimize

from .base import Acquisition
from ..model import Surrogate
from ..integrations.pymoo import PymooProblem
from .utils import FarEnoughSampleFilter


class EndPointsParetoFront(Acquisition):
    """Obtain endpoints of the Pareto front as described in [#]_.

    For each component i in the target space, this algorithm solves a cheap
    auxiliary optimization problem to minimize the i-th component of the
    trained surrogate model. Points that are too close to each other and to
    training sample points are eliminated. If all points were to be eliminated,
    consider the whole variable domain and sample at the point that maximizes
    the minimum distance to training sample points.

    :param seed: Seed for random number generator.

    .. attribute:: rng

        Random number generator.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, seed=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        exclusion_set: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points at most, where k <= objdim.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param exclusion_set: Known points, if any, in addition to the ones
            used to train the surrogate.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = surrogateModel.ntarget

        # Report unused kwargs
        super().report_unused_optimize_kwargs(kwargs)

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Find endpoints of the Pareto front
        endpoints = np.empty((0, dim))
        for i in range(objdim):
            minimumPointProblem = PymooProblem(
                functools.partial(surrogateModel, i=i), bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem,
                optimizer,
                seed=self.rng.integers(np.iinfo(np.int32).max).item(),
                verbose=False,
            )
            if res.X is not None:
                endpoints = np.vstack((endpoints, res.X.reshape(1, -1)))

        exclusion_set = (
            np.vstack((exclusion_set, surrogateModel.X))
            if exclusion_set is not None
            else surrogateModel.X
        )
        return FarEnoughSampleFilter(exclusion_set, self.tol(bounds))(
            endpoints
        )
