"""Endpoint Pareto front acquisition function for multi-objective optimization."""

import numpy as np

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

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points at most, where k <= n.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = surrogateModel.ntarget

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Find endpoints of the Pareto front
        endpoints = np.empty((0, dim))
        for i in range(objdim):
            minimumPointProblem = PymooProblem(
                lambda x: surrogateModel(x, i=i), bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem,
                optimizer,
                seed=surrogateModel.ntrain,
                verbose=False,
            )
            if res.X is not None:
                endpoints = np.vstack((endpoints, res.X.reshape(1, -1)))

        return FarEnoughSampleFilter(surrogateModel.X, self.tol(bounds))(
            endpoints
        )
