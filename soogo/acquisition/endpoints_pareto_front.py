"""Endpoint Pareto front acquisition function for multi-objective optimization."""

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from pymoo.optimize import minimize as pymoo_minimize

from soogo.acquisition.base import Acquisition
from soogo.acquisition.maximize_distance import MaximizeDistance
from soogo.model.base import Surrogate
from soogo.problem import PymooProblem


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
        endpoints = np.empty((objdim, dim))
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
            assert res.X is not None
            for j in range(dim):
                endpoints[i, j] = res.X[j]

        # Create KDTree with the already evaluated points
        tree = KDTree(surrogateModel.X)
        atol = self.tol(bounds)

        # Discard points that are too close to previously sampled points.
        distNeighbor = tree.query(endpoints)[0]
        endpoints = endpoints[distNeighbor >= atol, :]

        # Discard points that are too close to eachother
        if len(endpoints) > 0:
            selectedIdx = [0]
            for i in range(1, len(endpoints)):
                if (
                    cdist(
                        endpoints[i, :].reshape(1, -1),
                        endpoints[selectedIdx, :],
                    ).min()
                    >= atol
                ):
                    selectedIdx.append(i)
            endpoints = endpoints[selectedIdx, :]

        # Should all points be discarded, which may happen if the minima of
        # the surrogate surfaces do not change between iterations, we
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of sample points
        if endpoints.size == 0:
            maximizeDistance = MaximizeDistance(rtol=self.rtol)
            endpoints = maximizeDistance.optimize(surrogateModel, bounds, n=1)

            assert len(endpoints) == 1

        # Return a maximum of n points
        return endpoints[:n, :]
