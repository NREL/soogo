"""Pareto front acquisition functions for multi-objective optimization."""

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
from scipy.spatial import KDTree
from typing import Optional
import functools

from pymoo.core.initialization import Initialization
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.population import Population

from .base import Acquisition
from ..model import LinearRadialBasisFunction, RbfModel, Surrogate
from ..integrations.pymoo import PymooProblem
from ..utils import find_pareto_front
from .utils import FarEnoughSampleFilter


def _distance_to_target(surrogate, target_value, x):
    """Compute distance from surrogate prediction to target.

    :param surrogate: Surrogate model.
    :param target_value: Target value to compare against.
    :param x: Point to evaluate.
    :return: Distance from surrogate prediction at x to target_value.
    """
    return np.absolute(surrogate(x) - target_value)


class ParetoFront(Acquisition):
    """Obtain sample points that fill gaps in the Pareto front from [#]_.

    The algorithm proceeds as follows to find each new point:

    1. Find a target value :math:`\\tau` that should fill a gap in the Pareto
       front. Make sure to use a target value that wasn't used before.
    2. Solve a multi-objective optimization problem that minimizes
       :math:`\\|s_i(x)-\\tau\\|` for all :math:`x` in the search space, where
       :math:`s_i(x)` is the i-th target value predicted by the surrogate for
       :math:`x`.
    3. If a Pareto-optimal solution was found for the problem above, chooses the
       point that minimizes the L1 distance to :math:`\\tau` to be part of the
       new sample.

    :param oldTV: Old target values to be avoided in the acquisition.
        Copied to :attr:`oldTV`.
    :param seed: Seed for random number generator.

    .. attribute:: oldTV

        Old target values to be avoided in the acquisition of step 1.

    .. attribute:: rng

        Random number generator.

    .. attribute:: so_optimizer

        Single-objective optimizer to be used in step 1.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, oldTV=(), seed=None, **kwargs) -> None:
        self.oldTV = np.array(oldTV)
        super().__init__(multi_objective=True, **kwargs)
        self.rng = np.random.default_rng(seed)
        self.so_optimizer = self.default_optimizer(False)

    def pareto_front_target(self, paretoFront: np.ndarray) -> np.ndarray:
        """Find a target value that should fill a gap in the Pareto front.

        As suggested by Mueller (2017), the algorithm fits a linear RBF
        model with the points in the Pareto front. This will represent the
        (d-1)-dimensional Pareto front surface. Then, the algorithm searches the
        a value in the surface that maximizes the distances to previously
        selected target values and to the training points of the RBF model. This
        value is projected in the d-dimensional space to obtain :math:`\\tau`.

        :param paretoFront: Pareto front in the objective space.
        :return: The target value :math:`\\tau`.
        """
        objdim = paretoFront.shape[1]
        if objdim <= 1:
            return paretoFront[0]

        # Discard duplicated points in the Pareto front
        paretoFront = np.unique(paretoFront, axis=0)

        # Create a surrogate model for the Pareto front in the objective space
        paretoModel = RbfModel(LinearRadialBasisFunction())
        k = self.rng.choice(objdim)
        paretoModel.update(
            np.array([paretoFront[:, i] for i in range(objdim) if i != k]).T,
            paretoFront[:, k],
        )
        dim = paretoModel.dim

        # Bounds in the pareto sample
        xParetoLow = np.min(paretoModel.X, axis=0)
        xParetoHigh = np.max(paretoModel.X, axis=0)
        boundsPareto = [(xParetoLow[i], xParetoHigh[i]) for i in range(dim)]

        # Minimum of delta_f maximizes the distance inside the Pareto front
        tree = KDTree(
            np.concatenate(
                (paretoFront, self.oldTV.reshape(-1, objdim)), axis=0
            )
        )

        def delta_f(tau):
            tauk = paretoModel(tau)
            _tau = np.insert(tau, k, tauk, axis=1)
            return -tree.query(_tau)[0]

        # Minimize delta_f
        problem = PymooProblem(
            delta_f,
            boundsPareto,
        )
        res = pymoo_minimize(
            problem,
            self.so_optimizer,
            seed=self.rng.integers(np.iinfo(np.int32).max).item(),
            verbose=False,
        )
        if res.X is not None:
            tauk = paretoModel(res.X)
            tau = np.concatenate((res.X[0:k], tauk, res.X[k:]))
            return tau
        else:
            return np.array([])

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        xbest=None,
        ybest=None,
        exclusion_set: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        Perform n attempts to find n points to fill gaps in the Pareto front.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param xbest: Nondominated set in the objective space. If not
            provided, use the surrogate to compute it.
        :param ybest: Pareto front in the objective space. If not
            provided, use the surrogate to compute it.
        :param exclusion_set: Known points, if any, in addition to the ones
            used to train the surrogate.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = surrogateModel.ntarget

        if surrogateModel.ntarget < 2:
            raise ValueError(
                "The surrogate model must have at least two targets "
                "to perform multi-objective optimization."
            )

        # Report unused kwargs
        super().report_unused_optimize_kwargs(kwargs)

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Compute nondominated set and Pareto front if not provided
        if xbest is None or ybest is None:
            paretoFrontIdx = find_pareto_front(surrogateModel.Y)
            ybest = surrogateModel.Y[paretoFrontIdx]
            xbest = surrogateModel.X[paretoFrontIdx]

        # If the Pareto front has only one point or is empty, there is no
        # way to find a target value.
        if len(ybest) <= 1:
            return np.empty((0, dim))

        exclusion_set = (
            np.vstack((exclusion_set, surrogateModel.X))
            if exclusion_set is not None
            else surrogateModel.X
        )
        filter = FarEnoughSampleFilter(exclusion_set, self.tol(bounds))

        xselected = np.empty((0, dim))
        for _ in range(n):
            # Find a target value tau in the Pareto front
            tau = self.pareto_front_target(np.asarray(ybest))
            if len(tau) == 0:
                break
            self.oldTV = np.concatenate(
                (self.oldTV.reshape(-1, objdim), [tau]), axis=0
            )

            # Use non-dominated points if provided
            if len(xbest) > 0:
                Xinit = (
                    xbest
                    if len(iindex) == 0
                    else np.array(
                        [{i: x[i] for i in range(dim)} for x in xbest]
                    )
                )
                optimizer.initialization = Initialization(
                    Population.new("X", Xinit),
                    repair=optimizer.repair,
                    eliminate_duplicates=optimizer.eliminate_duplicates,
                )

            # Find the Pareto-optimal solution set that minimizes
            # dist(s(x),tau).
            # For discontinuous Pareto fronts in the original problem, such set
            # may not exist, or it may be too far from the target value.
            multiobjTVProblem = PymooProblem(
                functools.partial(_distance_to_target, surrogateModel, tau),
                bounds,
                iindex,
                n_obj=objdim,
            )
            res = pymoo_minimize(
                multiobjTVProblem,
                optimizer,
                seed=self.rng.integers(np.iinfo(np.int32).max).item(),
                verbose=False,
            )

            # If the Pareto-optimal solution set exists, define the sample point
            # that minimizes the L1 distance to the target value
            if res.X is not None:
                # Save X into an array
                newX = np.array([[x[i] for i in range(dim)] for x in res.X])

                # Eliminate points that are too close to previously samples
                # and to each other
                newX = filter(newX)

                # Transform the values of the optimization into a matrix
                sx = surrogateModel(newX)

                # Find the values that are expected to be in the Pareto front
                # of the original optimization problem
                nondominated_idx = find_pareto_front(
                    np.vstack((ybest, sx)), iStart=len(ybest)
                )
                nondominated_idx = [
                    idx - len(ybest)
                    for idx in nondominated_idx
                    if idx >= len(ybest)
                ]

                # Add a point that is expected to be non-dominated
                if len(nondominated_idx) > 0:
                    idx = np.sum(res.F[nondominated_idx], axis=1).argmin()
                    xselected = np.vstack(
                        (xselected, newX[nondominated_idx][idx : idx + 1])
                    )

        return filter(xselected)
