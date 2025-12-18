"""Pareto front acquisition functions for multi-objective optimization."""

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
from scipy.spatial import KDTree
from scipy.optimize import differential_evolution

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.initialization import Initialization
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.population import Population

from .base import Acquisition
from ..model import LinearRadialBasisFunction, RbfModel, Surrogate
from ..integrations.pymoo import PymooProblem, ListDuplicateElimination
from ..utils import find_pareto_front
from .utils import FarEnoughSampleFilter


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

    :param optimizer: Continuous multi-objective optimizer. If None, use
        NSGA2 from pymoo.
    :param mi_optimizer: Mixed-integer multi-objective optimizer. If None, use
        MixedVariableGA from pymoo with RankAndCrowding survival strategy.
    :param oldTV: Old target values to be avoided in the acquisition.
        Copied to :attr:`oldTV`.

    .. attribute:: oldTV

        Old target values to be avoided in the acquisition of step 1.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, oldTV=(), **kwargs) -> None:
        self.oldTV = np.array(oldTV)

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
        k = np.random.choice(objdim)
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
            _tau = np.concatenate((tau[0:k], tauk, tau[k:]))
            return -tree.query(_tau)[0]

        # Minimize delta_f
        res = differential_evolution(delta_f, boundsPareto)
        tauk = paretoModel(res.x)
        tau = np.concatenate((res.x[0:k], tauk, res.x[k:]))

        return tau

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

        Perform n attempts to find n points to fill gaps in the Pareto front.

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param nondominated: Nondominated set in the objective space.
        :param paretoFront: Pareto front in the objective space. If not
            provided, use the surrogate to compute it.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = surrogateModel.ntarget

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        if len(paretoFront) == 0:
            paretoFrontIdx = find_pareto_front(surrogateModel.Y)
            paretoFront = surrogateModel.Y[paretoFrontIdx]
            nondominated = surrogateModel.X[paretoFrontIdx]

        # If the Pareto front has only one point or is empty, there is no
        # way to find a target value.
        if len(paretoFront) <= 1:
            return np.empty((0, dim))

        filter = FarEnoughSampleFilter(surrogateModel.X, self.tol(bounds))

        xselected = np.empty((0, dim))
        for i in range(n):
            # Find a target value tau in the Pareto front
            tau = self.pareto_front_target(np.asarray(paretoFront))
            self.oldTV = np.concatenate(
                (self.oldTV.reshape(-1, objdim), [tau]), axis=0
            )

            # Use non-dominated points if provided
            if len(nondominated) > 0:
                Xinit = (
                    nondominated
                    if len(iindex) == 0
                    else np.array(
                        [{i: x[i] for i in range(dim)} for x in nondominated]
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
                lambda x: np.absolute(surrogateModel(x) - tau),
                bounds,
                iindex,
                n_obj=objdim,
            )
            res = pymoo_minimize(
                multiobjTVProblem,
                optimizer,
                seed=len(paretoFront),
                verbose=False,
            )

            # If the Pareto-optimal solution set exists, define the sample point
            # that minimizes the L1 distance to the target value
            if res.X is not None:
                # Save X into an array
                newX = np.array([[x[i] for i in range(dim)] for x in res.X])

                # Eliminate points that are too close to previously
                # sampled points
                newX = filter(newX)

                # Transform the values of the optimization into a matrix
                sx = surrogateModel(newX)

                # Find the values that are expected to be in the Pareto front
                # of the original optimization problem
                nondominated_idx = find_pareto_front(
                    np.vstack((paretoFront, sx)), iStart=len(paretoFront)
                )
                nondominated_idx = [
                    idx - len(paretoFront)
                    for idx in nondominated_idx
                    if idx >= len(paretoFront)
                ]

                # Add a point that is expected to be non-dominated
                if len(nondominated_idx) > 0:
                    idx = np.sum(res.F[nondominated_idx], axis=1).argmin()
                    xselected = np.vstack(
                        (xselected, newX[nondominated_idx][idx : idx + 1])
                    )

        return filter(xselected)
