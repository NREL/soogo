"""Target value acquisition function for RBF surrogate optimization."""

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

from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination

from .base import Acquisition
from ..model import RbfModel
from ..integrations.pymoo import PymooProblem
from .utils import FarEnoughSampleFilter


class TargetValueAcquisition(Acquisition):
    """Target value acquisition function for the RBF model based on [#]_, [#]_,
    and [#]_.

    Every iteration of the algorithm sequentially chooses a number from 0 to
    :attr:`cycleLength` + 1 (inclusive) and runs one of the procedures:

    * Inf-step (0): Selects a sample point that minimizes the
      :math:`\\mu` measure, i.e., :meth:`mu_measure()`. The point selected is
      the farthest from the current sample using the kernel measure.

    * Global search (1 to :attr:`cycleLength`): Minimizes the product of
      :math:`\\mu` measure by the distance to a target value. The target value
      is based on the distance to the current minimum of the surrogate. The
      described measure is known as the 'bumpiness measure'.

    * Local search (:attr:`cycleLength` + 1): Minimizes the bumpiness
      measure with a target value equal to the current minimum of the
      surrogate. If the current minimum is already represented by the
      training points of the
      surrogate, do a global search with a target value slightly smaller than
      the current minimum.

    After each sample point is chosen we verify how close it is from the current
    sample. If it is too close, we replace it by a random point in the domain
    drawn from an uniform distribution. This is strategy was proposed in [#]_.

    :param cycleLength: Length of the global search cycle. Stored in
        :attr:`cycleLength`.

    .. attribute:: cycleLength

        Length of the global search cycle to be used in :meth:`optimize()`.

    .. attribute:: _cycle

        Internal counter of cycles. The value to be used in the next call of
        :meth:`optimize()`.

    References
    ----------
    .. [#] Gutmann, HM. A Radial Basis Function Method for Global
        Optimization. Journal of Global Optimization 19, 201–227 (2001).
        https://doi.org/10.1023/A:1011255519438
    .. [#] Björkman, M., Holmström, K. Global Optimization of Costly
        Nonconvex Functions Using Radial Basis Functions. Optimization and
        Engineering 1, 373–397 (2000). https://doi.org/10.1023/A:1011584207202
    .. [#] Holmström, K. An adaptive radial basis algorithm (ARBF) for expensive
        black-box global optimization. J Glob Optim 41, 447–464 (2008).
        https://doi.org/10.1007/s10898-007-9256-8
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    """

    def __init__(self, cycleLength: int = 6, **kwargs) -> None:
        # Initialize cycle counter and cycle length
        self._cycle = 0
        self.cycleLength = cycleLength

        super().__init__(**kwargs)

        # Use termination criteria based on the relative tolerance. This is used
        # to reduce the time spent in the optimization process.
        default_termination = DefaultSingleObjectiveTermination(
            xtol=self.rtol, period=3
        )
        if "optimizer" not in kwargs:
            self.optimizer.termination = default_termination
        if "mi_optimizer" not in kwargs:
            self.mi_optimizer.termination = default_termination

    @staticmethod
    def bumpiness_measure(
        surrogate: RbfModel, x: np.ndarray, target, target_range=1.0
    ):
        r"""Compute the bumpiness of the surrogate model.

        The bumpiness measure :math:`g_y` was first defined by Gutmann (2001)
        with
        suggestions of usage for global optimization with RBF functions. Gutmann
        notes that :math:`g_y(x)` tends to infinity
        when :math:`x` tends to a training point of the surrogate, and so they
        use :math:`-1/g_y(x)` for the minimization problem. Björkman and
        Holmström use :math:`-\log(1/g_y(x))`, which is the same as minimizing
        :math:`\log(g_y(x))`, to avoid a flat minimum. This option seems to
        slow down convergence rates for :math:`g_y(x)` in `[0,1]` since it
        increases distances in that range.

        The present implementation uses genetic algorithms by default, so there
        is no point in trying to make :math:`g_y` smoother.

        :param surrogate: RBF surrogate model.
        :param x: Possible point to be added to the surrogate model.
        :param target: Target value.
        :param target_range: Known range in the target space. Used to scale
            the function values and avoid overflow.
        """
        absmu = surrogate.mu_measure(x)
        assert all(
            absmu > 0
        )  # if absmu == 0, the linear system in the surrogate model is singular

        # predict RBF value of x
        yhat = surrogate(x)

        # Compute the distance between the predicted value and the target
        dist = np.absolute(yhat - target) / target_range

        # Compute bumpiness measure
        return np.where(absmu < np.inf, (absmu * dist) * dist, np.inf)

    def optimize(
        self,
        surrogateModel: RbfModel,
        bounds,
        n: int = 1,
        *,
        sampleStage: int = -1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points following the algorithm from Holmström et al.(2008).

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param sampleStage: Stage of the sampling process. The default is -1,
            which means that the stage is not specified.
        :param kwargs: Additional keyword arguments (unused).
        :return: n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem
        assert n <= self.cycleLength + 2

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Compute fbounds of the surrogate. Use the filter as suggested by
        # Björkman and Holmström (2000)
        fbounds = [
            surrogateModel.Y.min(),
            surrogateModel.filter(surrogateModel.Y).max(),
        ]
        target_range = fbounds[1] - fbounds[0]
        if target_range == 0:
            target_range = 1

        # Allocate variables a priori targeting batched sampling
        x = np.empty((n, dim))
        mu_measure_is_prepared = False
        x_rbf = None
        f_rbf = None

        # Loop following Holmström (2008)
        for i in range(n):
            if sampleStage >= 0:
                sample_stage = sampleStage
            else:
                sample_stage = self._cycle
                self._cycle = (self._cycle + 1) % (self.cycleLength + 2)
            if sample_stage == 0:  # InfStep - minimize Mu_n
                if not mu_measure_is_prepared:
                    surrogateModel.prepare_mu_measure()
                    mu_measure_is_prepared = True
                problem = PymooProblem(
                    surrogateModel.mu_measure, bounds, iindex
                )

                res = pymoo_minimize(
                    problem,
                    optimizer,
                    seed=surrogateModel.ntrain,
                    verbose=False,
                )

                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])

            elif (
                1 <= sample_stage <= self.cycleLength
            ):  # cycle step global search
                # find min of surrogate model
                if f_rbf is None:
                    problem = PymooProblem(surrogateModel, bounds, iindex)
                    res = pymoo_minimize(
                        problem,
                        optimizer,
                        seed=surrogateModel.ntrain,
                        verbose=False,
                    )
                    assert res.X is not None
                    assert res.F is not None

                    x_rbf = np.asarray([res.X[i] for i in range(dim)])
                    f_rbf = res.F[0]

                wk = (
                    1 - (sample_stage - 1) / self.cycleLength
                ) ** 2  # select weight for computing target value
                f_target = f_rbf - wk * (
                    (fbounds[1] - f_rbf) if fbounds[1] != f_rbf else 1
                )  # target for objective function value

                # use GA method to minimize bumpiness measure
                if not mu_measure_is_prepared:
                    surrogateModel.prepare_mu_measure()
                    mu_measure_is_prepared = True
                problem = PymooProblem(
                    lambda x: TargetValueAcquisition.bumpiness_measure(
                        surrogateModel, x, f_target, target_range
                    ),
                    bounds,
                    iindex,
                )

                res = pymoo_minimize(
                    problem,
                    optimizer,
                    seed=surrogateModel.ntrain,
                    verbose=False,
                )

                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])
            else:  # cycle step local search
                # find the minimum of RBF surface
                if f_rbf is None:
                    problem = PymooProblem(surrogateModel, bounds, iindex)
                    res = pymoo_minimize(
                        problem,
                        optimizer,
                        seed=surrogateModel.ntrain,
                        verbose=False,
                    )
                    assert res.X is not None
                    assert res.F is not None

                    x_rbf = np.asarray([res.X[i] for i in range(dim)])
                    f_rbf = res.F[0]

                xselected = x_rbf
                if f_rbf > (
                    fbounds[0]
                    - 1e-6 * (abs(fbounds[0]) if fbounds[0] != 0 else 1)
                ):
                    f_target = fbounds[0] - 1e-2 * (
                        abs(fbounds[0]) if fbounds[0] != 0 else 1
                    )
                    # use GA method to minimize bumpiness measure
                    if not mu_measure_is_prepared:
                        surrogateModel.prepare_mu_measure()
                        mu_measure_is_prepared = True
                    problem = PymooProblem(
                        lambda x: TargetValueAcquisition.bumpiness_measure(
                            surrogateModel, x, f_target, target_range
                        ),
                        bounds,
                        iindex,
                    )

                    res = pymoo_minimize(
                        problem,
                        optimizer,
                        seed=surrogateModel.ntrain,
                        verbose=False,
                    )

                    assert res.X is not None
                    xselected = np.asarray([res.X[i] for i in range(dim)])

            x[i, :] = xselected

        return FarEnoughSampleFilter(surrogateModel.X, self.tol(bounds))(x)
