"""Acquisition functions for surrogate optimization."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC
# Copyright (C) 2014 Cornell University

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

__authors__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__deprecated__ = False

import numpy as np
from math import log
from abc import ABC, abstractmethod
from typing import Optional, Sequence

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.special import gamma
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize, differential_evolution

# Pymoo imports
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.initialization import Initialization
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.population import Population
from pymoo.termination.default import DefaultSingleObjectiveTermination

# Local imports
from .model.base import Surrogate
from .sampling import NormalSampler, Sampler, Mitchel91Sampler
from .model import LinearRadialBasisFunction, RbfModel, GaussianProcess
from .problem import PymooProblem, ListDuplicateElimination
from .termination import TerminationCondition, UnsuccessfulImprovement, IterateNTimes
from .utils import find_pareto_front
from .optimize_result import OptimizeResult


class AcquisitionFunction(ABC):
    """Base class for acquisition functions.

    This an abstract class. Subclasses must implement the method
    :meth:`optimize()`.

    Acquisition functions are strategies to propose new sample points to a
    surrogate. The acquisition functions here are modeled as objects with the
    goals of adding states to the learning process. Moreover, this design
    enables the definition of the :meth:`optimize()` method with a similar API
    when we compare different acquisition strategies.

    :param optimizer: Continuous optimizer to be used for the acquisition
        function. Default is Differential Evolution (DE) from pymoo.
    :param mi_optimizer: Mixed-integer optimizer to be used for the acquisition
        function. Default is Genetic Algorithm (MixedVariableGA) from pymoo.
    :param rtol: Minimum distance between a candidate point and the
        previously selected points relative to the domain size. Default is 1e-6.

    .. attribute:: optimizer

        Continuous optimizer to be used for the acquisition function. This is
        used in :meth:`optimize()`.

    .. attribute:: mi_optimizer

        Mixed-integer optimizer to be used for the acquisition function. This is
        used in :meth:`optimize()`.

    .. attribute:: rtol

        Minimum distance between a candidate point and the previously selected
        points.  This figures out as a constraint in the optimization problem
        solved in :meth:`optimize()`.
    """

    def __init__(
        self,
        optimizer=None,
        mi_optimizer=None,
        rtol: float = 1e-6,
        termination: Optional[TerminationCondition] = None,
    ) -> None:
        self.optimizer = DE() if optimizer is None else optimizer
        self.mi_optimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
            )
            if mi_optimizer is None
            else mi_optimizer
        )
        self.rtol = rtol
        self.termination = termination

    @abstractmethod
    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Propose a maximum of n new sample points to improve the surrogate.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of requested points. Mind that the number of points
            returned may be smaller than n, depending on the implementation.
        :return: m-by-dim matrix with the selected points, where m <= n.
        """
        pass

    def tol(self, bounds) -> float:
        """Compute tolerance used to eliminate points that are too close to
        previously selected ones.

        The tolerance value is based on :attr:`rtol` and the diameter of the
        largest d-dimensional cube that can be inscribed whithin the bounds.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        """
        return (
            self.rtol
            * np.sqrt(len(bounds))
            * np.min([abs(b[1] - b[0]) for b in bounds])
        )

    def has_converged(self) -> bool:
        """Check if the acquisition function has converged.

        This method is used to check if the acquisition function has converged
        based on a termination criterion. The default implementation always
        returns False.
        """
        if self.termination is not None:
            return self.termination.is_met()
        else:
            return False

    def update(self, out: OptimizeResult, model: Surrogate) -> None:
        """Update the acquisition function knowledge about the optimization
        process.
        """
        if self.termination is not None:
            self.termination.update(out, model)


class WeightedAcquisition(AcquisitionFunction):
    """Select candidates based on the minimization of an weighted average score.

    The weighted average is :math:`w f_s(x) + (1-w) (-d_s(x))`, where
    :math:`f_s(x)` is the surrogate value at :math:`x` and :math:`d_s(x)` is the
    distance of :math:`x` to its closest neighbor in the current sample. Both
    values are scaled to the interval [0, 1], based on the maximum and minimum
    values for the pool of candidates. The sampler generates the candidate
    points to be scored and then selected.

    This acquisition method is prepared deals with multi-objective optimization
    following the random perturbation strategy in [#]_ and [#]_. More
    specificaly, the
    algorithm takes the average value among the predicted target values given by
    the surrogate. In other words, :math:`f_s(x)` is the average value between
    the target components of the surrogate model evaluate at :math:`x`.

    :param Sampler sampler: Sampler to generate candidate points.
        Stored in :attr:`sampler`.
    :param float|sequence weightpattern: Weight(s) `w` to be used in the score.
        Stored in :attr:`weightpattern`.
        The default value is [0.2, 0.4, 0.6, 0.9, 0.95, 1].
    :param maxeval: Description
        Stored in :attr:`maxeval`.

    .. attribute:: neval

        Number of evaluations done so far. Used and updated in
        :meth:`optimize()`.

    .. attribute:: sampler

        Sampler to generate candidate points. Used in :meth:`optimize()`.

    .. attribute:: weightpattern

        Weight(s) `w` to be used in the score. This is a circular list that is
        rotated every time :meth:`optimize()` is called.

    .. attribute:: maxeval

        Maximum number of evaluations. A value 0 means there is no maximum.

    References
    ----------
    .. [#] Regis, R. G., & Shoemaker, C. A. (2012). Combining radial basis
        function surrogates and dynamic coordinate search in
        high-dimensional expensive black-box optimization.
        Engineering Optimization, 45(5), 529–555.
        https://doi.org/10.1080/0305215X.2012.687731
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(
        self,
        sampler,
        weightpattern=None,
        maxeval: int = 0,
        sigma_min: float = 0.0,
        sigma_max: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sampler = sampler
        if weightpattern is None:
            self.weightpattern = [0.2, 0.4, 0.6, 0.9, 0.95, 1]
        elif hasattr(weightpattern, "__len__"):
            self.weightpattern = list(weightpattern)
        else:
            self.weightpattern = [weightpattern]

        self.maxeval = maxeval
        self.neval = 0

        if isinstance(self.sampler, NormalSampler):
            # Continuous local search
            self.remainingCountinuousSearch = 0
            self.nMaxContinuousSearch = len(self.weightpattern)

            # Local search updates
            self.unsuccessful_improvement = UnsuccessfulImprovement(0.001)
            self.success_count = 0
            self.failure_count = 0
            self.success_period = 3
            self.sigma_min = sigma_min
            self.sigma_max = sigma_max

            # Best point found so far
            self.best_known_x = None

    @staticmethod
    def score(
        sx,
        dx,
        weight: float,
        sx_min: float = 0.0,
        sx_max: float = 1.0,
        dx_max: float = 1.0,
    ) -> float:
        r"""Computes the score.

        The score is

        .. math::

            w \frac{s(x)-s_{min}}{s_{max}-s_{min}} +
            (1-w) \frac{d_{max}-d(x,X)}{d_{max}},

        where:

        - :math:`w` is a weight.
        - :math:`s(x)` is the value for the surrogate model on x.
        - :math:`d(x,X)` is the minimum distance between x and the previously
            selected evaluation points.
        - :math:`s_{min}` is the minimum value of the surrogate model.
        - :math:`s_{max}` is the maximum value of the surrogate model.
        - :math:`d_{max}` is the maximum distance between a candidate point and
            the set X of previously selected evaluation points.

        In case :math:`s_{max} = s_{min}`, the score is computed as

        .. math::

            \frac{d_{max}-d(x,X)}{d_{max}}.

        :param sx: Function value(s) :math:`s(x)`.
        :param dx: Distance(s) between candidate(s) and the set X.
        :param weight: Weight :math:`w`.
        :param sx_min: Minimum value of the surrogate model.
        :param sx_max: Maximum value of the surrogate model.
        :param dx_max: Maximum distance between a candidate point and the set X.
        """
        if sx_max == sx_min:
            return (dx_max - dx) / dx_max
        else:
            return (
                weight * ((sx - sx_min) / (sx_max - sx_min))
                + (1 - weight) * (dx_max - dx) / dx_max
            )

    @staticmethod
    def argminscore(
        scaledvalue: np.ndarray,
        dist: np.ndarray,
        weight: float,
        tol: float,
    ) -> int:
        """Gets the index of the candidate point that minimizes the score.

        The score is :math:`w f_s(x) + (1-w) (-d_s(x))`, where

        - :math:`w` is a weight.
        - :math:`f_s(x)` is the estimated value for the objective function on x,
          scaled to [0,1].
        - :math:`d_s(x)` is the minimum distance between x and the previously
          selected evaluation points, scaled to [-1,0].

        Returns -1 if there is no feasible point.

        :param scaledvalue: Function values :math:`f_s(x)` scaled to [0, 1].
        :param dist: Minimum distance between a candidate point and previously
            evaluated sampled points.
        :param weight: Weight :math:`w`.
        :param tol: Tolerance value for excluding candidates that are too close to
            current sample points.
        """
        # Scale distance values to [0,1]
        maxdist = np.max(dist)
        mindist = np.min(dist)
        if maxdist == mindist:
            scaleddist = np.ones(dist.size)
        else:
            scaleddist = (dist - mindist) / (maxdist - mindist)

        # Compute weighted score for all candidates
        score = WeightedAcquisition.score(scaledvalue, scaleddist, weight)

        # Assign bad values to points that are too close to already
        # evaluated/chosen points
        score[dist < tol] = np.inf

        # Return index with the best (smallest) score
        iBest = np.argmin(score)
        if score[iBest] == np.inf:
            print(
                "Warning: all candidates are too close to already evaluated points. Choose a better tolerance."
            )
            return -1

        return int(iBest)

    def minimize_weightedavg_fx_distx(
        self,
        x: np.ndarray,
        distx: np.ndarray,
        fx: np.ndarray,
        n: int,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select n points from a pool of candidates using :meth:`argminscore()`
        iteratively.

        The score on the iteration `i > 1` uses the distances to cadidates
        selected in the iterations `0` to `i-1`.

        :param x: Matrix with candidate points.
        :param distx: Matrix with the distances between the candidate points and
            the m number of rows of x.
        :param fx: Vector with the estimated values for the objective function
            on the candidate points.
        :param n: Number of points to be selected for the next costly
            evaluation.
        :param tol: Tolerance value for excluding candidates that are too close to
            current sample points.
        :return:

            * n-by-dim matrix with the selected points.

            * n-by-(n+m) matrix with the distances between the n selected points
              and the (n+m) sampled points (m is the number of points that have
              been sampled so far).
        """
        # Compute neighbor distances
        dist = np.min(distx, axis=1)

        m = distx.shape[1]
        dim = x.shape[1]

        xselected = np.zeros((n, dim))
        distselected = np.zeros((n, m + n))

        # Scale function values to [0,1]
        if fx.ndim == 1:
            minval = np.min(fx)
            maxval = np.max(fx)
            if minval == maxval:
                scaledvalue = np.ones(fx.size)
            else:
                scaledvalue = (fx - minval) / (maxval - minval)
        elif fx.ndim == 2:
            minval = np.min(fx, axis=0)
            maxval = np.max(fx, axis=0)
            scaledvalue = np.average(
                np.where(
                    maxval - minval > 0, (fx - minval) / (maxval - minval), 1
                ),
                axis=1,
            )

        selindex = self.argminscore(
            scaledvalue, dist, self.weightpattern[0], tol
        )
        if selindex >= 0:
            xselected[0, :] = x[selindex, :]
            distselected[0, 0:m] = distx[selindex, :]
        else:
            return np.empty((0, dim)), np.empty((0, m))

        for ii in range(1, n):
            # compute distance of all candidate points to the previously selected
            # candidate point
            newDist = cdist(xselected[ii - 1, :].reshape(1, -1), x)[0]
            dist = np.minimum(dist, newDist)

            selindex = self.argminscore(
                scaledvalue,
                dist,
                self.weightpattern[ii % len(self.weightpattern)],
                tol,
            )
            if selindex >= 0:
                xselected[ii, :] = x[selindex, :]
            else:
                return xselected[0:ii], distselected[0:ii, 0 : m + ii]

            distselected[ii, 0:m] = distx[selindex, :]
            for j in range(ii - 1):
                distselected[ii, m + j] = np.linalg.norm(
                    xselected[ii, :] - xselected[j, :]
                )
                distselected[j, m + ii] = distselected[ii, m + j]
            distselected[ii, m + ii - 1] = newDist[selindex]
            distselected[ii - 1, m + ii] = distselected[ii, m + ii - 1]

        return xselected, distselected

    def tol(self, bounds) -> float:
        """Compute tolerance used to eliminate points that are too close to
        previously selected ones.

        The tolerance value is based on :attr:`rtol` and the diameter of the
        largest d-dimensional cube that can be inscribed whithin the bounds.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        """
        tol0 = super().tol(bounds)
        if isinstance(self.sampler, NormalSampler):
            # Consider the region with 95% of the values on each
            # coordinate, which has diameter `4*sigma`
            return tol0 * min(4 * self.sampler.sigma, 1.0)
        else:
            return tol0

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Generate a number of candidates using the :attr:`sampler`. Then,
        select up to n points that maximize the score.

        When `sampler.strategy` is
        :attr:`soogo.sampling.SamplingStrategy.DDS` or
        :attr:`soogo.sampling.SamplingStrategy.DDS_UNIFORM`, the
        probability is computed based on the DYCORS method as proposed by Regis
        and Shoemaker (2012).

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points requested.
        :param xbest: Best point so far. Used if :attr:`sampler` is an instance
            of :class:`soogo.sampling.NormalSampler`. If not provided,
            compute it based on the training data for the surrogate.
        :return: m-by-dim matrix with the selected points, where m <= n.
        """
        dim = len(bounds)  # Dimension of the problem
        objdim = surrogateModel.ntarget
        iindex = surrogateModel.iindex

        # Generate candidates
        if isinstance(self.sampler, NormalSampler):
            if "xbest" in kwargs:
                xbest = kwargs["xbest"]
            elif objdim > 1:
                xbest = surrogateModel.X[find_pareto_front(surrogateModel.Y)]
            else:
                xbest = surrogateModel.X[surrogateModel.Y.argmin()]

            # Do local continuous search when asked
            if self.remainingCountinuousSearch > 0:
                coord = [i for i in range(dim) if i not in iindex]
            else:
                coord = [i for i in range(dim)]

            # Compute probability in case DDS is used
            if self.maxeval > 1 and self.neval < self.maxeval:
                prob = min(20 / dim, 1) * (
                    1 - (log(self.neval + 1) / log(self.maxeval))
                )
            else:
                prob = 1.0

            x = self.sampler.get_sample(
                bounds,
                iindex=iindex,
                mu=xbest,
                probability=prob,
                coord=coord,
            )
        else:
            x = self.sampler.get_sample(bounds, iindex=iindex)

        # Evaluate candidates
        fx = surrogateModel(x)

        # Select best candidates
        xselected, _ = self.minimize_weightedavg_fx_distx(
            x, cdist(x, surrogateModel.X), fx, n, self.tol(bounds)
        )
        n = xselected.shape[0]

        # Rotate weight pattern
        self.weightpattern[:] = (
            self.weightpattern[n % len(self.weightpattern) :]
            + self.weightpattern[: n % len(self.weightpattern)]
        )

        # Update number of evaluations
        self.neval += n

        # In case of continuous search, update counter
        # Keep at least one iteration in continuous search mode since it can
        # only be deactivated by `update()`.
        if isinstance(self.sampler, NormalSampler):
            if self.remainingCountinuousSearch > 0:
                self.remainingCountinuousSearch = max(
                    self.remainingCountinuousSearch - n, 1
                )

        return xselected

    def update(self, out: OptimizeResult, model: Surrogate) -> None:
        # Update the termination condition if it is set
        if self.termination is not None:
            self.termination.update(out, model)

        # Check if the sampler is a NormalSampler and the output has only one
        # objective. If not, do nothing.
        if (not isinstance(self.sampler, NormalSampler)) or (out.nobj != 1):
            return

        # Check if the last sample was successful
        self.unsuccessful_improvement.update(out, model)
        recent_success = not self.unsuccessful_improvement.is_met()

        # In continuous search mode
        if self.remainingCountinuousSearch > 0:
            # In case of a successful sample, reset the counter to the maximum
            if recent_success:
                self.remainingCountinuousSearch = self.nMaxContinuousSearch
            # Otherwise, decrease the counter
            else:
                self.remainingCountinuousSearch -= 1

            # Update termination and reset internal state
            if self.termination is not None:
                self.termination.reset(keep_data_knowledge=True)

        # In case of a full search
        else:
            # Update counters and activate continuous search mode if needed
            if recent_success:
                # If there is an improvement in an integer variable
                if (
                    model is not None
                    and self.best_known_x is not None
                    and any(
                        [
                            out.x[i] != self.best_known_x[i]
                            for i in model.iindex
                        ]
                    )
                ):
                    # Activate the continuous search mode
                    self.remainingCountinuousSearch = self.nMaxContinuousSearch

                    # Reset the success and failure counters
                    self.success_count = 0
                    self.failure_count = 0
                else:
                    # Update counters
                    self.success_count += 1
                    self.failure_count = 0
            else:
                # Update counters
                self.success_count = 0
                self.failure_count += 1

            # Check if sigma should be reduced based on the failures
            # If the termination condition is set, use it instead of
            # failure_count
            if self.termination is not None:
                if self.termination.is_met():
                    self.sampler.sigma *= 0.5
                    if self.sampler.sigma < self.sigma_min:
                        # Algorithm is probably in a local minimum!
                        self.sampler.sigma = self.sigma_min
                    else:
                        self.failure_count = 0
                        self.termination.reset(keep_data_knowledge=True)
            else:
                dim = out.x.shape[-1]
                failure_period = max(5, dim)

                if self.failure_count >= failure_period:
                    self.sampler.sigma *= 0.5
                    if self.sampler.sigma < self.sigma_min:
                        # Algorithm is probably in a local minimum!
                        self.sampler.sigma = self.sigma_min
                    else:
                        self.failure_count = 0

            # Check if sigma should be increased based on the successes
            if self.success_count >= self.success_period:
                self.sampler.sigma *= 2
                if self.sampler.sigma > self.sigma_max:
                    self.sampler.sigma = self.sigma_max
                else:
                    self.success_count = 0

        # Update the best known x
        self.best_known_x = np.copy(out.x)


class TargetValueAcquisition(AcquisitionFunction):
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

    * Local search (:attr:`cycleLength` + 1): Minimizes the bumpiness measure with
      a target value equal to the current minimum of the surrogate. If the
      current minimum is already represented by the training points of the
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
        self._cycle = 0
        self.cycleLength = cycleLength

        # Use termination criteria based on the relative tolerance. This is used
        # to reduce the time spent in the optimization process.
        super().__init__(**kwargs)
        termination = DefaultSingleObjectiveTermination(
            xtol=self.rtol, period=3
        )
        self.optimizer.termination = termination
        self.mi_optimizer.termination = termination

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
            the function values to avoid overflow.
        """
        absmu = surrogate.mu_measure(x)
        assert all(
            absmu > 0
        )  # if absmu == 0, the linear system in the surrogate model singular

        # predict RBF value of x
        yhat = surrogate(x)

        # Compute the distance between the predicted value and the target
        dist = np.absolute(yhat - target) / target_range

        # Use sqrt(gy) as the bumpiness measure to avoid overflow due to
        # squaring big values. We do not make the function continuSee
        # Gutmann (2001). Underflow may happen when candidates are close to the
        # desired target value.
        #
        # Gutmann (2001):
        # return -1 / ((absmu * dist) * dist)
        #
        # Björkman, M., Holmström (2000):
        # return np.log((absmu * dist) * dist)
        #
        # Here:
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
        :return: n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem
        assert n <= self.cycleLength + 2

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Create a KDTree with the current training points
        tree = KDTree(surrogateModel.X)
        atol = self.tol(bounds)

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

            # Replace points that are too close to current sample
            current_sample = np.concatenate((surrogateModel.X, x[0:i]), axis=0)
            while np.any(tree.query(xselected)[0] < atol) or (
                i > 0 and cdist(xselected.reshape(1, -1), x[0:i]).min() < atol
            ):
                # the selected point is too close to already evaluated point
                # randomly select point from variable domain
                xselected = Mitchel91Sampler(1).get_mitchel91_sample(
                    bounds,
                    iindex=iindex,
                    current_sample=current_sample,
                )

            x[i, :] = xselected

        return x


class MinimizeSurrogate(AcquisitionFunction):
    """Obtain sample points that are local minima of the surrogate model.

    This implementation is based on the one of MISO-MS used in the paper [#]_.
    The original method, Multi-level Single-Linkage, was described in [#]_.
    In each iteration, the algorithm generates a pool of candidates and select
    the best candidates (lowest predicted value) that are far enough from each
    other. The number of candidates chosen as well as the distance threshold
    vary with each iteration. The hypothesis is that the successful candidates
    each belong to a different region in the space, which may contain a local
    minimum, and those regions cover the whole search space. In the sequence,
    the algorithm runs multiple local minimization procedures using the
    successful candidates as local guesses. The results of the minimization are
    collected for the new sample.

    :param nCand: Number of candidates used on each iteration.
    :param rtol: Minimum distance between a candidate point and the
        previously selected points relative to the domain size. Default is 1e-3.

    .. attribute:: sampler

        Sampler to generate candidate points.

    References
    ----------
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    .. [#] Rinnooy Kan, A.H.G., Timmer, G.T. Stochastic global optimization
        methods part II: Multi level methods. Mathematical Programming 39, 57–78
        (1987). https://doi.org/10.1007/BF02592071
    """

    def __init__(self, nCand: int, rtol: float = 1e-3, **kwargs) -> None:
        super().__init__(rtol=rtol, **kwargs)
        self.sampler = Sampler(nCand)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points based on MISO-MS from Müller (2016).

        The critical distance is the same used in the seminal work from
        Rinnooy Kan and Timmer (1987).

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Max number of points to be acquired.
        :return: n-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        volumeBounds = np.prod([b[1] - b[0] for b in bounds])

        # Get index and bounds of the continuous variables
        cindex = [i for i in range(dim) if i not in surrogateModel.iindex]
        cbounds = [bounds[i] for i in cindex]

        # Local parameters
        remevals = 1000 * dim  # maximum number of RBF evaluations
        maxiter = 10  # maximum number of iterations to find local minima.
        sigma = 4.0  # default value for computing crit distance
        critdist = (
            (gamma(1 + (dim / 2)) * volumeBounds * sigma) ** (1 / dim)
        ) / np.sqrt(np.pi)  # critical distance when 2 points are equal

        # Local space to store information
        candidates = np.empty((self.sampler.n * maxiter, dim))
        distCandidates = np.empty(
            (self.sampler.n * maxiter, self.sampler.n * maxiter)
        )
        fcand = np.empty(self.sampler.n * maxiter)
        startpID = np.full((self.sampler.n * maxiter,), False)
        selected = np.empty((n, dim))

        # Create a KDTree with the training data points
        tree = KDTree(surrogateModel.X)
        atol = self.tol(bounds)

        iter = 0
        k = 0
        while iter < maxiter and k < n and remevals > 0:
            iStart = iter * self.sampler.n
            iEnd = (iter + 1) * self.sampler.n

            # if computational budget is exhausted, then return
            if remevals <= iEnd - iStart:
                break

            # Critical distance for the i-th iteration
            critdistiter = critdist * (log(iEnd) / iEnd) ** (1 / dim)

            # Consider only the best points to start local minimization
            counterLocalStart = iEnd // maxiter

            # Choose candidate points uniformly in the search space
            candidates[iStart:iEnd, :] = self.sampler.get_uniform_sample(
                bounds, iindex=surrogateModel.iindex
            )

            # Compute the distance between the candidate points
            distCandidates[iStart:iEnd, iStart:iEnd] = cdist(
                candidates[iStart:iEnd, :], candidates[iStart:iEnd, :]
            )
            distCandidates[0:iStart, iStart:iEnd] = cdist(
                candidates[0:iStart, :], candidates[iStart:iEnd, :]
            )
            distCandidates[iStart:iEnd, 0:iStart] = distCandidates[
                0:iStart, iStart:iEnd
            ].T

            # Evaluate the surrogate model on the candidate points and sort them
            fcand[iStart:iEnd] = surrogateModel(candidates[iStart:iEnd, :])
            ids = np.argsort(fcand[0:iEnd])
            remevals -= iEnd - iStart

            # Select the best points that are not too close to each other
            chosenIds = np.zeros((counterLocalStart,), dtype=int)
            nSelected = 0
            for i in range(counterLocalStart):
                if not startpID[ids[i]]:
                    select = True
                    for j in range(i):
                        if distCandidates[ids[i], ids[j]] <= critdistiter:
                            select = False
                            break
                    if select:
                        chosenIds[nSelected] = ids[i]
                        nSelected += 1
                        startpID[ids[i]] = True

            # Evolve the best points to find the local minima
            for i in range(nSelected):
                xi = candidates[chosenIds[i], :]

                def func_continuous_search(x):
                    x_ = xi.copy()
                    x_[cindex] = x
                    return surrogateModel(x_)

                def dfunc_continuous_search(x):
                    x_ = xi.copy()
                    x_[cindex] = x
                    return surrogateModel.jac(x_)[cindex]

                # def hessp_continuous_search(x, p):
                #     x_ = xi.copy()
                #     x_[cindex] = x
                #     p_ = np.zeros(dim)
                #     p_[cindex] = p
                #     return surrogateModel.hessp(x_, p_)[cindex]

                res = minimize(
                    func_continuous_search,
                    xi[cindex],
                    method="L-BFGS-B",
                    jac=dfunc_continuous_search,
                    # hessp=hessp_continuous_search,
                    bounds=cbounds,
                    options={
                        "maxfun": remevals,
                        "maxiter": max(2, round(remevals / 20)),
                        "disp": False,
                    },
                )
                remevals -= res.nfev
                xi[cindex] = res.x

                if tree.n == 0 or tree.query(xi)[0] >= atol:
                    selected[k, :] = xi
                    k += 1
                    if k == n:
                        break
                    else:
                        tree = KDTree(
                            np.concatenate(
                                (surrogateModel.X, selected[0:k, :]),
                                axis=0,
                            )
                        )

                if remevals <= 0:
                    break

            e_nlocmin = (
                k * (counterLocalStart - 1) / (counterLocalStart - k - 2)
            )
            e_domain = (
                (counterLocalStart - k - 1)
                * (counterLocalStart + k)
                / (counterLocalStart * (counterLocalStart - 1))
            )
            if (e_nlocmin - k < 0.5) and (e_domain >= 0.995):
                break

            iter += 1

        if k > 0:
            return selected[0:k, :]
        else:
            # No new points found by the differential evolution method
            singleCandSampler = Mitchel91Sampler(1)
            selected = singleCandSampler.get_mitchel91_sample(
                bounds,
                iindex=surrogateModel.iindex,
                current_sample=surrogateModel.X,
            )
            while tree.query(selected)[0] < atol:
                selected = singleCandSampler.get_mitchel91_sample(
                    bounds,
                    iindex=surrogateModel.iindex,
                    current_sample=surrogateModel.X,
                )
            return selected.reshape(1, -1)


class ParetoFront(AcquisitionFunction):
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
        assert objdim > 1

        # Discard duplicated points in the Pareto front
        # TODO: Use a more efficient method to discard duplicates
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

            # Find the Pareto-optimal solution set that minimizes dist(s(x),tau).
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

        return xselected


class EndPointsParetoFront(AcquisitionFunction):
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
            endpoints = maximizeDistance.optimize(
                surrogateModel,
                bounds,
                n=1
            )

        # Return a maximum of n points
        return endpoints[:n, :]


class MinimizeMOSurrogate(AcquisitionFunction):
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


class CoordinatePerturbationOverNondominated(AcquisitionFunction):
    """Coordinate perturbation acquisition function over the nondominated set.

    This acquisition method was proposed in [#]_. It perturbs locally each of
    the non-dominated sample points to find new sample points. The perturbation
    is performed by :attr:`acquisitionFunc`.

    :param acquisitionFunc: Weighted acquisition function with a normal sampler.
        Stored in :attr:`acquisitionFunc`.

    .. attribute:: acquisitionFunc

        Weighted acquisition function with a normal sampler.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, acquisitionFunc: WeightedAcquisition, **kwargs) -> None:
        self.acquisitionFunc = acquisitionFunc
        assert isinstance(self.acquisitionFunc.sampler, NormalSampler)
        super().__init__(**kwargs)

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

        :param surrogateModel: Multi-target surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :param nondominated: Nondominated set in the objective space.
        :param paretoFront: Pareto front in the objective space.
        """
        dim = len(bounds)
        atol = self.acquisitionFunc.tol(bounds)
        assert isinstance(self.acquisitionFunc.sampler, NormalSampler)

        # Find a collection of points that are close to the Pareto front
        bestCandidates = np.empty((0, dim))
        for ndpoint in nondominated:
            x = self.acquisitionFunc.optimize(
                surrogateModel, bounds, 1, xbest=ndpoint
            )
            # Choose points that are not too close to previously selected points
            if bestCandidates.size == 0:
                if x.size > 0:
                    bestCandidates = x.reshape(1, -1)
            else:
                distNeighborOfx = cdist(x, bestCandidates).min()
                if distNeighborOfx >= atol:
                    bestCandidates = np.concatenate(
                        (bestCandidates, x), axis=0
                    )

        # Return if no point was found
        if bestCandidates.size == 0:
            return bestCandidates

        # Eliminate points predicted to be dominated
        fnondominatedAndBestCandidates = np.concatenate(
            (paretoFront, surrogateModel(bestCandidates)), axis=0
        )
        idxPredictedPareto = find_pareto_front(
            fnondominatedAndBestCandidates,
            iStart=len(nondominated),
        )
        idxPredictedBest = [
            i - len(nondominated)
            for i in idxPredictedPareto
            if i >= len(nondominated)
        ]
        bestCandidates = bestCandidates[idxPredictedBest, :]

        return bestCandidates[:n, :]


class GosacSample(AcquisitionFunction):
    """GOSAC acquisition function as described in [#]_.

    Minimize the objective function with surrogate constraints. If a feasible
    solution is found and is different from previous sample points, return it as
    the new sample. Otherwise, the new sample is the point that is farthest from
    previously selected sample points.

    This acquisition function is only able to acquire 1 point at a time.

    :param fun: Objective function. Stored in :attr:`fun`.

    .. attribute:: fun

        Objective function.

    References
    ----------
    .. [#] Juliane Mueller and Joshua D. Woodbury. GOSAC: global optimization
        with surrogate approximation of constraints.
        J Glob Optim, 69:117-136, 2017.
        https://doi.org/10.1007/s10898-017-0496-y
    """

    def __init__(self, fun, rtol: float = 1e-3, termination: Optional[TerminationCondition] = None, **kwargs) -> None:
        super().__init__(rtol=rtol, **kwargs)
        self.fun = fun
        self.termination = termination

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        constraintTransform: callable = None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire 1 point.

        :param surrogateModel: Multi-target surrogate model for the constraints.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Unused.
        :param constraintTransform: Function to transform the constraints.
            If not provided, use the identity function. The optimizer, pymoo,
            expects that negative and zero values are feasible, and positive
            values are infeasible.
        :return: 1-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        gdim = surrogateModel.ntarget

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        assert n == 1

        if constraintTransform is None:
            def constraintTransform(x):
                return x

        def transformedConstraint(x):
            surrogateOutput = surrogateModel(x)
            return constraintTransform(surrogateOutput)

        # Create KDTree with previously evaluated points
        tree = KDTree(surrogateModel.X)
        atol = self.tol(bounds)

        cheapProblem = PymooProblem(
            self.fun, bounds, iindex, gfunc=transformedConstraint, n_ieq_constr=gdim
        )
        res = pymoo_minimize(
            cheapProblem,
            optimizer,
            seed=surrogateModel.ntrain,
            verbose=False,
        )

        # If either no feasible solution was found or the solution found is too
        # close to already sampled points, we then
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of sample points.
        isGoodCandidate = True
        if res.X is None:
            isGoodCandidate = False
        else:
            xnew = np.asarray([[res.X[i] for i in range(dim)]])
            if tree.query(xnew)[0] < atol:
                isGoodCandidate = False

        if not isGoodCandidate:
            maximizeDistance = MaximizeDistance(rtol=self.rtol)

            xnew = maximizeDistance.optimize(
                surrogateModel,
                bounds,
                n=1
            )

        return xnew


class MaximizeEI(AcquisitionFunction):
    """Acquisition by maximization of the expected improvement of a Gaussian
    Process.

    It starts by running a
    global optimization algorithm to find a point `xs` that maximizes the EI. If
    this point is found and the sample size is 1, return this point. Else,
    creates a pool of candidates using :attr:`sampler` and `xs`. From this pool,
    select the set of points with that maximize the expected improvement. If
    :attr:`avoid_clusters` is `True` avoid points that are too close to already
    chosen ones inspired in the strategy from [#]_. Mind that the latter
    strategy can slow down considerably the acquisition process, although is
    advisable for a sample of good quality.

    :param sampler: Sampler to generate candidate points. Stored in
        :attr:`sampler`.
    :param avoid_clusters: When `True`, use a strategy that avoids points too
        close to already chosen ones. Stored in :attr:`avoid_clusters`.

    .. attribute:: sampler

        Sampler to generate candidate points.

    .. attribute:: avoid_clusters

        When `True`, use a strategy that avoids points too close to already
        chosen ones.

    References
    ----------
    .. [#] Che Y, Müller J, Cheng C. Dispersion-enhanced sequential batch
        sampling for adaptive contour estimation. Qual Reliab Eng Int. 2024;
        40: 131–144. https://doi.org/10.1002/qre.3245
    """

    def __init__(
        self, sampler=None, avoid_clusters: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.sampler = Sampler(0) if sampler is None else sampler
        self.avoid_clusters = avoid_clusters

    def optimize(
        self,
        surrogateModel: GaussianProcess,
        bounds,
        n: int = 1,
        *,
        ybest=None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Run a global optimization procedure to try to find a point that has the
        highest expected improvement for the Gaussian Process.
        Moreover, if `ybest` isn't provided, run a global optimization procedure
        to find the minimum value of the surrogate model. Use the minimum point
        as a candidate for this acquisition.

        This implementation only works for continuous design variables.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param ybest: Best point so far. If not provided, find the minimum value
            for the surrogate. Use it as a possible candidate.
        """
        assert len(surrogateModel.iindex) == 0
        if n == 0:
            return np.empty((0, len(bounds)))

        xbest = None
        if ybest is None:
            # Compute an estimate for ybest using the surrogate.
            res = differential_evolution(
                lambda x: surrogateModel(np.asarray([x])), bounds
            )
            ybest = res.fun
            if res.success:
                xbest = res.x

        # Use the point that maximizes the EI
        res = differential_evolution(
            lambda x: -surrogateModel.expected_improvement(
                np.asarray([x]), ybest
            ),
            bounds,
        )
        xs = res.x if res.success else None

        # Returns xs if n == 1
        if res.success and n == 1:
            return np.asarray([xs])

        # Generate the complete pool of candidates
        if isinstance(self.sampler, Mitchel91Sampler):
            current_sample = surrogateModel.X
            if xs is not None:
                current_sample = np.concatenate((current_sample, [xs]), axis=0)
            if xbest is not None:
                current_sample = np.concatenate(
                    (current_sample, [xbest]), axis=0
                )
            x = self.sampler.get_mitchel91_sample(
                bounds, current_sample=current_sample
            )
        else:
            x = self.sampler.get_sample(bounds)

        if xs is not None:
            x = np.concatenate(([xs], x), axis=0)
        if xbest is not None:
            x = np.concatenate((x, [xbest]), axis=0)
        nCand = len(x)

        # Create EI and kernel matrices
        eiCand = surrogateModel.expected_improvement(x, ybest)

        # If there is no need to avoid clustering return the maximum of EI
        if not self.avoid_clusters or n == 1:
            return x[np.flip(np.argsort(eiCand)[-n:]), :]
        # Otherwise see what follows...

        # Rescale EI to [0,1] and create the kernel matrix with all candidates
        if eiCand.max() > eiCand.min():
            eiCand = (eiCand - eiCand.min()) / (eiCand.max() - eiCand.min())
        else:
            eiCand = np.ones_like(eiCand)
        Kss = surrogateModel.eval_kernel(x)

        # Score to be maximized and vector with the indexes of the candidates
        # chosen.
        score = np.zeros(nCand)
        iBest = np.empty(n, dtype=int)

        # First iteration
        j = 0
        for i in range(nCand):
            Ksi = Kss[:, i]
            Kii = Kss[i, i]
            score[i] = ((np.dot(Ksi, Ksi) / Kii) / nCand) * eiCand[i]
        iBest[j] = np.argmax(score)
        eiCand[iBest[j]] = 0.0  # Remove this candidate expectancy

        # Remaining iterations
        for j in range(1, n):
            currentBatch = iBest[0:j]

            Ksb = Kss[:, currentBatch]
            Kbb = Ksb[currentBatch, :]

            # Cholesky factorization using points in the current batch
            Lfactor = cholesky(Kbb, lower=True)

            # Solve linear systems for KbbInvKbs
            LInvKbs = solve_triangular(Lfactor, Ksb.T, lower=True)
            KbbInvKbs = solve_triangular(
                Lfactor, LInvKbs, lower=True, trans="T"
            )

            # Compute the b part of the score
            scoreb = np.sum(np.multiply(Ksb, KbbInvKbs.T))

            # Reserve memory to avoid excessive dynamic allocations
            aux0 = np.empty(nCand)
            aux1 = np.empty((j, nCand))

            # If the remaining candidates are not expected to improve the
            # solution, choose sample based on the distance criterion only.
            if np.max(eiCand) == 0.0:
                eiCand[:] = 1.0

            # Compute the final score
            for i in range(nCand):
                if i in currentBatch:
                    score[i] = 0
                else:
                    # Compute the square of the diagonal term of the updated Cholesky factorization
                    li = LInvKbs[:, i]
                    d2 = Kss[i, i] - np.dot(li, li)

                    # Solve the linear system Kii*aux = Ksi.T
                    Ksi = Kss[:, i]
                    aux0[:] = (Ksi.T - LInvKbs.T @ li) / d2
                    aux1[:] = LInvKbs - np.outer(li, aux0)
                    aux1[:] = solve_triangular(
                        Lfactor, aux1, lower=True, trans="T", overwrite_b=True
                    )

                    # Local score computation
                    scorei = np.sum(np.multiply(Ksb, aux1.T)) + np.dot(
                        Ksi, aux0
                    )

                    # Final score
                    score[i] = ((scorei - scoreb) / nCand) * eiCand[i]
                    # assert(score[i] >= 0)

            iBest[j] = np.argmax(score)
            eiCand[iBest[j]] = 0.0  # Remove this candidate expectancy

        return x[iBest, :]


class TransitionSearch(AcquisitionFunction):
    """
    Transition search acquisition function as described in [#]_.

    This acquisition function is used to find new sample points by perturbing
    the current best sample point and uniformly selecting points from the
    domain. The scoreWeight parameter can be used to control the transition
    from local to global search. A scoreWeight close to 1.0 will favor
    the predicted function value (local search), while a scoreWeight close to
    0.0 will favor the distance to previously sampled points (global search).

    The evaluability of candidate points is predicted using the candidate
    surrogate model. If the evaluability probaility of a candidate point
    is below a threshold, the point is discarded. If no candidate points
    remain after this filtering, all candidate points are considered
    evaluable.

    The candidate points are scored using a weighted value of their predicted
    function value and the distance to previously sampled points. The candidate
    with the best total score is selected as the new sample point.

    :param rtol: Minimum distance between a candidate point and the
        previously selected points relative to the domain size. Default is 1e-3.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, rtol: float = 1e-3, termination: Optional[TerminationCondition] = None, **kwargs) -> None:
        super().__init__(rtol=rtol, **kwargs)
        self.termination = termination

    def generate_candidates(
        self,
        surrogateModel: Surrogate,
        bounds,
        nCand: int,
        xbest: np.ndarray = None,
    ) -> np.ndarray:
        """
        Generate candidate points by perturbing the current best point and
        uniformly sampling from the bounds. A total of 2* nCand candidate
        points are generated, where the first nCand points are perturbations of
        the current best point and the second nCand points are uniformly sampled
        from the bounds.

        :param surrogateModel: Surrogate model for the objective function.
        :param bounds: List with the limits [x_min, x_max] of each direction.
        :param nCand: Number of candidate points to be generated.
        :param xbest: Current best point. If None, use the best point from the
            surrogate model.
        :return: Array of candidate points.
        """
        dim = len(bounds)
        bounds = np.asarray(bounds)

        if xbest is None:
            best_idx = np.argmin(surrogateModel.Y)
            xbest = surrogateModel.X[best_idx]

        ## Create nCand points by perturbing the current best point
        # Set perturbation probability
        if dim <= 10:
            perturbProbability = 1.0
        else:
            perturbProbability = np.random.uniform(0, 1)

        # Generate perturbation candidates
        perturbationSampler = NormalSampler(n=nCand, sigma=0.02)
        perturbationCandidates = perturbationSampler.get_dds_sample(
            bounds, mu=xbest, probability=perturbProbability
        )

        ## Generate nCand points uniformly from the bounds
        uniformSampler = Sampler(nCand)
        uniformCandidates = uniformSampler.get_uniform_sample(bounds)

        # Combine perturbation and uniform candidates
        candidates = np.vstack((perturbationCandidates, uniformCandidates))
        return candidates

    def select_candidates(
        self,
        surrogateModel: Surrogate,
        candidates: np.ndarray,
        bounds,
        n: int = 1,
        scoreWeight: float = 0.5,
        evaluabilityThreshold: float = 0.25,
        evaluabilitySurrogate: Surrogate = None,
    ) -> np.ndarray:
        """
        Select the best candidate points based on the predicted function
        value and distance to previously sampled points. The candidates are
        scored using a weighted score that combines the predicted function
        value and the distance to previously sampled points. Uses iterative
        selection for multiple points.

        :param surrogateModel: Surrogate model for the objective function.
        :param candidates: Array of candidate points.
        :param bounds: List with the limits [x_min, x_max] of each direction.
        :param n: Number of best candidates to return.
        :param scoreWeight: Weight for the predicted function value and distance
            scores in the total score.
        :param evaluabilityThreshold: Threshold for the evaluability
            probability. Candidates with evaluability probability below this
            threshold are discarded.
        :param evaluabilitySurrogate: Surrogate model for the evaluability
            probability of the candidate points. If provided, candidates with
            evaluability probability below the threshold are discarded.
        :return: The n best candidate points.
        """
        # Calculate tolerance using the tol function
        atol = self.tol(bounds)
        ## Check evaluability of candidates
        if evaluabilitySurrogate is not None:
            evaluability = evaluabilitySurrogate(candidates)
            # Keep candidates above the evaluability threshold
            if len(candidates[evaluability > evaluabilityThreshold]) > 0:
                candidates = candidates[evaluability > evaluabilityThreshold]
            else:
                # If no candidates are above the evaluability threshold, keep
                # candidates with positive evaluability
                candidates = candidates[evaluability > 0]

        ## Rank candidates based on their predicted function value and
        # distance to previously sampled points
        # Get the predicted function values for the candidates
        predictedValues = surrogateModel(candidates)

        # Scale the predicted values to [0, 1]
        # Maps highest value to 1 and lowest to 0
        # Smaller predicted values are better
        if predictedValues.max() == predictedValues.min():
            valueScore = np.ones_like(predictedValues)
        else:
            valueScore = (predictedValues - predictedValues.min()) / (
                predictedValues.max() - predictedValues.min()
            )
        # Compute distances to previously evaluated points
        if evaluabilitySurrogate is not None:
            tree = KDTree(evaluabilitySurrogate.X)
        else:
            tree = KDTree(surrogateModel.X)

        distances, _ = tree.query(candidates, k=1)

        scorer = WeightedAcquisition(None)

        # Initialize arrays to store selected points
        dim = candidates.shape[1]
        selectedPoints = np.zeros((n, dim))
        nSelected = 0

        # Copy distances array for iterative updates
        currentDistances = distances.copy()

        # Iteratively select n points
        for i in range(n):
            best_idx = scorer.argminscore(
                valueScore, currentDistances, weight=scoreWeight, tol=atol
            )

            if best_idx == -1:
                break

            selectedPoints[i] = candidates[best_idx]
            nSelected += 1

            # If more points needed, update distances to include distance to
            # newly selected point
            if i < n - 1:
                newDistances = cdist(
                    candidates[best_idx].reshape(1, -1), candidates
                )[0]
                currentDistances = np.minimum(
                    currentDistances, newDistances
                )

        # Return only the successfully selected points
        if nSelected == 0:
            return np.empty((0, dim))
        elif nSelected == 1:
            return selectedPoints
        else:
            return selectedPoints[:nSelected]

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        *,
        evaluabilitySurrogate: Surrogate = None,
        evaluabilityThreshold: float = 0.25,
        scoreWeight: float = 0.5,
        **kwargs,
    ) -> np.ndarray:
        """
        This acquisition function generates candidate points by perturbing the
        current best point and uniformly sampling from the bounds. It then
        selects the best candidate point based on a weighted score that combines
        the predicted function value and the distance to previously sampled
        points.

        :param surrogateModel: Surrogate model for the objective function.
        :param bounds: List with the limits [x_min, x_max] of each direction.
        :param n: Number of points to be acquired.
        :param evaluabilitySurrogate: Surrogate model for the evaluability
            probability of the candidate points. If provided, candidates with
            evaluability probability below the threshold are discarded.
        :param evaluabilityThreshold: Threshold for the evaluability probability.
        :param scoreWeight: Weight for the predicted function value and distance
            scores in the total score. The total score is computed as:
            `scoreWeight * valueScore + (1 - scoreWeight) * distanceScore`.
        """
        # Set Ncand = 500*dim
        dim = len(bounds)
        nCand = 500 * dim

        bounds = np.asarray(bounds)

        # Get current best point
        best_idx = np.argmin(surrogateModel.Y)
        xbest = surrogateModel.X[best_idx]

        # Generate candidate points
        candidates = self.generate_candidates(
            surrogateModel, bounds, nCand, xbest=xbest
        )

        # Select n best candidates
        bestCandidates = self.select_candidates(
            surrogateModel,
            candidates,
            bounds,
            n=n,
            scoreWeight=scoreWeight,
            evaluabilityThreshold=evaluabilityThreshold,
            evaluabilitySurrogate=evaluabilitySurrogate,
        )

        return bestCandidates


class MaximizeDistance(AcquisitionFunction):
    """
    Maximizing distance acquisition function as described in [#]_.

    This acquisition function is used to find new sample points that maximize
    the minimum distance to previously sampled points.

    :param rtol: Minimum distance between a candidate point and the
        previously selected points relative to the domain size. Default is 1e-3.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, rtol: float = 1e-3, termination: Optional[TerminationCondition] = None, **kwargs) -> None:
        super().__init__(rtol=rtol, **kwargs)
        self.termination = termination

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        points: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Acquire n points that maximize the minimum distance to previously
        sampled points.

        :param surrogateModel: The surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param points: Points to consider for distance maximization. If None,
            use all previously sampled points in the surrogate model.
        """
        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Calculate tolerance using the tol function
        atol = self.tol(bounds)

        selectedPoints = []
        if points is None:
            currentPoints = surrogateModel.X.copy()
        else:
            currentPoints = points.copy()

        for i in range(n):
            tree = KDTree(currentPoints)

            problem = PymooProblem(
                lambda x: -tree.query(x)[0],
                bounds,
                iindex,
            )

            res = pymoo_minimize(
                problem,
                optimizer,
                seed=surrogateModel.ntrain + 1,
                verbose=False
            )
            if res.X is not None:
                newPoint = np.array([res.X[j] for j in range(len(bounds))])

                # Check if the new point is far enough from existing points
                distanceToExisting = tree.query(newPoint.reshape(1, -1))[0]
                if distanceToExisting >= atol:
                    selectedPoints.append(newPoint)
                    currentPoints = np.vstack([currentPoints, newPoint])

        return (
            np.array(selectedPoints)
            if selectedPoints
            else np.empty((0, len(bounds)))
        )


class AlternatedAcquisition(AcquisitionFunction):
    """
    Alternated acquisition function that cycles through a list of acquisition
    functions.

    The current acquisition function moves to the next in the list when the
    current one's termination criterion is met. To progress through the
    acquisition functions, the `update` method must be called after each
    optimization step. This provides the function with the current optimization
    state, allowing it to determine if the termination condition has been
    satisfied.

    :param acquisitionFuncArray: List of acquisition functions to be used in
        sequence.
    """
    def __init__(
        self,
        acquisitionFuncArray: Sequence[AcquisitionFunction],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.acquisitionFuncArray = acquisitionFuncArray
        self.idx = 0

    def optimize(
        self,
        surrogateModel: RbfModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        return self.acquisitionFuncArray[self.idx].optimize(
            surrogateModel, bounds, n, **kwargs
        )

    def update(self, out: OptimizeResult, model: Surrogate) -> None:
        self.acquisitionFuncArray[self.idx].update(out, model)

        # Alternate if the current acquisition function's termination is met
        if self.acquisitionFuncArray[self.idx].has_converged():
            self.idx = (self.idx + 1) % len(self.acquisitionFuncArray)
            self.acquisitionFuncArray[self.idx].termination.reset()
