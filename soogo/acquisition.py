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

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.special import gamma
from scipy.linalg import ldl, cholesky, solve_triangular
from scipy.optimize import minimize, differential_evolution

# Pymoo imports
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
from pymoo.core.initialization import Initialization
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.core.population import Population
from pymoo.termination.default import DefaultSingleObjectiveTermination

# Local imports
from .sampling import NormalSampler, Sampler, Mitchel91Sampler
from .rbf import RbfModel, RbfKernel
from .gp import GaussianProcess
from .problem import (
    ProblemWithConstraint,
    ProblemNoConstraint,
    MultiobjTVProblem,
    MultiobjSurrogateProblem,
    ListDuplicateElimination,
)


def expected_improvement(delta, sigma):
    """Expected Improvement function for a distribution from [#]_.

    :param delta: Difference :math:`f^*_n - \\mu_n(x)`, where :math:`f^*_n` is
        the current best function value and :math:`\\mu_n(x)` is the expected
        value for :math:`f(x)`.
    :param sigma: The standard deviation :math:`\\sigma_n(x)`.

    References
    ----------
    .. [#] Donald R. Jones, Matthias Schonlau, and William J. Welch. Efficient
        global optimization of expensive black-box functions. Journal of Global
        Optimization, 13(4):455–492, 1998.
    """
    from scipy.stats import norm

    return delta * norm.cdf(delta / sigma) + sigma * norm.pdf(delta / sigma)


def gp_expected_improvement(model: GaussianProcess, x, ybest):
    mu, sigma = model(x)
    return expected_improvement(ybest - mu, sigma)


def find_pareto_front(fx, iStart: int = 0) -> list:
    """Find the Pareto front given a set of points in the target space.

    :param fx: List with n points in the m-dimensional target space.
    :param iStart: Points from 0 to iStart - 1 are already known to be in the
        Pareto front.
    :return: Indices of the points in the Pareto front.
    """
    pareto = [True] * len(fx)
    for i in range(iStart, len(fx)):
        for j in range(i):
            if pareto[j]:
                if all(fx[i] <= fx[j]) and any(fx[i] < fx[j]):
                    # x[i] dominates x[j]
                    pareto[j] = False
                elif all(fx[j] <= fx[i]) and any(fx[j] < fx[i]):
                    # x[j] dominates x[i]
                    # No need to continue checking, otherwise the previous
                    # iteration was not a balid Pareto front
                    pareto[i] = False
                    break
    return [i for i in range(len(fx)) if pareto[i]]


class AcquisitionFunction:
    """Base class for acquisition functions.

    This an abstract class. Subclasses must implement the method
    :meth:`acquire()`.

    Acquisition functions are strategies to propose new sample points to a
    surrogate. The acquisition functions here are modeled as objects with the
    goals of adding states to the learning process. Moreover, this design
    enables the definition of the :meth:`acquire()` method with a similar API
    when we compare different acquisition strategies.
    """

    def __init__(self) -> None:
        pass

    def acquire(
        self,
        surrogateModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Propose a maximum of n new sample points to improve the surrogate.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired, or maximum requested number.
        :return: m-by-dim matrix with the selected points, where m <= n.
        """
        raise NotImplementedError


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
    :param rtol: Description
        Stored in :attr:`rtol`.
    :param maxeval: Description
        Stored in :attr:`maxeval`.

    .. attribute:: neval

        Number of evaluations done so far. Used and updated in
        :meth:`acquire()`.

    .. attribute:: sampler

        Sampler to generate candidate points. Used in :meth:`acquire()`.

    .. attribute:: weightpattern

        Weight(s) `w` to be used in the score. This is a circular list that is
        rotated every time :meth:`acquire()` is called.

    .. attribute:: rtol

        Tolerance value for excluding candidates that are too close to
        current sample points. This value is used to compute the final
        tolerance in :meth:`tol()`.

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
        rtol: float = 1e-6,
        maxeval: int = 0,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        if weightpattern is None:
            self.weightpattern = [0.2, 0.4, 0.6, 0.9, 0.95, 1]
        elif hasattr(weightpattern, "__len__"):
            self.weightpattern = list(weightpattern)
        else:
            self.weightpattern = [weightpattern]
        self.rtol = rtol
        self.maxeval = maxeval
        self.neval = 0

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
            scaleddist = (maxdist - dist) / (maxdist - mindist)

        # Compute weighted score for all candidates
        score = weight * scaledvalue + (1 - weight) * scaleddist

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
        tol0 = (
            self.rtol
            * np.sqrt(len(bounds))
            * np.min([abs(b[1] - b[0]) for b in bounds])
        )
        if isinstance(self.sampler, NormalSampler):
            # Consider the region with 95% of the values on each
            # coordinate, which has diameter `4*sigma`
            return tol0 * min(4 * self.sampler.sigma, 1.0)
        else:
            return tol0

    def acquire(
        self,
        surrogateModel,
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
        :param bool countinuousSearch:
            If True,
            optimize over the continuous variables only. Used if :attr:`sampler`
            is an instance of :class:`soogo.sampling.NormalSampler`.
        :return: m-by-dim matrix with the selected points, where m <= n.
        """
        dim = len(bounds)  # Dimension of the problem

        # Check if surrogateModel is a list of models
        listOfSurrogates = hasattr(surrogateModel, "__len__")
        iindex = (
            surrogateModel[0].iindex
            if listOfSurrogates
            else surrogateModel.iindex
        )

        # Generate candidates
        if isinstance(self.sampler, NormalSampler):
            if "xbest" in kwargs:
                xbest = kwargs["xbest"]
            elif listOfSurrogates:
                xbest = surrogateModel.samples()[
                    find_pareto_front(surrogateModel.ytrain())
                ]
            else:
                xbest = surrogateModel.samples()[
                    surrogateModel.ytrain().argmin()
                ]

            # Do local continuous search when asked
            if "countinuousSearch" in kwargs and kwargs["countinuousSearch"]:
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
        nCand = x.shape[0]

        # Evaluate candidates
        if not listOfSurrogates:
            sample = surrogateModel.xtrain()
            fx, _ = surrogateModel(x)
        else:
            sample = surrogateModel[0].xtrain()
            objdim = len(surrogateModel)
            fx = np.empty((nCand, objdim))
            for i in range(objdim):
                fx[:, i], _ = surrogateModel[i](x)

        # Select best candidates
        xselected, _ = self.minimize_weightedavg_fx_distx(
            x, cdist(x, sample), fx, n, self.tol(bounds)
        )
        n = xselected.shape[0]

        # Rotate weight pattern
        self.weightpattern[:] = (
            self.weightpattern[n % len(self.weightpattern) :]
            + self.weightpattern[: n % len(self.weightpattern)]
        )

        # Update number of evaluations
        self.neval += n

        return xselected


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

    :param optimizer: Single-objective optimizer. If None, use MixedVariableGA
        from pymoo.
    :param rtol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.
    :param cycleLength: Length of the global search cycle. Stored in
        :attr:`cycleLength`.

    .. attribute:: optimizer

        Single-objective optimizer.

    .. attribute:: rtol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    .. attribute:: cycleLength

        Length of the global search cycle to be used in :meth:`acquire()`.

    .. attribute:: _cycle

        Internal counter of cycles. The value to be used in the next call of
        :meth:`acquire()`.

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

    def __init__(
        self, optimizer=None, rtol: float = 1e-6, cycleLength: int = 6
    ) -> None:
        self._cycle = 0
        self.cycleLength = cycleLength
        self.rtol = rtol
        self.optimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
                termination=DefaultSingleObjectiveTermination(
                    xtol=rtol, period=3
                ),
            )
            if optimizer is None
            else optimizer
        )

    @staticmethod
    def mu_measure(surrogate: RbfModel, x: np.ndarray, LDLt):
        """Compute the value of abs(mu) for an RBF model.

        The mu measure was first defined in [#]_ with suggestions of usage for
        global optimization with RBF functions. In [#]_, the authors detail the
        strategy to make the evaluations computationally viable.

        The current
        implementation, uses a different strategy than that from Björkman and
        Holmström (2000), where a single LDLt factorization is used instead of
        the QR and Cholesky factorizations. The new algorithm's performs 10
        times less operations than the former. Like the former, the new
        algorithm is also able to use high-intensity linear algebra operations
        when the routine is called with multiple points :math:`x` are evaluated
        at once.

        :param surrogate: RBF surrogate model.
        :param x: Possible point to be added to the surrogate model.
        :param LDLt: LDLt factorization of the matrix A as returned by the
            function scipy.linalg.ldl.

        References
        ----------
        .. [#] Gutmann, HM. A Radial Basis Function Method for Global
            Optimization. Journal of Global Optimization 19, 201–227 (2001).
            https://doi.org/10.1023/A:1011255519438
        .. [#] Björkman, M., Holmström, K. Global Optimization of Costly
            Nonconvex Functions Using Radial Basis Functions. Optimization and
            Engineering 1, 373–397 (2000). https://doi.org/10.1023/A:1011584207202
        """
        # compute rbf value of the new point x
        xdist = cdist(surrogate.xtrain(), x)
        newCols = np.concatenate(
            (np.asarray(surrogate.phi(xdist)), surrogate.pbasis(x).T), axis=0
        )

        # Get the L factor, the block-diagonal matrix D, and the permutation
        # vector p
        ptL, D, p = LDLt
        L = ptL[p]

        # 0. Permute the new terms
        newCols = newCols[p]

        # 1. Solve P [a;b] = L (D l) for (D l)
        Dl = solve_triangular(
            L,
            newCols,
            lower=True,
            unit_diagonal=True,
            # check_finite=False,
            overwrite_b=True,
        )

        # 2. Compute l := inv(D) (Dl)
        ell = Dl.copy()
        i = 0
        while i < len(ell) - 1:
            if D[i + 1, i] == 0:
                # Invert block of size 1x1
                ell[i] /= D[i, i]
                i += 1
            else:
                # Invert block of size 2x2
                det = D[i, i] * D[i + 1, i + 1] - D[i, i + 1] ** 2
                ell[i], ell[i + 1] = (
                    (ell[i] * D[i + 1, i + 1] - ell[i + 1] * D[i, i + 1])
                    / det,
                    (ell[i + 1] * D[i, i] - ell[i] * D[i, i + 1]) / det,
                )
                i += 2
        if i == len(ell) - 1:
            # Invert last block of size 1x1
            ell[i] /= D[i, i]

        # 3. d = \phi(0) - l^T D l and \mu = 1/d
        d = surrogate.phi(0) - (ell * Dl).sum(axis=0)
        mu = np.where(d != 0, 1 / d, np.inf)

        # Get the absolute value of mu
        if surrogate.kernel == RbfKernel.LINEAR:
            mu = -mu
        elif surrogate.kernel not in (RbfKernel.CUBIC, RbfKernel.THINPLATE):
            raise ValueError("Unknown RBF type")

        # Return huge value if the matrix is ill-conditioned
        mu = np.where(mu <= 0, np.inf, mu)

        return mu

    @staticmethod
    def bumpiness_measure(
        surrogate: RbfModel, x: np.ndarray, target, LDLt, target_range=1.0
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
        :param LDLt: LDLt factorization of the matrix A as returned by the
            function scipy.linalg.ldl.
        :param target_range: Known range in the target space. Used to scale
            the function values to avoid overflow.
        """
        absmu = TargetValueAcquisition.mu_measure(surrogate, x, LDLt)
        assert all(
            absmu > 0
        )  # if absmu == 0, the linear system in the surrogate model singular

        # predict RBF value of x
        yhat, _ = surrogate(x)

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

    def acquire(
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

        # Create a KDTree with the current training points
        tree = KDTree(surrogateModel.xtrain())
        tol = self.rtol * np.min([abs(b[1] - b[0]) for b in bounds])

        # Compute fbounds of the surrogate. Use the filter as suggested by
        # Björkman and Holmström (2000)
        fbounds = [
            surrogateModel.ytrain().min(),
            surrogateModel.filter(surrogateModel.ytrain()).max(),
        ]
        target_range = fbounds[1] - fbounds[0]
        if target_range == 0:
            target_range = 1

        # Allocate variables a priori targeting batched sampling
        x = np.empty((n, dim))
        LDLt = None
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
                if LDLt is None:
                    LDLt = ldl(surrogateModel.get_RBFmatrix())
                problem = ProblemNoConstraint(
                    lambda x: TargetValueAcquisition.mu_measure(
                        surrogateModel, x, LDLt
                    ),
                    bounds,
                    surrogateModel.iindex,
                )

                res = pymoo_minimize(
                    problem,
                    self.optimizer,
                    seed=surrogateModel.ntrain(),
                    verbose=False,
                )

                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])

            elif (
                1 <= sample_stage <= self.cycleLength
            ):  # cycle step global search
                # find min of surrogate model
                if f_rbf is None:
                    problem = ProblemNoConstraint(
                        lambda x: surrogateModel(x)[0],
                        bounds,
                        surrogateModel.iindex,
                    )
                    res = pymoo_minimize(
                        problem,
                        self.optimizer,
                        seed=surrogateModel.ntrain(),
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
                if LDLt is None:
                    LDLt = ldl(surrogateModel.get_RBFmatrix())
                problem = ProblemNoConstraint(
                    lambda x: TargetValueAcquisition.bumpiness_measure(
                        surrogateModel, x, f_target, LDLt, target_range
                    ),
                    bounds,
                    surrogateModel.iindex,
                )

                res = pymoo_minimize(
                    problem,
                    self.optimizer,
                    seed=surrogateModel.ntrain(),
                    verbose=False,
                )

                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])
            else:  # cycle step local search
                # find the minimum of RBF surface
                if f_rbf is None:
                    problem = ProblemNoConstraint(
                        lambda x: surrogateModel(x)[0],
                        bounds,
                        surrogateModel.iindex,
                    )
                    res = pymoo_minimize(
                        problem,
                        self.optimizer,
                        seed=surrogateModel.ntrain(),
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
                    if LDLt is None:
                        LDLt = ldl(surrogateModel.get_RBFmatrix())
                    problem = ProblemNoConstraint(
                        lambda x: TargetValueAcquisition.bumpiness_measure(
                            surrogateModel, x, f_target, LDLt, target_range
                        ),
                        bounds,
                        surrogateModel.iindex,
                    )

                    res = pymoo_minimize(
                        problem,
                        self.optimizer,
                        seed=surrogateModel.ntrain(),
                        verbose=False,
                    )

                    assert res.X is not None
                    xselected = np.asarray([res.X[i] for i in range(dim)])

            # Replace points that are too close to current sample
            current_sample = np.concatenate(
                (surrogateModel.xtrain(), x[0:i]), axis=0
            )
            while np.any(tree.query(xselected)[0] < tol) or (
                i > 0 and cdist(xselected.reshape(1, -1), x[0:i]).min() < tol
            ):
                # the selected point is too close to already evaluated point
                # randomly select point from variable domain
                xselected = Mitchel91Sampler(1).get_mitchel91_sample(
                    bounds,
                    iindex=surrogateModel.iindex,
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
    :param rtol: Tolerance for excluding points that are too close to each other
        from the new sample.

    .. attribute:: sampler

        Sampler to generate candidate points.

    .. attribute:: rtol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    .. [#] Rinnooy Kan, A.H.G., Timmer, G.T. Stochastic global optimization
        methods part II: Multi level methods. Mathematical Programming 39, 57–78
        (1987). https://doi.org/10.1007/BF02592071
    """

    def __init__(self, nCand: int, rtol: float = 1e-3) -> None:
        self.sampler = Sampler(nCand)
        self.rtol = rtol

    def acquire(
        self,
        surrogateModel,
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
        tree = KDTree(surrogateModel.xtrain())
        tol = self.rtol * np.min([abs(b[1] - b[0]) for b in bounds])

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
            fcand[iStart:iEnd], _ = surrogateModel(candidates[iStart:iEnd, :])
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
                    return surrogateModel(x_)[0]

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

                if tree.n == 0 or tree.query(xi)[0] >= tol:
                    selected[k, :] = xi
                    k += 1
                    if k == n:
                        break
                    else:
                        tree = KDTree(
                            np.concatenate(
                                (surrogateModel.xtrain(), selected[0:k, :]),
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
                current_sample=surrogateModel.xtrain(),
            )
            while tree.query(selected)[0] < tol:
                selected = singleCandSampler.get_mitchel91_sample(
                    bounds,
                    iindex=surrogateModel.iindex,
                    current_sample=surrogateModel.xtrain(),
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

    :param mooptimizer: Multi-objective optimizer. If None, use MixedVariableGA
        from pymoo with RankAndCrowding survival strategy.
    :param oldTV: Old target values to be avoided in the acquisition.
        Copied to :attr:`oldTV`.

    .. attribute:: mooptimizer

        Multi-objective optimizer used in the step 2 of the algorithm.

    .. attribute:: oldTV

        Old target values to be avoided in the acquisition of step 1.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(
        self,
        mooptimizer=None,
        oldTV=(),
    ) -> None:
        self.mooptimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
                survival=RankAndCrowding(),
            )
            if mooptimizer is None
            else mooptimizer
        )
        self.oldTV = np.array(oldTV)

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

        # Create a surrogate model for the Pareto front in the objective space
        paretoModel = RbfModel(RbfKernel.LINEAR)
        k = np.random.choice(objdim)
        paretoModel.update(
            np.array([paretoFront[:, i] for i in range(objdim) if i != k]).T,
            paretoFront[:, k],
        )
        dim = paretoModel.dim()

        # Bounds in the pareto sample
        xParetoLow = np.min(paretoModel.xtrain(), axis=0)
        xParetoHigh = np.max(paretoModel.xtrain(), axis=0)
        boundsPareto = [(xParetoLow[i], xParetoHigh[i]) for i in range(dim)]

        # Minimum of delta_f maximizes the distance inside the Pareto front
        tree = KDTree(
            np.concatenate(
                (paretoFront, self.oldTV.reshape(-1, objdim)), axis=0
            )
        )

        def delta_f(tau):
            tauk, _ = paretoModel(tau)
            _tau = np.concatenate((tau[0:k], tauk, tau[k:]))
            return -tree.query(_tau)[0]

        # Minimize delta_f
        res = differential_evolution(delta_f, boundsPareto)
        tauk, _ = paretoModel(res.x)
        tau = np.concatenate((res.x[0:k], tauk, res.x[k:]))

        return tau

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        *,
        nondominated=(),
        paretoFront=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        Perform n attempts to find n points to fill gaps in the Pareto front.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param nondominated: Nondominated set in the objective space.
        :param paretoFront: Pareto front in the objective space. If not
            provided, use the surrogate to compute it.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = len(surrogateModels)

        if len(paretoFront) == 0:
            paretoFront = np.array(
                [s.ytrain().flatten() for s in surrogateModels]
            ).T
            paretoFrontIdx = find_pareto_front(paretoFront)
            paretoFront = paretoFront[paretoFrontIdx]
            nondominated = surrogateModels[0].xtrain()[paretoFrontIdx]

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
                Xinit = np.array(
                    [{i: x[i] for i in range(dim)} for x in nondominated]
                )
                self.mooptimizer.initialization = Initialization(
                    Population.new("X", Xinit),
                    repair=self.mooptimizer.repair,
                    eliminate_duplicates=self.mooptimizer.eliminate_duplicates,
                )

            # Find the Pareto-optimal solution set that minimizes dist(s(x),tau).
            # For discontinuous Pareto fronts in the original problem, such set
            # may not exist, or it may be too far from the target value.
            multiobjTVProblem = MultiobjTVProblem(surrogateModels, tau, bounds)
            res = pymoo_minimize(
                multiobjTVProblem,
                self.mooptimizer,
                seed=len(paretoFront),
                verbose=False,
            )

            # If the Pareto-optimal solution set exists, define the sample point
            # that minimizes the L1 distance to the target value
            if res.X is not None:
                # Save X into an array
                newX = np.array([[x[i] for i in range(dim)] for x in res.X])

                # Transform the values of the optimization into a matrix
                sx = np.array(
                    [s(newX)[0].flatten() for s in surrogateModels]
                ).T

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

    :param optimizer: Single-objective optimizer. If None, use MixedVariableGA
        from pymoo.
    :param rtol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.

    .. attribute:: optimizer

        Single-objective optimizer.

    .. attribute:: rtol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, optimizer=None, rtol=1e-6) -> None:
        self.optimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
            )
            if optimizer is None
            else optimizer
        )
        self.rtol = rtol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points at most, where k <= n.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = len(surrogateModels)
        iindex = surrogateModels[0].iindex

        # Find endpoints of the Pareto front
        endpoints = np.empty((objdim, dim))
        for i in range(objdim):
            minimumPointProblem = ProblemNoConstraint(
                lambda x: surrogateModels[i](x)[0], bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem,
                self.optimizer,
                seed=surrogateModels[0].ntrain(),
                verbose=False,
            )
            assert res.X is not None
            for j in range(dim):
                endpoints[i, j] = res.X[j]

        # Create KDTree with the already evaluated points
        tree = KDTree(surrogateModels[0].xtrain())
        tol = self.rtol * np.min([abs(b[1] - b[0]) for b in bounds])

        # Discard points that are too close to previously sampled points.
        distNeighbor = tree.query(endpoints)[0]
        endpoints = endpoints[distNeighbor >= tol, :]

        # Discard points that are too close to eachother
        if len(endpoints) > 0:
            selectedIdx = [0]
            for i in range(1, len(endpoints)):
                if (
                    cdist(
                        endpoints[i, :].reshape(1, -1),
                        endpoints[selectedIdx, :],
                    ).min()
                    >= tol
                ):
                    selectedIdx.append(i)
            endpoints = endpoints[selectedIdx, :]

        # Should all points be discarded, which may happen if the minima of
        # the surrogate surfaces do not change between iterations, we
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of sample points
        if endpoints.size == 0:
            minimumPointProblem = ProblemNoConstraint(
                lambda x: -tree.query(x)[0], bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem,
                self.optimizer,
                verbose=False,
                seed=surrogateModels[0].ntrain() + 1,
            )
            assert res.X is not None
            endpoints = np.empty((1, dim))
            for j in range(dim):
                endpoints[0, j] = res.X[j]

        # Return a maximum of n points
        return endpoints[:n, :]


class MinimizeMOSurrogate(AcquisitionFunction):
    """Obtain pareto-optimal sample points for the multi-objective surrogate
    model.

    :param mooptimizer: Multi-objective optimizer. If None, use MixedVariableGA
        from pymoo with RankAndCrowding survival strategy.
    :param rtol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.

    .. attribute:: mooptimizer

        Multi-objective optimizer for the surrogate optimization problem.

    .. attribute:: rtol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    """

    def __init__(self, mooptimizer=None, rtol=1e-6) -> None:
        self.mooptimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
                survival=RankAndCrowding(),
            )
            if mooptimizer is None
            else mooptimizer
        )
        self.rtol = rtol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired. If n is zero, use all
            points in the Pareto front.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)

        # Solve the surrogate multiobjective problem
        multiobjSurrogateProblem = MultiobjSurrogateProblem(
            surrogateModels, bounds
        )
        res = pymoo_minimize(
            multiobjSurrogateProblem,
            self.mooptimizer,
            seed=surrogateModels[0].ntrain(),
            verbose=False,
        )

        # If the Pareto-optimal solution set exists, randomly select n
        # points from the Pareto front
        if res.X is not None:
            bestCandidates = np.array(
                [[x[i] for i in range(dim)] for x in res.X]
            )

            # Create tolerance based on smallest variable length
            tol = self.rtol * np.min([abs(b[1] - b[0]) for b in bounds])

            # Discard points that are too close to previously sampled points.
            distNeighbor = cdist(
                bestCandidates, surrogateModels[0].xtrain()
            ).min(axis=1)
            bestCandidates = bestCandidates[distNeighbor >= tol, :]

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
                    >= tol
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

    def __init__(self, acquisitionFunc: WeightedAcquisition) -> None:
        self.acquisitionFunc = acquisitionFunc
        assert isinstance(self.acquisitionFunc.sampler, NormalSampler)

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        *,
        nondominated=(),
        paretoFront=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :param nondominated: Nondominated set in the objective space.
        :param paretoFront: Pareto front in the objective space.
        """
        dim = len(bounds)
        tol = self.acquisitionFunc.tol(bounds)
        assert isinstance(self.acquisitionFunc.sampler, NormalSampler)

        # Find a collection of points that are close to the Pareto front
        bestCandidates = np.empty((0, dim))
        for ndpoint in nondominated:
            x = self.acquisitionFunc.acquire(
                surrogateModels,
                bounds,
                1,
                xbest=ndpoint,
            )
            # Choose points that are not too close to previously selected points
            if bestCandidates.size == 0:
                if x.size > 0:
                    bestCandidates = x.reshape(1, -1)
            else:
                distNeighborOfx = cdist(x, bestCandidates).min()
                if distNeighborOfx >= tol:
                    bestCandidates = np.concatenate(
                        (bestCandidates, x), axis=0
                    )

        # Return if no point was found
        if bestCandidates.size == 0:
            return bestCandidates

        # Eliminate points predicted to be dominated
        fnondominatedAndBestCandidates = np.concatenate(
            (
                paretoFront,
                np.array([s(bestCandidates)[0] for s in surrogateModels]).T,
            ),
            axis=0,
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

    This acquisition function is only able to acuire 1 point at a time.

    :param fun: Objective function. Stored in :attr:`fun`.
    :param optimizer: Single-objective optimizer. If None, use MixedVariableGA
        from pymoo.
    :param rtol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.

    .. attribute:: fun

        Objective function.

    .. attribute:: optimizer

        Single-objective optimizer.

    .. attribute:: rtol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Juliane Mueller and Joshua D. Woodbury. GOSAC: global optimization
        with surrogate approximation of constraints.
        J Glob Optim, 69:117-136, 2017.
        https://doi.org/10.1007/s10898-017-0496-y
    """

    def __init__(self, fun, optimizer=None, rtol: float = 1e-6):
        self.fun = fun
        self.optimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
            )
            if optimizer is None
            else optimizer
        )
        self.rtol = rtol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire 1 point.

        :param surrogateModels: List of surrogate models for the constraints.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Unused.
        :return: 1-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        gdim = len(surrogateModels)
        iindex = surrogateModels[0].iindex
        assert n == 1

        # Create KDTree with previously evaluated points
        tree = KDTree(surrogateModels[0].xtrain())
        tol = self.rtol * np.min([abs(b[1] - b[0]) for b in bounds])

        cheapProblem = ProblemWithConstraint(
            self.fun,
            lambda x: np.transpose(
                [surrogateModels[i](x)[0] for i in range(gdim)]
            ),
            bounds,
            iindex,
            n_ieq_constr=gdim,
        )
        res = pymoo_minimize(
            cheapProblem,
            self.optimizer,
            seed=surrogateModels[0].ntrain(),
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
            if tree.query(xnew)[0] < tol:
                isGoodCandidate = False

        if not isGoodCandidate:
            minimumPointProblem = ProblemNoConstraint(
                lambda x: -tree.query(x)[0], bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem,
                self.optimizer,
                seed=surrogateModels[0].ntrain() + 1,
                verbose=False,
            )
            assert res.X is not None
            xnew = np.asarray([[res.X[i] for i in range(dim)]])

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

    def __init__(self, sampler=None, avoid_clusters: bool = True) -> None:
        super().__init__()
        self.sampler = Sampler(0) if sampler is None else sampler
        self.avoid_clusters = avoid_clusters

    def acquire(
        self, surrogateModel, bounds, n: int = 1, *, ybest=None
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
        assert len(surrogateModel.get_iindex()) == 0
        if n == 0:
            return np.empty((0, len(bounds)))

        xbest = None
        if ybest is None:
            # Compute an estimate for ybest using the surrogate.
            res = differential_evolution(
                lambda x: surrogateModel(np.asarray([x]))[0], bounds
            )
            ybest = res.fun
            if res.success:
                xbest = res.x

        # Use the point that maximizes the EI
        res = differential_evolution(
            lambda x: -gp_expected_improvement(
                surrogateModel, np.asarray([x]), ybest
            ),
            bounds,
        )
        xs = res.x if res.success else None

        # Returns xs if n == 1
        if res.success and n == 1:
            return np.asarray([xs])

        # Generate the complete pool of candidates
        if isinstance(self.sampler, Mitchel91Sampler):
            current_sample = surrogateModel.xtrain()
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
        eiCand = gp_expected_improvement(surrogateModel, x, ybest)

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
