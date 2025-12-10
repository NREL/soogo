"""Utility functions for Soogo."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC

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
    "Weslley S. Pereira",
    "Byron Selvage",
]
__authors__ = [
    "Weslley S. Pereira",
    "Byron Selvage",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Weslley S. Pereira",
    "Byron Selvage",
]
__deprecated__ = False

from scipy.spatial.distance import cdist
from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from .sampling import Sampler
from .model.base import Surrogate


@dataclass
class OptimizeResult:
    """Optimization result for the global optimizers provided by this
    package."""

    x: Optional[np.ndarray] = None  #: Best sample point found so far
    fx: Union[float, np.ndarray, None] = None  #: Best objective function value
    nit: int = 0  #: Number of active learning iterations
    nfev: int = 0  #: Number of function evaluations taken
    sample: Optional[np.ndarray] = None  #: n-by-dim matrix with all n samples
    fsample: Optional[np.ndarray] = None  #: Vector with all n objective values
    nobj: int = 1  #: Number of objective function targets

    def init(
        self,
        fun,
        bounds,
        mineval: int,
        maxeval: int,
        surrogateModel: Surrogate,
        ntarget: int = 1,
    ) -> None:
        """Initialize :attr:`nfev` and :attr:`sample` and :attr:`fsample` with
        data about the optimization that is starting.

        This routine calls the objective function :attr:`nfev` times.

        By default, all targets are considered to be used in the objective. If
        that is not the case, set `nobj` after calling this function.

        :param fun: The objective function to be minimized.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param mineval: Minimum number of function evaluations to build the
            surrogate model.
        :param maxeval: Maximum number of function evaluations.
        :param surrogateModel: Surrogate model to be used.
        :param ntarget: Number of target dimensions. Default is 1.
        """
        dim = len(bounds)  # Dimension of the problem
        assert dim > 0

        # Local variables
        m0 = surrogateModel.ntrain  # Number of initial sample points
        m_for_surrogate = surrogateModel.min_design_space_size(
            dim
        )  # Smallest sample for a valid surrogate
        iindex = surrogateModel.iindex  # Integer design variables

        # Initialize sample array in this object
        self.sample = np.empty((maxeval, dim))
        self.sample[:] = np.nan

        # If the surrogate is empty and no initial sample was given
        m = 0
        if m0 == 0:
            # Create a new sample with SLHD
            m = min(maxeval, max(mineval, 2 * m_for_surrogate))
            self.sample[0:m] = Sampler(m).get_slhd_sample(
                bounds, iindex=iindex
            )
            if m >= 2 * m_for_surrogate:
                count = 0
                while not surrogateModel.check_initial_design(
                    self.sample[0:m]
                ):
                    self.sample[0:m] = Sampler(m).get_slhd_sample(
                        bounds, iindex=iindex
                    )
                    count += 1
                    if count > 100:
                        raise RuntimeError(
                            "Cannot create valid initial design"
                        )

        # Initialize fsample and nobj
        if m0 == 0:
            # Compute f(sample)
            fsample = np.array(fun(self.sample[0:m]))
            self.nfev += m
            self.nobj = fsample.shape[1] if fsample.ndim > 1 else 1

            # Store the function values
            self.fsample = np.empty(
                maxeval if self.nobj <= 1 else (maxeval, self.nobj)
            )
            self.fsample[0:m] = fsample
        else:
            self.nobj = max(ntarget, surrogateModel.ntarget)
            self.fsample = np.empty(
                maxeval if self.nobj <= 1 else (maxeval, self.nobj)
            )
        self.fsample[m:] = np.nan

    def init_best_values(
        self, surrogateModel: Optional[Surrogate] = None
    ) -> None:
        """Initialize :attr:`x` and :attr:`fx` based on the best values obtained
        so far.

        :param surrogateModel: Surrogate model.
        """
        # Initialize self.x and self.fx
        assert self.sample is not None
        assert self.fsample is not None
        m = self.nfev

        if surrogateModel is not None and surrogateModel.ntrain > 0:
            combined_x = np.concatenate(
                (self.sample[0:m], surrogateModel.X), axis=0
            )
            if self.fsample.ndim == 1:
                combined_y = np.concatenate(
                    (self.fsample[0:m], surrogateModel.Y), axis=0
                )
            else:
                nrows = surrogateModel.ntrain
                ncols = self.fsample.shape[1]
                combined_y = np.concatenate(
                    (self.fsample[0:m], np.empty((nrows, ncols))), axis=0
                )
                combined_y[m:, 0 : surrogateModel.ntarget] = surrogateModel.Y
        else:
            combined_x = self.sample[0:m]
            combined_y = self.fsample[0:m]

        if self.nobj == 1:
            if combined_y.ndim == 1:
                iBest = np.argmin(combined_y).item()
                self.fx = combined_y[iBest].item()
            else:
                iBest = np.argmin(combined_y[:, 0]).item()
                self.fx = combined_y[iBest].copy()

            self.x = combined_x[iBest].copy()
        else:
            iPareto = find_pareto_front(combined_y[:, 0 : self.nobj])
            self.x = combined_x[iPareto].copy()
            self.fx = combined_y[iPareto].copy()


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


def gp_expected_improvement(delta, sigma):
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


def uncertainty_score(candidates, points, fvals, k=3):
    """
    Calculate the uncertainty (distance and fitness value criterion)
    score as defined in _[#].

    :param candidates: The candidate points to find the scores for.
    :param points: The set of already evaluated points.
    :param fvals: The set of corresponding function values.
    :param k: The number of nearest neighbors to consider in
        the uncertainty calculation. Default is 3.

    :return: The uncertainty score for each candidate point.

    References
    ----------
    .. [#] Li, F., Shen, W., Cai, X., Gao, L., & Gary Wang, G. 2020; A fast
        surrogate-assisted particle swarm optimization algorithm for computationally
        expensive problems. Applied Soft Computing, 92, 106303.
        https://doi.org/10.1016/j.asoc.2020.106303
    """
    candidates = np.asarray(candidates)
    points = np.asarray(points)
    fvals = np.asarray(fvals)

    # Compute all distances
    dists = cdist(candidates, points)

    # For each candidate, get indices of k nearest points
    nearestIndices = np.argsort(dists, axis=1)[:, :k]

    # Extract distances and function values for k nearest points
    nCandidates = candidates.shape[0]
    distances = np.zeros((nCandidates, k))
    functionValues = np.zeros((nCandidates, k))

    for i in range(nCandidates):
        indices = nearestIndices[i]
        distances[i] = dists[i, indices]
        functionValues[i] = fvals[indices]

    # Calculate the mean dist and std of k nearest
    distMean = np.mean(distances, axis=1)
    sigma = np.std(functionValues, axis=1)

    # Normalize
    distMean /= np.sum(distMean)
    sigma /= np.sum(sigma)

    # Calculate scaled dist to nearest neighbor
    nearestScaled = 5 * distances[:, 0] / np.sum(distances[:, 0])

    # Calculate Sigmoid function value
    sigmoid = 1 / (1 + np.exp(-nearestScaled)) - 0.5

    # Calculate the final scores
    scores = sigmoid * (distMean + sigma)

    return scores
