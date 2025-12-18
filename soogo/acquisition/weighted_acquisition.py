"""Weighted acquisition function for surrogate-based optimization."""

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
from math import log
from scipy.spatial.distance import cdist

from .base import Acquisition
from .utils import select_weighted_candidates
from ..model import Surrogate
from ..optimize.result import OptimizeResult
from ..sampling import NormalSampler
from ..termination import UnsuccessfulImprovement
from ..utils import find_pareto_front


class WeightedAcquisition(Acquisition):
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
        *,
        constr_fun=None,
        perturbation_probability=None,
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
        :param constr_fun: Optional constraint function that returns True for
            feasible points.
        :param perturbation_probability: Probability for perturbing each
            coordinate when using DDS sampling. If None, computed dynamically
            based on iteration count.
        :param xbest: Best point so far. Used if :attr:`sampler` is an instance
            of :class:`.NormalSampler`. If not provided,
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
            if perturbation_probability is None:
                if self.maxeval > 1 and self.neval < self.maxeval:
                    prob = min(20 / dim, 1) * (
                        1 - (log(self.neval + 1) / log(self.maxeval))
                    )
                else:
                    prob = 1.0
            else:
                prob = perturbation_probability

            x = self.sampler.get_sample(
                bounds,
                iindex=iindex,
                mu=xbest,
                probability=prob,
                coord=coord,
            )
        else:
            x = self.sampler.get_sample(bounds, iindex=iindex)

        if constr_fun is not None:
            # Filter out candidates that do not satisfy the constraints
            constr_values = constr_fun(x)
            if constr_values.ndim == 1:
                feasible_idx = constr_values <= 0
            else:
                feasible_idx = np.all(constr_values <= 0, axis=1)
            x = x[feasible_idx]
            if x.shape[0] == 0:
                return np.empty((0, dim))

        # Evaluate candidates
        fx = surrogateModel(x)

        # Select best candidates
        xselected, _ = select_weighted_candidates(
            x,
            cdist(x, surrogateModel.X),
            fx,
            n,
            self.tol(bounds),
            self.weightpattern,
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
