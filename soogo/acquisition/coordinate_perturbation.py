"""Coordinate perturbation of the best points."""

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
from typing import Union

from .weighted_acquisition import WeightedAcquisition
from .utils import select_weighted_candidates
from ..model import Surrogate
from ..sampling import dds_sample, dds_uniform_sample
from ..optimize.result import OptimizeResult
from ..termination import UnsuccessfulImprovement
from ..utils import find_pareto_front


class BoundedParameter:
    """Class to manage a bounded parameter with increase/decrease methods.

    Use factor of 2 to increase/decrease the parameter value.

    :param value: Initial value of the parameter.
    :param min: Lower bound for the parameter.
    :param max: Upper bound for the parameter.

    .. attribute:: min

        Lower bound for the parameter.

    .. attribute:: max

        Upper bound for the parameter.

    .. attribute:: value

        Current value of the parameter.
    """

    def __init__(self, value, min=None, max=None) -> None:
        if isinstance(value, BoundedParameter):
            self.value = value.value
            self.min = value.min
            self.max = value.max
        else:
            self.value = value
            self.min = min if min is not None else value
            self.max = max if max is not None else value

        assert self.min <= self.value <= self.max, (
            "Initial value must be within bounds."
        )

    def increase(self) -> None:
        """Increase the parameter value, up to the upper bound."""
        self.value = min(2 * self.value, self.max)

    def decrease(self) -> None:
        """Decrease the parameter value, down to the lower bound."""
        self.value = max(0.5 * self.value, self.min)


class CoordinatePerturbation(WeightedAcquisition):
    """Weighted selection with coordinate perturbations around promising points.

    This acquisition extends :class:`WeightedAcquisition` by generating
    candidates via local, truncated-normal perturbations of the best-known
    points (DDS/DYCORS-style). It adapts the perturbation scale ``sigma``
    based on recent successes/failures and can enter a continuous-search mode
    over only continuous coordinates when improvements in integer variables are
    detected. See [#]_ for details.

    For multi-objective surrogates, the score uses the average predicted value
    across targets as :math:`f_s(x)`. See [#]_ for details.

    :param sampling_strategy: Sampling strategy for candidate generation.
        Currently, only ``"dds"`` (default) and ``"dds_uniform"`` are supported.
    :param sigma: Initial perturbation scale; can be a float or a
        :class:`.BoundedParameter`. Default is ``0.25`` in the range
        [0.0, 0.25].
    :param perturbation_probability: Initial probability to perturb a
        coordinate; if ``None``, a dynamic schedule (as in DYCORS) is used.
    :param int n_continuous_search: Number of iterations to remain in
        continuous-search mode after an improvement in integer variables.

    .. attribute:: sampling_strategy

        Sampling strategy for candidate generation.

    .. attribute:: sigma

        Perturbation scale used in candidate generation.

    .. attribute:: perturbation_probability

        Default probability of perturbing each coordinate
        (updated in :meth:`update()`).

    .. attribute:: dynamic_perturbation_probability

        Whether to use a dynamic schedule for the perturbation probability.

    .. attribute:: remainingCountinuousSearch

        Counter for the number of iterations remaining in continuous search
        mode.

    .. attribute:: n_continuous_search

        Maximum number of iterations in continuous search mode.

    .. attribute:: unsuccessful_improvement

        Termination condition used to track recent improvements.

    .. attribute:: success_count

        Counter for consecutive successful iterations.

    .. attribute:: failure_count

        Counter for consecutive failed iterations.

    .. attribute:: success_period

        Number of consecutive successes needed to increase ``sigma``.

    .. attribute:: best_known_x

        Best point found so far.

    References
    ----------
    .. [#] Regis, R. G., & Shoemaker, C. A. (2012). Combining radial basis
        function surrogates and dynamic coordinate search in
        high-dimensional expensive black-box optimization.
        Engineering Optimization, 45(5), 529–555.
        https://doi.org/10.1080/0305215X.2012.687731
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581–783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(
        self,
        sampling_strategy: str = "dds",
        sigma: Union[float, BoundedParameter, None] = None,
        perturbation_probability=None,
        n_continuous_search: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sampling_strategy = sampling_strategy

        # Sigma
        self.sigma = (
            BoundedParameter(0.25, min=0.0, max=0.25)
            if sigma is None
            else BoundedParameter(sigma)
        )

        # Perturbation probability for every coordinate (updated in `update()`)
        if perturbation_probability is not None:
            self.perturbation_probability = perturbation_probability
            self.dynamic_perturbation_probability = False
        else:
            self.perturbation_probability = 1.0
            self.dynamic_perturbation_probability = True

        # Continuous local search
        self.remainingCountinuousSearch = 0
        self.n_continuous_search = n_continuous_search

        # Local search updates
        self.unsuccessful_improvement = UnsuccessfulImprovement(0.001)
        self.success_count = 0
        self.failure_count = 0
        self.success_period = 3

        # Best point found so far
        self.best_known_x = None

    def tol(self, bounds) -> float:
        """Compute tolerance used to eliminate points that are too close to
        previously selected ones.

        The tolerance value is based on :attr:`rtol` and the diameter of the
        largest d-dimensional cube that can be inscribed whithin the bounds.
        Consider the region with 95% of the values on each coordinate, which has
        diameter :math:`4*sigma`.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        """
        # Consider the region with 95% of the values on each
        # coordinate, which has diameter `4*sigma`
        tol0 = super().tol(bounds)
        return tol0 * min(4 * self.sigma.value, 1.0)

    def generate_candidates(
        self, bounds, mu, prob=None, iindex: tuple[int, ...] = ()
    ):
        # Compute probability of perturbing each coordinate
        if prob is None:
            prob = self.perturbation_probability

        if self.sampling_strategy == "dds":
            return dds_sample(
                self.pool_size,
                bounds,
                probability=prob,
                mu=mu,
                sigma_ref=self.sigma.value,
                iindex=iindex,
                seed=self.rng,
            )
        elif self.sampling_strategy == "dds_uniform":
            return dds_uniform_sample(
                self.pool_size,
                bounds,
                probability=prob,
                mu=mu,
                sigma_ref=self.sigma.value,
                iindex=iindex,
                seed=self.rng,
            )
        else:
            raise ValueError(
                f"Unsupported sampling strategy '{self.sampling_strategy}'."
                " Supported strategies are 'dds' and 'dds_uniform'."
            )

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        constr_fun=None,
        perturbation_probability=None,
        xbest=None,
        **kwargs,
    ) -> np.ndarray:
        """Generate candidates via local DDS-like perturbations and select up
        to ``n`` points minimizing the weighted score.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points requested.
        :param constr_fun: Optional constraint function. Must return a vector
            (or 2D array) with non-positive values for feasible candidates.
        :param perturbation_probability: Probability to perturb each coordinate.
            If ``None``, it is computed dynamically using the DYCORS schedule.
        :param xbest: Best point so far. If ``None``, inferred from
            the surrogate’s training data (min for single-objective;
            nondominated set for multi-objective).
        :return: m-by-dim matrix with the selected points, where m <= n.
        """
        dim = len(bounds)  # Dimension of the problem
        objdim = surrogateModel.ntarget
        iindex = surrogateModel.iindex

        # Report unused kwargs
        super().report_unused_kwargs(kwargs)

        # Choose best point if not provided
        if xbest is None:
            if objdim > 1:
                xbest = surrogateModel.X[find_pareto_front(surrogateModel.Y)]
            else:
                xbest = surrogateModel.X[surrogateModel.Y.argmin()]

        # Generate sample (do local continuous search when asked)
        if self.remainingCountinuousSearch > 0:
            coord = [i for i in range(dim) if i not in iindex]
            x = np.repeat(xbest[np.newaxis, :], self.pool_size, axis=0)
            x[:, coord] = self.generate_candidates(
                [bounds[i] for i in coord],
                xbest[coord],
                perturbation_probability,
            )
        else:
            x = self.generate_candidates(
                bounds, xbest, perturbation_probability, iindex
            )

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

        # In case of continuous search, update counter
        # Keep at least one iteration in continuous search mode since it can
        # only be deactivated by `update()`.
        if self.remainingCountinuousSearch > 0:
            self.remainingCountinuousSearch = max(
                self.remainingCountinuousSearch - n, 1
            )

        return xselected

    def update(self, out: OptimizeResult, model: Surrogate) -> None:
        # Problem dimension
        dim = out.x.shape[-1]

        # Update the termination condition if it is set
        if self.termination is not None:
            self.termination.update(out, model)

        # Parameters
        failure_period = max(5, dim)

        # Check if the last sample was successful
        self.unsuccessful_improvement.update(out, model)
        recent_success = not self.unsuccessful_improvement.is_met()

        # In continuous search mode
        if self.remainingCountinuousSearch > 0:
            # In case of a successful sample, reset the counter to the maximum
            if recent_success:
                self.remainingCountinuousSearch = self.n_continuous_search
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
                    self.remainingCountinuousSearch = self.n_continuous_search

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
                if (
                    self.termination.is_met()
                    and self.sigma.value > self.sigma.min
                ):
                    self.sigma.decrease()
                    self.failure_count = 0
                    self.termination.reset(keep_data_knowledge=True)
            else:
                if (
                    self.failure_count >= failure_period
                    and self.sigma.value > self.sigma.min
                ):
                    self.sigma.decrease()
                    self.failure_count = 0

            # Check if sigma should be increased based on the successes
            if (
                self.success_count >= self.success_period
                and self.sigma.value < self.sigma.max
            ):
                self.sigma.increase()
                self.success_count = 0

        # Update the best known x
        self.best_known_x = np.copy(out.x)

        # Update perturbarion probability
        if self.dynamic_perturbation_probability:
            maxeval = len(out.x)
            if out.nfev < maxeval:
                self.perturbation_probability = min(20 / dim, 1) * (
                    1 - (log(out.nfev + 1) / log(maxeval))
                )
            else:
                self.perturbation_probability = 1.0
