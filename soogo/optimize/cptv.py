"""Coordinate perturbation and target value strategy."""

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

from typing import Callable, Optional
import logging
import warnings

import numpy as np
from scipy.optimize import minimize

from ..acquisition import (
    MaximizeDistance,
    MultipleAcquisition,
    TargetValueAcquisition,
    CoordinatePerturbation,
    BoundedParameter,
    FarEnoughSampleFilter,
)
from ..model import MedianLpfFilter, RbfModel
from .utils import OptimizeResult, PartialVariableFunction
from ..termination import RobustCondition, UnsuccessfulImprovement
from .surrogate_optimization import surrogate_optimization
from ..utils import report_unused_kwargs

logger = logging.getLogger(__name__)


def cptv(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[RbfModel] = None,
    acquisitionFunc: Optional[CoordinatePerturbation] = None,
    improvementTol: float = 1e-3,
    useLocalSearch: bool = False,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    seed=None,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using the coordinate
    perturbation and target value strategy.

    This is an implementation of the algorithm desribed in [#]_. The algorithm
    uses a sequence of different acquisition functions as follows:

     1. CP step: :func:`.surrogate_optimization()` with ``acquisitionFunc``.
         This step uses a coordinate-perturbation acquisition
         (:class:`.CoordinatePerturbation`) that adapts the perturbation scale
         and explores locally around promising points.

    2. TV step: :func:`.surrogate_optimization()` with a
       :class:`.TargetValueAcquisition` object.

    3. Local step (only when `useLocalSearch` is True): Runs a local
       continuous optimization with the true objective using the best point
       found so far as initial guess.

    The stopping criteria of steps 1 and 2 is related to the number of
    consecutive attempts that fail to improve the best solution by at least
    `improvementTol`. The algorithm alternates between steps 1 and 2 until there
    is a sequence (CP,TV,CP) where the individual steps do not meet the
    successful improvement tolerance. In that case, the algorithm switches to
    step 3. When the local step is finished, the algorithm goes back top step 1.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. If None is provided, a
        :class:`.RbfModel` model with median low-pass filter is used.
        On exit, if provided, the surrogate model the points used during the
        optimization.
    :param acquisitionFunc: Acquisition function for the CP step. If ``None``,
        uses :class:`.CoordinatePerturbation` configured as in Müller (2016).
    :param improvementTol: Expected improvement in the global optimum per
        iteration.
    :param useLocalSearch: If True, the algorithm will perform a continuous
        local search when a significant improvement is not found in a sequence
        of (CP,TV,CP) steps.
    :param disp: Deprecated and ignored. Configure logging instead using
        standard Python logging levels.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result.
    :param seed: Seed or random number generator.
    :return: The optimization result.

    References
    ----------
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    """
    if disp:
        warnings.warn(
            "'disp' is deprecated and ignored; use logging levels instead",
            DeprecationWarning,
            stacklevel=2,
        )

    dim = len(bounds)  # Dimension of the problem
    if dim <= 0:
        raise ValueError("bounds must define at least one dimension")

    # Random number generator
    rng = np.random.default_rng(seed)

    # Initialize optional variables
    if surrogateModel is None:
        surrogateModel = RbfModel(filter=MedianLpfFilter())
    if acquisitionFunc is None:
        acquisitionFunc = CoordinatePerturbation(
            pool_size=min(500 * dim, 5000),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            sigma=BoundedParameter(0.2, 0.2 * 0.5**6, 0.2),
            termination=RobustCondition(
                UnsuccessfulImprovement(improvementTol), max(5, dim)
            ),
            n_continuous_search=0 if useLocalSearch else 4,
            seed=rng.integers(np.iinfo(np.int32).max).item(),
        )

    tv_acquisition = MultipleAcquisition(
        (
            TargetValueAcquisition(
                cycleLength=10,
                rtol=acquisitionFunc.rtol,
                seed=rng.integers(np.iinfo(np.int32).max).item(),
            ),
            MaximizeDistance(
                rtol=acquisitionFunc.rtol,
                seed=rng.integers(np.iinfo(np.int32).max).item(),
            ),
        ),
        termination=RobustCondition(
            UnsuccessfulImprovement(improvementTol), 12
        ),
    )

    # Get index and bounds of the continuous variables
    cindex = [i for i in range(dim) if i not in surrogateModel.iindex]
    cbounds = [bounds[i] for i in cindex]

    # Initialize output
    out = OptimizeResult()
    out.x = np.full(dim, np.nan)
    out.fx = np.inf
    out.sample = np.zeros((maxeval, dim))
    out.fsample = np.zeros(maxeval)

    # do until max number of f-evals reached
    method = 0
    localSearchCounter = 0
    while out.nfev < maxeval:
        if method == 0:
            if out.nfev > 0:
                # DDS params
                acquisitionFunc.sigma.value = acquisitionFunc.sigma.max
                if acquisitionFunc.perturbation_strategy == "dycors":
                    acquisitionFunc._perturbation_probability = 1.0

                # Local search params
                acquisitionFunc.remainingCountinuousSearch = 0

                # Improvement state
                acquisitionFunc.unsuccessful_improvement.update(
                    out, surrogateModel
                )
                acquisitionFunc.unsuccessful_improvement.reset(
                    keep_data_knowledge=True
                )
                acquisitionFunc.success_count = 0
                acquisitionFunc.failure_count = 0

                # Best known point
                acquisitionFunc.best_known_x = np.copy(out.x)

                # Reset termination parameters
                termination = acquisitionFunc.termination
                if termination is not None:
                    termination.update(out, surrogateModel)
                    termination.reset(keep_data_knowledge=True)

            # Run the CP step
            out_local = surrogate_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                surrogateModel=surrogateModel,
                acquisitionFunc=acquisitionFunc,
                seed=rng.integers(np.iinfo(np.int32).max).item(),
            )
            assert isinstance(out_local.fx, float), (
                "Expected float, got %s" % type(out_local.fx)
            )

            logger.info("CP step ended after %d f evals.", out_local.nfev)

            # Switch method
            if useLocalSearch:
                if out.nfev == 0 or (
                    out.fx - out_local.fx
                ) > improvementTol * (out.fsample.max() - out.fx):
                    localSearchCounter = 0
                else:
                    localSearchCounter += 1

                if localSearchCounter >= 3:
                    method = 2
                    localSearchCounter = 0
                else:
                    method = 1
            else:
                method = 1
        elif method == 1:
            # Reset acquisition parameters
            termination = tv_acquisition.termination
            if termination is not None:
                termination.update(out, surrogateModel)
                termination.reset(keep_data_knowledge=True)

            # Run the TV step
            out_local = surrogate_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                surrogateModel=surrogateModel,
                acquisitionFunc=tv_acquisition,
                seed=rng.integers(np.iinfo(np.int32).max).item(),
            )
            assert isinstance(out_local.fx, float), (
                "Expected float, got %s" % type(out_local.fx)
            )

            logger.info("TV step ended after %d f evals.", out_local.nfev)

            # Switch method and update counter for local search
            method = 0
            if useLocalSearch:
                if out.nfev == 0 or (
                    out.fx - out_local.fx
                ) > improvementTol * (out.fsample.max() - out.fx):
                    localSearchCounter = 0
                else:
                    localSearchCounter += 1
        else:
            # Initialize output for local search
            n_loc = maxeval - out.nfev
            out_local = OptimizeResult()
            out_local.sample = np.zeros((n_loc, len(cindex)))
            out_local.fsample = np.zeros(n_loc)

            # Use a callback function to record the samples and function values
            # during the local search
            out_scipy = minimize(
                PartialVariableFunction(fun, out.x, cindex, out_local),
                out.x[cindex],
                method="Powell",
                bounds=cbounds,
                options={"maxfev": n_loc},
            )
            assert out_scipy.nfev == out_local.nfev, (
                f"Sanity check, {out_scipy.nfev} != {out_local.nfev}. Fix me!"
            )
            assert out_scipy.nfev <= n_loc, (
                f"Sanity check, {out_scipy.nfev} <= {n_loc}. We should adjust either `maxfun` or change the `method`"
            )

            # Update local output with the results of the local search
            out_local.x = out.x
            out_local.x[cindex] = out_scipy.x
            out_local.fx = out_scipy.fun
            out_local.nit = out_scipy.nit
            _sample = np.zeros((out_local.nfev, dim))
            _sample[:, cindex] = out_local.sample[: out_local.nfev]
            out_local.sample = _sample
            out_local.fsample = out_local.fsample[: out_local.nfev]

            # Update the surrogate model with the subset of samples that are
            # far enough from the existing samples.
            idx = FarEnoughSampleFilter(
                surrogateModel.X,
                acquisitionFunc.tol(bounds),
                approach="greedy",
            ).indices(out_local.sample[::-1])[::-1]
            surrogateModel.update(
                out_local.sample[idx], out_local.fsample[idx]
            )

            logger.info("Local step ended after %d f evals.", out_local.nfev)

            # Switch method
            method = 0

        # Update output
        if out_local.fx < out.fx:
            out.x[:] = out_local.x
            out.fx = out_local.fx
        out.sample[out.nfev : out.nfev + out_local.nfev, :] = out_local.sample
        out.fsample[out.nfev : out.nfev + out_local.nfev] = out_local.fsample
        out.nfev = out.nfev + out_local.nfev

        # Call the callback function
        if callback is not None:
            callback(out)

        # Update counters
        out.nit = out.nit + 1

    # Update output
    out.sample = out.sample[: out.nfev]
    out.fsample = out.fsample[: out.nfev]

    return out


def cptvl(*args, **kwargs) -> OptimizeResult:
    """Wrapper to cptv. See :func:`.cptv()`."""
    if "useLocalSearch" in kwargs:
        report_unused_kwargs(
            "cptvl", {"useLocalSearch": kwargs["useLocalSearch"]}
        )
    kwargs["useLocalSearch"] = True
    return cptv(*args, **kwargs)
