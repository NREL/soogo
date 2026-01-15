"""Coordinate perturbation and target value strategy."""

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

from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from ..acquisition import (
    MaximizeDistance,
    MultipleAcquisition,
    TargetValueAcquisition,
    CoordinatePerturbation,
    BoundedParameter,
)
from ..model import MedianLpfFilter, RbfModel
from .utils import OptimizeResult
from ..termination import RobustCondition, UnsuccessfulImprovement
from .surrogate_optimization import surrogate_optimization


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
    :param disp: If True, print information about the optimization process.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result.
    :param seed: Seed or random number generator.
    :return: The optimization result.

    References
    ----------
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Random number generator
    rng = np.random.default_rng(seed)

    # Initialize optional variables
    if surrogateModel is None:
        surrogateModel = RbfModel(filter=MedianLpfFilter())
    if acquisitionFunc is None:
        acquisitionFunc = CoordinatePerturbation(
            pool_size=min(500 * dim, 5000),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            rtol=1e-6,
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
    k = 0
    while out.nfev < maxeval:
        if method == 0:
            if out.nfev > 0:
                # DDS params
                acquisitionFunc.sigma.value = acquisitionFunc.sigma.max
                if acquisitionFunc.dynamic_perturbation_probability:
                    acquisitionFunc.perturbation_probability = 1.0

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
                disp=disp,
                seed=rng.integers(np.iinfo(np.int32).max).item(),
            )
            assert isinstance(out_local.fx, float)

            if disp:
                print("CP step ended after ", out_local.nfev, "f evals.")

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
                disp=disp,
                seed=rng.integers(np.iinfo(np.int32).max).item(),
            )
            assert isinstance(out_local.fx, float)

            if disp:
                print("TV step ended after ", out_local.nfev, "f evals.")

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

            def func_continuous_search(x):
                x_ = out.x.reshape(1, -1).copy()
                x_[0, cindex] = x
                return fun(x_)

            out_local_ = minimize(
                func_continuous_search,
                out.x[cindex],
                method="Powell",
                bounds=cbounds,
                options={"maxfev": maxeval - out.nfev},
            )
            assert out_local_.nfev <= (maxeval - out.nfev), (
                f"Sanity check, {out_local_.nfev} <= ({maxeval} - {out.nfev}). We should adjust either `maxfun` or change the `method`"
            )

            out_local = OptimizeResult()
            out_local.x = out.x.copy()
            out_local.fx = out_local_.fun
            out_local.nit = out_local_.nit
            out_local.nfev = out_local_.nfev
            out_local.sample = np.array(
                [out.x for i in range(out_local_.nfev)]
            )
            out_local.fsample = np.array(
                [out.fx for i in range(out_local_.nfev)]
            )
            out_local.x[cindex] = out_local_.x
            out_local.sample[-1, cindex] = out_local_.x
            out_local.fsample[-1] = out_local_.fun

            if out_local.fx < out.fx:
                surrogateModel.update(
                    out_local.x.reshape(1, -1), [out_local.fx]
                )

            if disp:
                print("Local step ended after ", out_local.nfev, "f evals.")

            # Switch method
            method = 0

        # Update knew
        knew = out_local.sample.shape[0]

        # Update output
        if out_local.fx < out.fx:
            out.x[:] = out_local.x
            out.fx = out_local.fx
        out.sample[k : k + knew, :] = out_local.sample
        out.fsample[k : k + knew] = out_local.fsample
        out.nfev = out.nfev + out_local.nfev

        # Call the callback function
        if callback is not None:
            callback(out)

        # Update k
        k = k + knew

        # Update counters
        out.nit = out.nit + 1

    # Update output
    out.sample.resize(k, dim)
    out.fsample.resize(k)

    return out


def cptvl(*args, **kwargs) -> OptimizeResult:
    """Wrapper to cptv. See :func:`.cptv()`."""
    if "useLocalSearch" in kwargs:
        assert kwargs["useLocalSearch"] is True, (
            "`useLocalSearch` must be True for `cptvl`."
        )
    else:
        kwargs["useLocalSearch"] = True
    return cptv(*args, **kwargs)
