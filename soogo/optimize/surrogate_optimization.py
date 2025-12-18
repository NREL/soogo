"""Core surrogate optimization routine."""

# Copyright (c) 2025 Alliance for Energy Innovation, LLC
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

import time
from typing import Callable, Optional

import numpy as np

from ..acquisition import (
    Acquisition,
    MaximizeDistance,
    MultipleAcquisition,
    TargetValueAcquisition,
)
from ..model import MedianLpfFilter, RbfModel, Surrogate
from .utils import OptimizeResult


def surrogate_optimization(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    acquisitionFunc: Optional[Acquisition] = None,
    batchSize: int = 1,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a surrogate
    model and an acquisition strategy.

    This is a more generic implementation of the RBF algorithm described in
    [#]_, using multiple ideas from [#]_ especially in what concerns
    mixed-integer optimization. Briefly, the implementation works as follows:

    1. If a surrogate model or initial sample points are not provided,
    choose the initial sample using a Symmetric Latin Hypercube design.
    Evaluate the objective function at the initial sample points.

    2. Repeat 3-8 until there are no function evaluations left.

    3. Update the surrogate model with the last sample.

    4. Acquire a new sample based on the provided acquisition function.

    5. Evaluate the objective function at the new sample.

    6. Update the optimization solution and best function value if needed.

    7. Determine if there is a significant improvement and update counters.

    8. Exit after `nFailTol` successive failures to improve the minimum.

    Mind that, when solving mixed-integer optimization, the algorithm may
    perform a continuous search whenever a significant improvement is found by
    updating an integer variable. In the continuous search mode, the algorithm
    executes step 4 only on continuous variables. The continuous search ends
    when there are no significant improvements for a number of times as in
    Müller (2016).

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. If None is provided, a
        :class:`.RbfModel` model with median low-pass filter is used.
        On exit, if provided, the surrogate model the points used during the
        optimization.
    :param acquisitionFunc: Acquisition function to be used. If None is
        provided, the :class:`.TargetValueAcquisition` is used.
    :param batchSize: Number of new sample points to be generated per iteration.
    :param improvementTol: Expected improvement in the global optimum per
        iteration.
    :param nSuccTol: Number of consecutive successes before updating the
        acquisition when necessary. A zero value means there is no need to
        update the acquisition based no the number of successes.
    :param nFailTol: Number of consecutive failures before updating the
        acquisition when necessary. A zero value means there is no need to
        update the acquisition based no the number of failures.
    :param termination: Termination condition. Possible values: "nFailTol" and
        None.
    :param performContinuousSearch: If True, the algorithm will perform a
        continuous search when a significant improvement is found by updating an
        integer variable.
    :param disp: If True, print information about the optimization process.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result.
    :return: The optimization result.

    References
    ----------
    .. [#] Björkman, M., Holmström, K. Global Optimization of Costly
        Nonconvex Functions Using Radial Basis Functions. Optimization and
        Engineering 1, 373–397 (2000). https://doi.org/10.1023/A:1011584207202
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize optional variables
    return_surrogate = True
    if surrogateModel is None:
        return_surrogate = False
        surrogateModel = RbfModel(filter=MedianLpfFilter())
    if acquisitionFunc is None:
        acquisitionFunc = MultipleAcquisition(
            (TargetValueAcquisition(), MaximizeDistance())
        )

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(surrogateModel.ntrain + maxeval, dim)

    # Initialize output
    out = OptimizeResult()
    out.init(fun, bounds, batchSize, maxeval, surrogateModel)
    out.init_best_values(surrogateModel)

    # Call the callback function
    if callback is not None:
        callback(out)

    # do until max number of f-evals reached or local min found
    xselected = np.array(out.sample[0 : out.nfev, :], copy=True)
    ySelected = np.array(out.fsample[0 : out.nfev], copy=True)
    while out.nfev < maxeval:
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % out.nfev)
            print("Best value: %f" % out.fx)

        # number of new sample points in an iteration
        batchSize = min(batchSize, maxeval - out.nfev)

        # Update surrogate model
        t0 = time.time()
        surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Acquire new sample points
        t0 = time.time()
        xselected = acquisitionFunc.optimize(
            surrogateModel, bounds, batchSize, xbest=out.x, ybest=out.fx
        )
        tf = time.time()
        if disp:
            print("Time to acquire new sample points: %f s" % (tf - t0))

        # Compute f(xselected)
        if len(xselected) > 0:
            selectedBatchSize = xselected.shape[0]
            ySelected = np.asarray(fun(xselected))
        else:
            ySelected = np.empty((0,))
            out.nit = out.nit + 1
            print(
                "Acquisition function has failed to find a new sample! "
                "Consider modifying it."
            )
            break

        # determine best one of newly sampled points
        iSelectedBest = np.argmin(ySelected).item()
        fxSelectedBest = ySelected[iSelectedBest]
        if fxSelectedBest < out.fx:
            out.x[:] = xselected[iSelectedBest, :]
            out.fx = fxSelectedBest

        # Update x, y, out.nit and out.nfev
        out.sample[out.nfev : out.nfev + selectedBatchSize, :] = xselected
        out.fsample[out.nfev : out.nfev + selectedBatchSize] = ySelected
        out.nfev = out.nfev + selectedBatchSize
        out.nit = out.nit + 1

        # Call the callback function
        if callback is not None:
            callback(out)

        # Terminate if acquisition function has converged
        acquisitionFunc.update(out, surrogateModel)
        if acquisitionFunc.has_converged():
            break

    # Update output
    out.sample.resize(out.nfev, dim)
    out.fsample.resize(out.nfev)

    # Update surrogate model if it lives outside the function scope
    if return_surrogate:
        t0 = time.time()
        surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

    return out
