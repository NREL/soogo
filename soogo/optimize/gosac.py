"""GOSAC constrained optimization routine."""

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

import time
from typing import Callable, Optional

import numpy as np

from ..acquisition import GosacSample, MaximizeDistance, MinimizeMOSurrogate
from ..model import RbfModel, Surrogate
from .utils import OptimizeResult


def gosac(
    fun,
    gfun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
):
    """Minimize a scalar function of one or more variables subject to
    constraints.

    The surrogate models are used to approximate the constraints. The objective
    function is assumed to be cheap to evaluate, while the constraints are
    assumed to be expensive to evaluate.

    This method is based on [#]_.

    :param fun: The objective function to be minimized.
    :param gfun: The constraint function to be minimized. The
        constraints must be formulated as g(x) <= 0.
    :param bounds: List with the limits [x_min,x_max] of each direction
        x in the search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used for the
        constraints. If None is provided, a :class:`.RbfModel` model is
        used.
    :param disp: If True, print information about the optimization
        process. The default is False.
    :param callback: If provided, the callback function will be called
        after each iteration with the current optimization result. The
        default is None.
    :return: The optimization result.

    References
    ----------
    .. [#] Juliane Müller and Joshua D. Woodbury. 2017. GOSAC: global
        optimization with surrogate approximation of constraints. J. of Global
        Optimization 69, 1 (September 2017), 117–136.
        https://doi.org/10.1007/s10898-017-0496-y
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize optional variables
    return_surrogate = True
    if surrogateModel is None:
        return_surrogate = False
        surrogateModel = RbfModel()

    # Initialize output
    out = OptimizeResult()
    out.init(
        lambda x: np.column_stack((fun(x), gfun(x))),
        bounds,
        0,
        maxeval,
        surrogateModel,
        ntarget=1 + surrogateModel.ntarget,
    )
    out.nobj = 1
    out.init_best_values()
    assert isinstance(out.fx, np.ndarray)

    # Reserve space for the surrogate model to avoid repeated allocations
    gdim = out.fsample.shape[1] - 1
    assert gdim > 0
    surrogateModel.reserve(surrogateModel.ntrain + maxeval, dim, gdim)

    # Acquisition functions
    rtol = 1e-3
    acquisition1 = MinimizeMOSurrogate(rtol=rtol)
    acquisition2 = GosacSample(fun, rtol=rtol)
    maximizeDistance = MaximizeDistance(
        rtol=rtol
    )  # fallback if no point is found in step 2

    xselected = np.array(out.sample[0 : out.nfev, :], copy=True)
    ySelected = np.array(out.fsample[0 : out.nfev, 1:], copy=True)
    if gdim == 1:
        ySelected = ySelected.flatten()

    # Phase 1: Find a feasible solution
    while out.nfev < maxeval and out.x.size == 0:
        if disp:
            print("(Phase 1) Iteration: %d" % out.nit)
            print("fEvals: %d" % out.nfev)
            print(
                "Constraint violation in the last step: %f" % np.max(ySelected)
            )

        # Update surrogate models
        t0 = time.time()
        if out.nfev > 0:
            surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Solve the surrogate multiobjective problem
        t0 = time.time()
        bestCandidates = acquisition1.optimize(surrogateModel, bounds, n=0)
        tf = time.time()
        if disp:
            print(
                "Solving the surrogate multiobjective problem: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        # Evaluate the surrogate at the best candidates
        sCandidates = surrogateModel(bestCandidates)

        # Find the minimum number of constraint violations
        constraintViolation = [
            np.sum(sCandidates[i, :] > 0) for i in range(len(bestCandidates))
        ]
        minViolation = np.min(constraintViolation)
        idxMinViolation = np.where(constraintViolation == minViolation)[0]

        # Find the candidate with the minimum violation
        idxSelected = np.argmin(
            [
                np.sum(np.maximum(sCandidates[i, :], 0.0))
                for i in idxMinViolation
            ]
        )
        xselected = bestCandidates[idxSelected, :].reshape(1, -1)

        # Compute g(xselected)
        ySelected = np.asarray(gfun(xselected))

        # Check if xselected is a feasible sample
        if np.max(ySelected) <= 0:
            fxSelected = fun(xselected)
            out.x = xselected[0]
            out.fx = np.empty(gdim + 1)
            out.fx[0] = fxSelected
            out.fx[1:] = ySelected
            out.fsample[out.nfev, 0] = fxSelected
        else:
            out.fsample[out.nfev, 0] = np.inf

        # Update sample and fsample in out
        out.sample[out.nfev, :] = xselected
        out.fsample[out.nfev, 1:] = ySelected

        # Update the counters
        out.nfev = out.nfev + 1
        out.nit = out.nit + 1

        # Call the callback function
        if callback is not None:
            callback(out)

    if out.x.size == 0:
        # No feasible solution was found
        out.sample.resize(out.nfev, dim)
        out.fsample.resize(out.nfev, gdim)

        # Update surrogate model if it lives outside the function scope
        t0 = time.time()
        if out.nfev > 0:
            surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        return out

    # Phase 2: Optimize the objective function
    while out.nfev < maxeval:
        if disp:
            print("(Phase 2) Iteration: %d" % out.nit)
            print("fEvals: %d" % out.nfev)
            print("Best value: %f" % out.fx[0])

        # Update surrogate models
        t0 = time.time()
        if out.nfev > 0:
            surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Solve cheap problem with multiple constraints
        t0 = time.time()
        xselected = acquisition2.optimize(surrogateModel, bounds)
        if len(xselected) == 0:
            xselected = maximizeDistance.optimize(surrogateModel, bounds, n=1)
        tf = time.time()
        if disp:
            print(
                "Solving the cheap problem with surrogate cons: %d points in %f s"
                % (len(xselected), tf - t0)
            )

        # Compute g(xselected)
        ySelected = np.asarray(gfun(xselected))

        # Check if xselected is a feasible sample
        if np.max(ySelected) <= 0:
            fxSelected = fun(xselected)[0]
            if fxSelected < out.fx[0]:
                out.x = xselected[0]
                out.fx[0] = fxSelected
                out.fx[1:] = ySelected
            out.fsample[out.nfev, 0] = fxSelected
        else:
            out.fsample[out.nfev, 0] = np.inf

        # Update sample and fsample in out
        out.sample[out.nfev, :] = xselected
        out.fsample[out.nfev, 1:] = ySelected

        # Update the counters
        out.nfev = out.nfev + 1
        out.nit = out.nit + 1

        # Call the callback function
        if callback is not None:
            callback(out)

    # Update surrogate model if it lives outside the function scope
    if return_surrogate:
        t0 = time.time()
        if out.nfev > 0:
            surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

    return out
