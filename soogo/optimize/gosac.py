"""GOSAC constrained optimization routine."""

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

import time
import logging
import warnings
from typing import Callable, Optional
import numpy as np
import functools

from pymoo.optimize import minimize as pymoo_minimize

from ..acquisition import (
    GosacSample,
    MaximizeDistance,
    MinimizeMOSurrogate,
    MultipleAcquisition,
)
from ..model import RbfModel, Surrogate
from .utils import OptimizeResult
from ..integrations.pymoo import PymooProblem

logger = logging.getLogger(__name__)


def _combined_objectives(objective_fn, constraint_fn, x):
    """Stack objective and constraint values.

    :param objective_fn: Objective function to minimize.
    :param constraint_fn: Constraint function to minimize (should return array).
    :param x: Point to evaluate.
    :return: Combined array of objective and constraint values.
    """
    return np.column_stack((objective_fn(x), constraint_fn(x)))


def gosac(
    fun,
    gfun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    seed=None,
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
    :param disp: Deprecated and ignored. Configure logging instead using
        standard Python logging levels.
    :param callback: If provided, the callback function will be called
        after each iteration with the current optimization result. The
        default is None.
    :param seed: Seed or random number generator.
    :return: The optimization result.

    References
    ----------
    .. [#] Juliane Müller and Joshua D. Woodbury. 2017. GOSAC: global
        optimization with surrogate approximation of constraints. J. of Global
        Optimization 69, 1 (September 2017), 117–136.
        https://doi.org/10.1007/s10898-017-0496-y
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
    return_surrogate = True
    if surrogateModel is None:
        return_surrogate = False
        surrogateModel = RbfModel()

    # Initialize output
    out = OptimizeResult()
    out.init(
        functools.partial(_combined_objectives, fun, gfun),
        bounds,
        0,
        maxeval,
        surrogateModel,
        seed=seed,
    )

    # Fix nobj and fsample to account for constraints
    out.nobj = 1
    if out.nfev == 0:
        out.fsample = np.column_stack(
            (np.full(len(out.fsample), np.nan), out.fsample)
        )

    # Initialize best values if a feasible solution is found at the start
    if out.nfev == 0:
        feasible_idx = np.where(np.all(surrogateModel.X <= 0, axis=1))[0]
        feasible_x = surrogateModel.X[feasible_idx]
        if len(feasible_x) > 0:
            feasible_fx = np.asarray(fun(feasible_x))
            feasible_gx = surrogateModel.Y[feasible_idx]
            best_feasible_idx = np.argmin(feasible_fx)
            out.x = feasible_x[best_feasible_idx].copy()
            out.fx = np.hstack(
                (
                    feasible_fx[best_feasible_idx],
                    feasible_gx[best_feasible_idx],
                )
            )
    else:
        feasible_idx = np.where(np.all(out.fsample[:, 1:] <= 0, axis=1))[0]
        if len(feasible_idx) > 0:
            best_feasible_idx = feasible_idx[
                np.argmin(out.fsample[feasible_idx, 0])
            ]
            out.x = out.sample[best_feasible_idx].copy()
            out.fx = out.fsample[best_feasible_idx].copy()

    # Reserve space for the surrogate model to avoid repeated allocations
    gdim = out.fsample.shape[1] - 1
    if gdim <= 0:
        raise ValueError("Constraint dimension must be positive")
    surrogateModel.reserve(surrogateModel.ntrain + maxeval, dim, gdim)

    # Acquisition functions
    acquisition1 = MinimizeMOSurrogate(
        seed=rng.integers(np.iinfo(np.int32).max).item()
    )
    if gdim == 1:
        problem1 = PymooProblem(surrogateModel, bounds, surrogateModel.iindex)
        optimizer1 = acquisition1.default_optimizer(
            len(surrogateModel.iindex) > 0
        )
    acquisition2 = MultipleAcquisition(
        (
            GosacSample(fun, seed=rng.integers(np.iinfo(np.int32).max).item()),
            MaximizeDistance(seed=rng.integers(np.iinfo(np.int32).max).item()),
        )
    )

    xselected = np.array(out.sample[0 : out.nfev, :], copy=True)
    ySelected = np.array(out.fsample[0 : out.nfev, 1:], copy=True)
    if gdim == 1:
        ySelected = ySelected.flatten()

    # Phase 1: Find a feasible solution
    while out.nfev < maxeval and out.x is None:
        logger.info("(Phase 1) Iteration: %d", out.nit)
        logger.info("fEvals: %d", out.nfev)
        logger.info(
            "Constraint violation in the last step: %f", np.max(ySelected)
        )

        # Update surrogate models
        t0 = time.time()
        surrogateModel.update(xselected, ySelected)
        tf = time.time()
        logger.info("Time to update surrogate model: %f s", (tf - t0))

        # Solve the surrogate multiobjective problem
        t0 = time.time()
        if gdim == 1:
            res = pymoo_minimize(
                problem1,
                optimizer1,
                seed=rng.integers(np.iinfo(np.int32).max).item(),
                verbose=False,
            )
            if res.X is not None:
                bestCandidates = np.asarray([[res.X[i] for i in range(dim)]])
            else:
                bestCandidates = np.empty((0, dim))
        else:
            bestCandidates = acquisition1.optimize(
                surrogateModel, bounds, n=(maxeval - out.nfev)
            )
        tf = time.time()
        logger.info(
            "Solving the surrogate multiobjective problem: %d points in %f s",
            len(bestCandidates),
            tf - t0,
        )

        # Exit if no candidates were found
        if len(bestCandidates) == 0:
            raise RuntimeError(
                "Acquisition function failed to provide new candidates. "
                "Please try a different surrogate model or acquisition function."
            )

        # Evaluate the surrogate at the best candidates
        sCandidates = np.atleast_2d(surrogateModel(bestCandidates))

        # Find the minimum number of constraint violations
        constraintViolation = np.sum(sCandidates > 0, axis=1)
        minViolation = constraintViolation.min()
        idxMinViolation = np.where(constraintViolation == minViolation)[0]

        # Find the candidate with the minimum violation
        idxSelected = np.argmin(
            np.sum(np.maximum(sCandidates[idxMinViolation], 0.0), axis=1)
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
            out.fsample[out.nfev, 0] = np.nan

        # Update sample and fsample in out
        out.sample[out.nfev, :] = xselected
        out.fsample[out.nfev, 1:] = ySelected

        # Update the counters
        out.nfev = out.nfev + 1
        out.nit = out.nit + 1

        # Call the callback function
        if callback is not None:
            callback(out)

    if out.x is None:
        # No feasible solution was found
        assert out.nfev == maxeval

        # Update surrogate model if it lives outside the function scope
        t0 = time.time()
        surrogateModel.update(xselected, ySelected)
        tf = time.time()
        logger.info("Time to update surrogate model: %f s", (tf - t0))

        return out

    # Phase 2: Optimize the objective function
    while out.nfev < maxeval:
        logger.info("(Phase 2) Iteration: %d", out.nit)
        logger.info("fEvals: %d", out.nfev)
        logger.info("Best value: %f", out.fx[0])

        # Update surrogate models
        t0 = time.time()
        surrogateModel.update(xselected, ySelected)
        tf = time.time()
        logger.info("Time to update surrogate model: %f s", (tf - t0))

        # Solve cheap problem with multiple constraints
        t0 = time.time()
        xselected = acquisition2.optimize(surrogateModel, bounds)
        tf = time.time()
        logger.info(
            "Solving the cheap problem with surrogate cons: %d points in %f s",
            len(xselected),
            tf - t0,
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
            out.fsample[out.nfev, 0] = np.nan

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
        try:
            surrogateModel.update(xselected, ySelected)
        except Exception as e:
            logger.error("Failed to update surrogate model: %s", e)
        tf = time.time()
        logger.info("Time to update surrogate model: %f s", (tf - t0))

    return out
