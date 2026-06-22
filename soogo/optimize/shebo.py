"""SHEBO optimization algorithm."""

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
import warnings
import logging
from typing import Callable, Optional
import functools

import numpy as np
from scipy.spatial.distance import cdist

from ..acquisition import (
    Acquisition,
    AlternatedAcquisition,
    GosacSample,
    MaximizeDistance,
    MultipleAcquisition,
    CoordinatePerturbation,
)
from ..acquisition.utils import FarEnoughSampleFilter
from ..model import (
    LinearRadialBasisFunction,
    RbfModel,
    Surrogate,
    MedianLpfFilter,
)
from .utils import OptimizeResult, evaluate_and_log_point
from ..sampling import SpaceFillingSampler
from ..termination import IterateNTimes
from ..integrations.nomad import NomadProblem

logger = logging.getLogger(__name__)

try:
    import PyNomad
except ImportError:
    PyNomad = None


def _constraint_function(threshold, surrogate, x):
    """Constraint: threshold - surrogate(x) <= 0

    :param threshold: Threshold for feasibility.
    :param surrogate: Surrogate model for the evaluation function.
    :param x: Point to evaluate.
    :return: Value of the constraint function at x.
    """
    return threshold - surrogate(x)


def shebo(
    fun,
    bounds,
    maxeval: int,
    *,
    objSurrogate: Optional[RbfModel] = None,
    evalSurrogate: Optional[Surrogate] = None,
    acquisitionFunc: Optional[Acquisition] = None,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    seed=None,
) -> OptimizeResult:
    """
    Minimize a function using the SHEBO algorithm from [#]_.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param objSurrogate: Surrogate model for the objective function. If None is
        provided, a :class:`.RbfModel` model with Cubic Radial Basis Function is
        used. On exit, if provided, the surrogate model will contain the points
        used during the optimization process.
    :param evalSurrogate: Surrogate model for the evaluation function. If None
        is provided, a :class:`.RbfModel` model with Linear Radial Basis
        Function is used. On exit, if provided, the surrogate model will contain
        the points used during the optimization process.
    :param acquisitionFunc: Acquisition function used in the main optimization
        loop. If ``None``, uses the cycle from Müller and Day (2019).
        Each call provides the surrogate objective model, bounds, and the
        requested number of points as positional arguments, and may pass
        keywords such as ``points`` (existing sample), ``mu`` (best point),
        ``constr`` (feasibility function), and
        ``perturbation_probability`` (for DDS-like samplers).
    :param disp: Deprecated and ignored. Configure logging via standard Python
        logging levels instead.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result. The default is
        None.
    :param seed: Seed or random number generator.
    :return: The optimization result.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """
    if disp:
        warnings.warn(
            "'disp' is deprecated and ignored; use logging levels instead",
            DeprecationWarning,
            stacklevel=2,
        )
    # Check that required PyNomad package is available
    if PyNomad is None:
        warnings.warn(
            "PyNomad package is required but not installed. Install the PyNomad package and try again.",
            ImportWarning,
            stacklevel=2,
        )
        return OptimizeResult()

    # Random number generator
    rng = np.random.default_rng(seed)

    dim = len(bounds)  # Dimension of the problem
    if dim <= 0:
        raise ValueError("bounds must define at least one dimension")

    rtol = Acquisition.DEFAULT_RTOL
    return_surrogate = (objSurrogate is not None) or (
        evalSurrogate is not None
    )  # Whether to return the surrogate models

    # Acquisition function to maximize distance
    maxDistAcq = MaximizeDistance(
        seed=rng.integers(np.iinfo(np.int32).max).item(), rtol=rtol
    )

    # Initialize optional variables
    if objSurrogate is None:
        objSurrogate = RbfModel(filter=MedianLpfFilter())
    if evalSurrogate is None:
        evalSurrogate = RbfModel(LinearRadialBasisFunction())
    if acquisitionFunc is None:
        acquisitionFunc = MultipleAcquisition(
            [
                AlternatedAcquisition(
                    [
                        GosacSample(
                            objSurrogate,
                            rtol=rtol,
                            seed=rng.integers(
                                np.iinfo(np.int32).max, size=1
                            ).item(),
                            termination=IterateNTimes(1),
                        ),
                        CoordinatePerturbation(
                            sampling_strategy="dds_uniform",
                            perturbation_strategy=(
                                "random" if dim > 10 else "fixed"
                            ),
                            sigma=0.2,
                            pool_size=min(1000 * dim, 5000),
                            weightpattern=[
                                1.0,
                                0.95,
                                0.85,
                                0.75,
                                0.5,
                                0.35,
                                0.25,
                                0.1,
                                0.0,
                            ],
                            seed=rng.integers(
                                np.iinfo(np.int32).max, size=1
                            ).item(),
                            rtol=rtol,
                            termination=IterateNTimes(9),
                        ),
                        MaximizeDistance(
                            seed=rng.integers(
                                np.iinfo(np.int32).max, size=1
                            ).item(),
                            rtol=rtol,
                            termination=IterateNTimes(1),
                        ),
                    ]
                ),
                maxDistAcq,
            ]
        )

    # Create lists of points not in each surrogate
    x_not_in_obj = np.empty((0, dim))
    x_not_in_eval = np.empty((0, dim))
    if evalSurrogate.ntrain == 0 and objSurrogate.ntrain > 0:
        x_not_in_eval = objSurrogate.X
    elif evalSurrogate.ntrain > 0 and objSurrogate.ntrain == 0:
        x_not_in_obj = evalSurrogate.X[evalSurrogate.Y == 1]
    elif evalSurrogate.ntrain > 0 and objSurrogate.ntrain > 0:
        x_not_in_eval = FarEnoughSampleFilter(evalSurrogate.X, tol=rtol)(
            objSurrogate.X
        )
        x_not_in_obj = FarEnoughSampleFilter(objSurrogate.X, tol=rtol)(
            evalSurrogate.X[evalSurrogate.Y == 1]
        )

    # Reserve space for the surrogates
    objSurrogate.reserve(objSurrogate.ntrain + maxeval, dim)
    evalSurrogate.reserve(
        evalSurrogate.ntrain
        + maxeval
        + len(x_not_in_eval)
        - len(x_not_in_obj),
        dim,
    )

    # Update evalSurrogate with points from the objective surrogate
    # Assumption: points in objSurrogate are enough to train evalSurrogate
    if len(x_not_in_eval) > 0:
        evalSurrogate.update(x_not_in_eval, np.ones(len(x_not_in_eval)))

    # Initialize output
    # At this point, either
    #
    # - both evalSurrogate and objSurrogate are empty, or
    # - evalSurrogate is initialized.
    out = OptimizeResult()
    out.init(fun, bounds, 1, maxeval, evalSurrogate, seed=seed)

    # Evaluate x_not_in_obj points and log results
    if len(x_not_in_obj) > 0:
        evaluate_and_log_point(fun, x_not_in_obj, out)

    # Initialize best values in out
    out.init_best_values(objSurrogate)
    assert isinstance(out.x, np.ndarray), "Expected np.ndarray, got %s" % type(
        out.fx
    )

    # Call the callback function with the current optimization result
    if callback is not None:
        callback(out)

    # Keep adding points until there is a sufficient initial design for
    # the objective surrogate
    if objSurrogate.ntrain == 0:
        sampler = SpaceFillingSampler(
            seed=rng.integers(np.iinfo(np.int32).max).item()
        )
        n_points_to_add = objSurrogate.check_initial_design(
            out.sample[: out.nfev][np.isfinite(out.fsample[: out.nfev])]
        )
        while (n_points_to_add > 0) and (out.nfev < maxeval):
            logger.info(
                "Iteration: %d (Objective surrogate under construction)",
                out.nit,
            )
            logger.info("fEvals: %d", out.nfev)
            logger.info(
                "Number of feasible points: %d",
                np.sum(np.isfinite(out.fsample[: out.nfev])),
            )

            dist = cdist(out.sample[: out.nfev], out.sample[: out.nfev])
            dist += np.eye(out.nfev) * np.max(dist)
            logger.info(
                "Max distance between neighbors: %f",
                np.max(np.min(dist, axis=1)),
            )
            logger.info("Last sampled point: %s", out.sample[out.nfev - 1])

            # Acquire new sample point
            xNew = sampler.generate(
                min(n_points_to_add, maxeval - out.nfev),
                bounds,
                current_sample=out.sample[: out.nfev],
                iindex=objSurrogate.iindex,
            )

            # Compute f(xNew) and update out
            evaluate_and_log_point(fun, xNew, out)
            out.init_best_values(objSurrogate)
            out.nit += 1

            # Call the callback function
            if callback is not None:
                callback(out)

            # Recompute number of points to add
            n_points_to_add = objSurrogate.check_initial_design(
                out.sample[: out.nfev][np.isfinite(out.fsample[: out.nfev])]
            )

    # Prepare for the main optimization loop
    #
    # - At this point, we have enough points to build both surrogates
    nStart = out.nfev
    nomadFunction = NomadProblem(fun, out)
    xselected = np.array(out.sample[0 : out.nfev, :], copy=True)
    ySelected = np.array(out.fsample[0 : out.nfev], copy=True)

    # do until max number of f-evals reached or local min found
    while out.nfev < maxeval:
        logger.info("Iteration: %d", out.nit)
        logger.info("fEvals: %d", out.nfev)
        logger.info("Best value: %f", out.fx)

        # Update surrogate models
        t0 = time.time()
        feasible_idx = np.isfinite(ySelected)
        evalSurrogate.update(xselected, feasible_idx.astype(float))
        if np.any(feasible_idx):
            objSurrogate.update(
                xselected[feasible_idx], ySelected[feasible_idx]
            )
        tf = time.time()
        logger.info("Time to update surrogate model: %f s", (tf - t0))

        # Calculate the threshold for evaluability
        threshold = float(
            np.log(max(1, out.nfev - nStart + 1)) / np.log(maxeval - nStart)
        )

        # Acquire new sample points
        t0 = time.time()
        xselected = acquisitionFunc.optimize(
            objSurrogate,
            bounds,
            n=1,
            exclusion_set=evalSurrogate.X[evalSurrogate.Y == 0],
            xbest=out.x,
            ybest=out.fx,
            constr=functools.partial(
                _constraint_function, threshold, evalSurrogate
            ),
        )
        if len(xselected) == 0:
            threshold = float(np.finfo(float).eps)
        tf = time.time()
        logger.info("Time to acquire new sample points: %f s", (tf - t0))

        # Compute f(xselected)
        if len(xselected) > 0:
            ySelected = evaluate_and_log_point(fun, xselected, out)
        else:
            ySelected = np.empty((0,))
            out.nit = out.nit + 1
            logger.warning(
                "Acquisition function failed to find a new sample; consider modifying it."
            )
            break

        # determine best one of newly sampled points
        iSelectedBest = np.argmin(ySelected).item()
        fxSelectedBest = ySelected[iSelectedBest]
        if fxSelectedBest < out.fx:
            out.x[:] = xselected[iSelectedBest, :]
            out.fx = fxSelectedBest
            new_best_point_found = True
        else:
            new_best_point_found = False

        # If the new point was better than current best, run NOMAD
        if new_best_point_found:
            logger.info("New best point found, running NOMAD...")

            nomadFunction.reset()

            res = PyNomad.optimize(
                fBB=nomadFunction,
                pX0=out.x,
                pLB=[b[0] for b in bounds],
                pUB=[b[1] for b in bounds],
                params=[
                    "BB_OUTPUT_TYPE OBJ",
                    f"MAX_BB_EVAL {min(4 * dim, maxeval - out.nfev)}",
                    "DISPLAY_DEGREE 0",
                    "QUAD_MODEL_SEARCH 0",
                    f"SEED {rng.integers(np.iinfo(np.int32).max).item()}",
                ],
            )

            # Use the best point found by NOMAD
            if res["f_best"] < out.fx:
                out.x[:] = res["x_best"]
                out.fx = res["f_best"]

            # Get the points sampled by NOMAD
            nomadSample = np.array(nomadFunction.get_x_history())
            nomadFSample = np.array(nomadFunction.get_f_history())

            logger.info(
                "NOMAD optimization completed. NOMAD used %d evaluations.",
                len(nomadSample),
            )

            # Filter out points that are too close to existing samples
            idxes = FarEnoughSampleFilter(
                np.vstack((evalSurrogate.X, xselected)), tol=rtol
            ).indices(nomadSample)
            xselected = np.vstack((xselected, nomadSample[idxes]))
            ySelected = np.hstack((ySelected, nomadFSample[idxes]))

        # Update out.nit
        out.nit = out.nit + 1

        # Call the callback function
        if callback is not None:
            callback(out)

        # Terminate if acquisition function has converged
        acquisitionFunc.update(out, objSurrogate)
        if acquisitionFunc.has_converged():
            break

    # Update output
    out.sample = out.sample[: out.nfev]
    out.fsample = out.fsample[: out.nfev]

    # Update surrogate model if it lives outside the function scope
    if return_surrogate and evalSurrogate.ntrain > 0:
        t0 = time.time()
        feasible_idx = np.isfinite(ySelected)
        try:
            evalSurrogate.update(xselected, feasible_idx.astype(float))
            if np.any(feasible_idx):
                objSurrogate.update(
                    xselected[feasible_idx], ySelected[feasible_idx]
                )
        except Exception as e:
            logger.error("Failed to update surrogate model: %s", e)
        tf = time.time()
        logger.info("Time to update surrogate model: %f s", (tf - t0))

    return out
