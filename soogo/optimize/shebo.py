"""SHEBO optimization algorithm."""

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

import warnings
from typing import Callable, Optional

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

from ..acquisition import (
    AlternatedAcquisition,
    GosacSample,
    MaximizeDistance,
    TransitionSearch,
)
from ..model import (
    CubicRadialBasisFunction,
    LinearRadialBasisFunction,
    RbfModel,
    Surrogate,
)
from .utils import OptimizeResult, evaluate_and_log_point
from ..sampling import Sampler
from ..termination import IterateNTimes
from ..integrations.nomad import NomadProblem

try:
    import PyNomad
except ImportError:
    PyNomad = None


def shebo(
    fun,
    bounds,
    maxeval: int,
    *,
    objSurrogate: Optional[Surrogate] = None,
    evalSurrogate: Optional[Surrogate] = None,
    acquisitionFunc: Optional[AlternatedAcquisition] = None,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
) -> OptimizeResult:
    """
    Minimize a function using the SHEBO algorithm from [#]_.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param objSurrogate: Surrogate model for the objective function. If None is
        provided, a :class:`RbfModel` model with Cubic Radial Basis Function is
        used. On exit, if provided, the surrogate model will contain the points
        used during the optimization process.
    :param evalSurrogate: Surrogate model for the evaluation function. If None
        is provided, a :class:`RbfModel` model with Linear Radial Basis Function
        is used. On exit, if provided, the surrogate model will contain the
        points used during the optimization process.
    :param acquisitionFunc: Acquisition function to be used in the optimization
        loop. If None is provided, the acquisition cycle described by
        Müller and Day (2019) is used. Each call, the acquisition function is
        provided with the
        surrogate objective model, bounds, and number of points to sample as
        positional arguments and the keyword arguments points,
        evaluabilitySurrogate, evaluabilityThreshold, and scoreWeight.
    :param disp: If True, print information about the optimization process. The
        default is False.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result. The default is
        None.
    :return: The optimization result.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """
    # Check that required PyNomad package is available
    if PyNomad is None:
        warnings.warn(
            "PyNomad package is required but not installed. Install the PyNomad package and try again."
        )
        return

    # Initialize parameters
    weightPattern = [1, 0.95, 0.85, 0.75, 0.5, 0.35, 0.25, 0.1, 0.0]
    dim = len(bounds)
    nStart = 4 * (dim + 1)
    bounds = np.asarray(bounds)

    # Define function wrapper to rescale variables
    # This is needed as SHEBO internally rescales all variables to [0, 1]
    def rescaledFunc(x):
        rescaledX = x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        return fun(rescaledX)

    # Initialize output
    out = OptimizeResult(
        x=np.empty((0, dim)),
        fx=np.inf,
        nfev=0,
        sample=np.zeros((maxeval, dim)),
        fsample=np.zeros(maxeval),
    )

    # Create Nomad wrapper
    nomadFunction = NomadProblem(rescaledFunc, out)

    # Initialize or use provided surrogates
    if objSurrogate is None:
        objSurrogate = RbfModel(CubicRadialBasisFunction())
    if evalSurrogate is None:
        evalSurrogate = RbfModel(LinearRadialBasisFunction())

    # Reserve space for the surrogates
    objSurrogate.reserve(objSurrogate.ntrain + maxeval, dim)
    evalSurrogate.reserve(evalSurrogate.ntrain + maxeval, dim)

    # Check if surrogates already trained
    objTrained = objSurrogate.ntrain > 0
    evalTrained = evalSurrogate.ntrain > 0

    # Default acquisition function
    if acquisitionFunc is None:
        acquisitionFunc = AlternatedAcquisition(
            [
                GosacSample(
                    objSurrogate, rtol=0.001, termination=IterateNTimes(1)
                ),
                TransitionSearch(rtol=0.001, termination=IterateNTimes(9)),
                MaximizeDistance(rtol=0.001, termination=IterateNTimes(1)),
            ]
        )

    if objTrained and evalTrained:
        # Both surrogates are pre-trained, skip initial sampling
        if disp:
            print("Both surrogates pre-trained. Skipping initial sampling.")
        out.fx = np.min(objSurrogate.Y)
        out.x = objSurrogate.X[np.argmin(objSurrogate.Y)]

    elif objTrained and not evalTrained:
        # Only obj surrogate trained - initialize eval surrogate
        if disp:
            print(
                "Initializing evaluation surrogate from objective surrogate."
            )

        evalSurrogate.update(objSurrogate.X, np.ones(len(objSurrogate.X)))
        out.fx = np.min(objSurrogate.Y)
        out.x = objSurrogate.X[np.argmin(objSurrogate.Y)]

    elif not objTrained and evalTrained:
        # Only eval surrogate trained - initialize obj surrogate
        if disp:
            print(
                "Initializing objective surrogate from evaluation surrogate."
            )

        # Extract all points in evalSurrogate with a value of 1
        evalPoints = evalSurrogate.X[evalSurrogate.Y == 1]

        # Evaluate each point
        for x in evalPoints:
            if out.nfev >= maxeval:
                break

            _ = evaluate_and_log_point(rescaledFunc, x, out)

            if disp:
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

        # Check if more points are needed for the objective surrogate
        if not objSurrogate.check_initial_design(
            np.array(
                out.sample[: out.nfev][~np.isnan(out.fsample[: out.nfev])]
            )
        ):
            maximizeDistance = MaximizeDistance(rtol=0.001)
            if disp:
                print(
                    "Sampling additional points to initialize the surrogate model..."
                )

            while (
                not objSurrogate.check_initial_design(
                    np.array(
                        out.sample[: out.nfev][
                            ~np.isnan(out.fsample[: out.nfev])
                        ]
                    )
                )
            ) and (out.nfev < maxeval):
                ## Generate new point
                xNew = maximizeDistance.optimize(
                    evalSurrogate, [[0, 1] for _ in range(dim)], 1
                )

                f = evaluate_and_log_point(rescaledFunc, xNew, out)

                if disp:
                    print("fEvals: %d" % out.nfev)
                    print("Best value: %f" % out.fx)

                # Update the surrogate model with the new point
                evalSurrogate.update(
                    np.array(xNew), np.logical_not(np.isnan(f)).astype(float)
                )

    else:
        # Neither surrogate is trained - generate initial sample
        if disp:
            print("Performing initial sampling...")

        # Generate initial points using Latin Hypercube sampling
        sampler = Sampler(nStart)
        x0 = sampler.get_slhd_sample([[0, 1] for _ in range(dim)])

        # Check that all points are far enough apart
        distances = squareform(pdist(x0))
        keptIndices = [0]

        for i in range(1, len(x0)):
            # Check if this point is far enough from all previously kept points
            if all(
                distances[i, kept_idx] >= 0.001 for kept_idx in keptIndices
            ):
                keptIndices.append(i)

        x0 = x0[np.array(keptIndices)]

        if disp:
            print("Evaluating initial points...")

        # Evaluate the initial points
        for x in x0:
            if out.nfev >= maxeval:
                break

            _ = evaluate_and_log_point(rescaledFunc, x, out)

            if disp:
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

        # Build the evaluability surrogate model
        evalSurrogate.update(
            np.array(out.sample[0 : out.nfev, :]),
            np.logical_not(np.isnan(out.fsample[0 : out.nfev])).astype(float),
        )

        # Check if initial design sufficient to initialize objective surrogate
        if not objSurrogate.check_initial_design(
            np.array(
                out.sample[: out.nfev][~np.isnan(out.fsample[: out.nfev])]
            )
        ):
            maximizeDistance = MaximizeDistance(rtol=0.001)
            if disp:
                print(
                    "Sampling additional points to initialize the surrogate model..."
                )

            while (
                not objSurrogate.check_initial_design(
                    np.array(
                        out.sample[: out.nfev][
                            ~np.isnan(out.fsample[: out.nfev])
                        ]
                    )
                )
            ) and (out.nfev < maxeval):
                ## Generate new point
                xNew = maximizeDistance.optimize(
                    evalSurrogate, [[0, 1] for _ in range(dim)], 1
                )

                f = evaluate_and_log_point(rescaledFunc, xNew, out)

                if disp:
                    print("fEvals: %d" % out.nfev)
                    print("Best value: %f" % out.fx)

                # Update the surrogate model with the new point
                evalSurrogate.update(
                    np.array(xNew), np.logical_not(np.isnan(f)).astype(float)
                )

    # If we have run out of evaluations, we cannot continue
    if out.nfev >= maxeval:
        print(
            "Maximum number of evaluations reached before enough points were sampled to initialize the surrogate model."
        )
        return out

    # Generate the surrogate model for the objective function
    objSurrogate.update(
        np.array(out.sample[: out.nfev][~np.isnan(out.fsample[: out.nfev])]),
        np.array(out.fsample[: out.nfev][~np.isnan(out.fsample[: out.nfev])]),
    )

    if disp:
        print(
            "Objective surrogate model initialized with %d points."
            % objSurrogate.ntrain
        )
        print("Starting optimization search...")

    # Call the callback function with the current optimization result
    if callback is not None:
        callback(out)

    # Main optimization loop
    while out.nfev < maxeval:
        # Calculate the threshold for evaluability
        threshold = np.log(max(1, out.nfev - nStart + 1)) / np.log(
            maxeval - nStart
        )

        # Generate new point
        xNew = acquisitionFunc.optimize(
            objSurrogate,
            [[0, 1] for _ in range(dim)],
            n=1,
            points=evalSurrogate.X,
            constraintTransform=lambda x: -x + threshold,
            evaluabilitySurrogate=evalSurrogate,
            evaluabilityThreshold=threshold,
            scoreWeight=weightPattern[
                acquisitionFunc.acquisitionFuncArray[
                    acquisitionFunc.idx
                ].termination.iterationCount
            ],
        )

        # Evaluate new point
        f = evaluate_and_log_point(rescaledFunc, xNew, out)

        if disp:
            print("fEvals: %d" % out.nfev)
            print("Best value: %f" % out.fx)

        # Update evaluability surrogate model
        evalSurrogate.update(
            np.array(xNew), np.logical_not(np.isnan(f)).astype(float)
        )

        # If successful, update the obj surrogate
        if not np.isnan(f):
            objSurrogate.update(np.array(xNew), np.array(f))

        # If the new point was better than current best, run NOMAD
        if np.array_equiv(xNew, out.x):
            if disp:
                print("New best point found, running NOMAD...")

            nomadFunction.reset()

            PyNomad.optimize(
                fBB=nomadFunction,
                pX0=xNew.flatten(),
                pLB=np.array([0 for _ in range(dim)]),
                pUB=np.array([1 for _ in range(dim)]),
                params=[
                    "BB_OUTPUT_TYPE OBJ",
                    f"MAX_BB_EVAL {min(4 * dim, maxeval - out.nfev)}",
                    "DISPLAY_DEGREE 0",
                    "QUAD_MODEL_SEARCH 0",
                ],
            )

            # Get the points sampled by NOMAD
            nomadSample = nomadFunction.get_x_history()
            nomadFSample = nomadFunction.get_f_history()

            for i in range(len(nomadSample)):
                point = nomadSample[i].reshape(1, -1)
                # Check distance to all previously sampled points
                if cdist(point, evalSurrogate.X).min() >= 1e-7:
                    fval = nomadFSample[i]
                    # Update surrogates
                    evalSurrogate.update(
                        point, np.logical_not(np.isnan(fval)).astype(float)
                    )
                    if not np.isnan(fval):
                        objSurrogate.update(point, fval)
            if disp:
                print(
                    f"NOMAD optimization completed. NOMAD used {len(nomadSample)} evaluations."
                )
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

        # Call the callback function with the current optimization result
        if callback is not None:
            callback(out)

        # Update the acquisition function
        acquisitionFunc.update(out, objSurrogate)
        if acquisitionFunc.has_converged():
            break

    # Rescale the x and sample arrays to the original bounds
    out.x = out.x * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    out.sample = out.sample * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    # Return OptimizeResult
    return out
