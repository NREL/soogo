"""SHEBO optimization algorithm."""

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
import warnings
from typing import Callable, Optional

import numpy as np
from scipy.spatial.distance import cdist

from soogo.acquisition.base import Acquisition
from soogo.model.rbf import MedianLpfFilter

from ..acquisition import (
    AlternatedAcquisition,
    GosacSample,
    MaximizeDistance,
    WeightedAcquisition,
    MultipleAcquisition,
)
from ..sampling import NormalSampler, SamplingStrategy
from ..acquisition.utils import FarEnoughSampleFilter
from ..model import LinearRadialBasisFunction, RbfModel, Surrogate
from .utils import OptimizeResult, evaluate_and_log_point
from ..termination import IterateNTimes
from ..integrations.nomad import NomadProblem

try:
    import PyNomad
except ImportError:
    PyNomad = None


class NormalAndUniformSampler(NormalSampler):
    """Sampler that generates half normal and half uniform samples."""

    def get_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), **kwargs
    ) -> np.ndarray:
        """Generate n samples, half from normal distribution and half from
        uniform distribution.

        :param bounds: List with the limits [x_min,x_max] of each direction x
            in the space.
        :param iindex: Indices of the dimensions to be sampled. If None, all
            dimensions are sampled.
        :return: Matrix with a sample point per line.
        """
        _n = self.n

        self.n = _n // 2
        x_normal = super().get_sample(bounds, iindex=iindex, **kwargs)

        self.n = _n - self.n
        x_uniform = self.get_uniform_sample(bounds, iindex=iindex)

        self.n = _n
        return np.vstack((x_normal, x_uniform))


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
        return OptimizeResult()

    dim = len(bounds)  # Dimension of the problem
    assert dim > 0
    rtol = 1e-3  # Relative tolerance for distance-based operations
    return_surrogate = (objSurrogate is not None) or (
        evalSurrogate is not None
    )  # Whether to return the surrogate models

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
                            termination=IterateNTimes(1),
                        ),
                        WeightedAcquisition(
                            sampler=NormalAndUniformSampler(
                                1000 * dim,
                                sigma=0.02,
                                strategy=SamplingStrategy.DDS,
                            ),
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
                            sigma_min=0.02,
                            sigma_max=0.02,
                            rtol=rtol,
                            termination=IterateNTimes(9),
                        ),
                        MaximizeDistance(
                            rtol=rtol, termination=IterateNTimes(1)
                        ),
                    ]
                ),
                MaximizeDistance(rtol=rtol),
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
    out.init(fun, bounds, 1, maxeval, evalSurrogate)

    # Evaluate x_not_in_obj points and log results
    if len(x_not_in_obj) > 0:
        evaluate_and_log_point(fun, x_not_in_obj, out)

    # Initialize best values in out
    out.init_best_values(objSurrogate)

    # Call the callback function with the current optimization result
    if callback is not None:
        callback(out)

    # Keep adding points until there is a sufficient initial design for
    # the objective surrogate
    if objSurrogate.ntrain == 0:
        while not objSurrogate.check_initial_design(
            out.sample[: out.nfev][np.isfinite(out.fsample[: out.nfev])]
        ) and (out.nfev < maxeval):
            if disp:
                print(
                    "Iteration: %d (Objective surrogate under construction)"
                    % out.nit
                )
                print("fEvals: %d" % out.nfev)
                print(
                    "Number of feasible points: %d"
                    % np.sum(np.isfinite(out.fsample[: out.nfev]))
                )

                dist = cdist(out.sample[: out.nfev], out.sample[: out.nfev])
                dist += np.eye(out.nfev) * np.max(dist)
                print(
                    "Max distance between neighbors: %f"
                    % np.max(np.min(dist, axis=1))
                )
                print(f"Last sampled point: {out.sample[out.nfev - 1]}")

            # Acquire new sample point
            xNew = MaximizeDistance(rtol=rtol).optimize(
                objSurrogate, bounds, 1, points=out.sample[: out.nfev]
            )

            # Compute f(xNew) and update out
            evaluate_and_log_point(fun, xNew, out)
            out.init_best_values(objSurrogate)
            out.nit += 1

            # Call the callback function
            if callback is not None:
                callback(out)

    # Prepare for the main optimization loop
    #
    # - At this point, we have enough points to build both surrogates
    nStart = out.nfev
    nomadFunction = NomadProblem(fun, out)
    xselected = np.array(out.sample[0 : out.nfev, :], copy=True)
    ySelected = np.array(out.fsample[0 : out.nfev], copy=True)

    # do until max number of f-evals reached or local min found
    while out.nfev < maxeval:
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % out.nfev)
            print("Best value: %f" % out.fx)

        # Update surrogate models
        t0 = time.time()
        feasible_idx = np.isfinite(ySelected)
        evalSurrogate.update(xselected, feasible_idx.astype(float))
        if np.any(feasible_idx):
            objSurrogate.update(
                xselected[feasible_idx], ySelected[feasible_idx]
            )
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Calculate the threshold for evaluability
        threshold = float(
            np.log(max(1, out.nfev - nStart + 1)) / np.log(maxeval - nStart)
        )

        # Set perturbation probability
        if dim <= 10:
            perturbProbability = 1.0
        else:
            perturbProbability = np.random.uniform(0, 1)

        # Acquire new sample points
        t0 = time.time()
        xselected = acquisitionFunc.optimize(
            objSurrogate,
            bounds,
            1,
            points=evalSurrogate.X,
            constr_fun=lambda x: threshold - evalSurrogate(x),
            perturbation_probability=perturbProbability,
        )
        if len(xselected) == 0:
            threshold = float(np.finfo(float).eps)
        tf = time.time()
        if disp:
            print("Time to acquire new sample points: %f s" % (tf - t0))

        # Compute f(xselected)
        if len(xselected) > 0:
            ySelected = evaluate_and_log_point(fun, xselected, out)
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
            new_best_point_found = True
        else:
            new_best_point_found = False

        # If the new point was better than current best, run NOMAD
        if new_best_point_found:
            if disp:
                print("New best point found, running NOMAD...")

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
                ],
            )

            # Use the best point found by NOMAD
            if res["f_best"] < out.fx:
                out.x[:] = res["x_best"]
                out.fx = res["f_best"]

            # Get the points sampled by NOMAD
            nomadSample = np.array(nomadFunction.get_x_history())
            nomadFSample = np.array(nomadFunction.get_f_history())

            if disp:
                print(
                    f"NOMAD optimization completed. NOMAD used {len(nomadSample)} evaluations."
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
    out.sample.resize(out.nfev, dim)
    out.fsample.resize(out.nfev)

    # Update surrogate model if it lives outside the function scope
    if return_surrogate and evalSurrogate.ntrain > 0:
        t0 = time.time()
        feasible_idx = np.isfinite(ySelected)
        evalSurrogate.update(xselected, feasible_idx.astype(float))
        if np.any(feasible_idx):
            objSurrogate.update(
                xselected[feasible_idx], ySelected[feasible_idx]
            )
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

    return out
