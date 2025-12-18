"""SOCEMO multiobjective optimization routine."""

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
from scipy.spatial.distance import cdist

from ..acquisition import (
    CoordinatePerturbationOverNondominated,
    EndPointsParetoFront,
    MaximizeDistance,
    MinimizeMOSurrogate,
    ParetoFront,
    WeightedAcquisition,
)
from ..model import RbfModel, Surrogate
from .utils import OptimizeResult
from ..sampling import NormalSampler, Sampler
from ..utils import find_pareto_front


def socemo(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    acquisitionFunc: Optional[WeightedAcquisition] = None,
    acquisitionFuncGlobal: Optional[WeightedAcquisition] = None,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
):
    """Minimize a multiobjective function using the surrogate model approach
    from [#]_.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction
        x in the search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Multi-target surrogate model to be used. If
        None is provided, a :class:`.RbfModel` model is used.
    :param acquisitionFunc: Acquisition function to be used in the CP
        step. The default is WeightedAcquisition(0).
    :param acquisitionFuncGlobal: Acquisition function to be used in the
        global step. The default is WeightedAcquisition(Sampler(0),
        0.95).
    :param disp: If True, print information about the optimization
        process. The default is False.
    :param callback: If provided, the callback function will be called
        after each iteration with the current optimization result. The
        default is None.
    :return: The optimization result.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize optional variables
    return_surrogate = True
    if surrogateModel is None:
        return_surrogate = False
        surrogateModel = RbfModel()
    if acquisitionFunc is None:
        acquisitionFunc = WeightedAcquisition(NormalSampler(0, 0.1))
    if acquisitionFuncGlobal is None:
        acquisitionFuncGlobal = WeightedAcquisition(Sampler(0), 0.95)

    # Use a number of candidates that is greater than 1
    if acquisitionFunc.sampler.n <= 1:
        acquisitionFunc.sampler.n = min(500 * dim, 5000)
    if acquisitionFuncGlobal.sampler.n <= 1:
        acquisitionFuncGlobal.sampler.n = min(500 * dim, 5000)

    # Initialize output
    out = OptimizeResult()
    out.init(fun, bounds, 0, maxeval, surrogateModel)
    out.init_best_values(surrogateModel)
    assert isinstance(out.fx, np.ndarray)

    # Reserve space for the surrogate model to avoid repeated allocations
    objdim = out.nobj
    assert objdim > 1
    surrogateModel.reserve(surrogateModel.ntrain + maxeval, dim, objdim)

    # Define acquisition functions
    tol = acquisitionFunc.tol(bounds)
    step1acquisition = ParetoFront()
    step2acquisition = CoordinatePerturbationOverNondominated(acquisitionFunc)
    step3acquisition = EndPointsParetoFront(rtol=acquisitionFunc.rtol)
    step5acquisition = MinimizeMOSurrogate(rtol=acquisitionFunc.rtol)
    maximizeDistance = MaximizeDistance(rtol=acquisitionFunc.rtol)

    # do until max number of f-evals reached or local min found
    xselected = np.array(out.sample[0 : out.nfev, :], copy=True)
    ySelected = np.array(out.fsample[0 : out.nfev], copy=True)
    while out.nfev < maxeval:
        nMax = maxeval - out.nfev
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % out.nfev)

        # Update surrogate models
        t0 = time.time()
        if out.nfev > 0:
            surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        #
        # 0. Adjust parameters in the acquisition
        #
        acquisitionFunc.neval = max(
            acquisitionFunc.maxeval - (maxeval - out.nfev), 0
        )

        #
        # 1. Define target values to fill gaps in the Pareto front
        #
        t0 = time.time()
        xselected = step1acquisition.optimize(
            surrogateModel,
            bounds,
            n=1,
            nondominated=out.x,
            paretoFront=out.fx,
        )
        tf = time.time()
        if disp:
            print(
                "Fill gaps in the Pareto front: %d points in %f s"
                % (len(xselected), tf - t0)
            )

        #
        # 2. Random perturbation of the currently nondominated points
        #
        t0 = time.time()
        bestCandidates = step2acquisition.optimize(
            surrogateModel,
            bounds,
            n=nMax,
            nondominated=out.x,
            paretoFront=out.fx,
        )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Random perturbation of the currently nondominated points: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 3. Minimum point sampling to examine the endpoints of the Pareto front
        #
        # Should all points be discarded, which may happen if the minima of
        # the surrogate surfaces do not change between iterations, we
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of sample points
        #
        t0 = time.time()
        bestCandidates = step3acquisition.optimize(
            surrogateModel, bounds, n=nMax
        )
        if len(bestCandidates) == 0:
            bestCandidates = maximizeDistance.optimize(
                surrogateModel, bounds, n=1
            )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Minimum point sampling: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 4. Uniform random points and scoring
        #
        t0 = time.time()
        bestCandidates = acquisitionFuncGlobal.optimize(
            surrogateModel, bounds, 1
        )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Uniform random points and scoring: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 5. Solving the surrogate multiobjective problem
        #
        t0 = time.time()
        bestCandidates = step5acquisition.optimize(
            surrogateModel, bounds, n=min(nMax, 2 * objdim)
        )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Solving the surrogate multiobjective problem: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 6. Discard selected points that are too close to each other
        #
        if xselected.size > 0:
            idxs = [0]
            for i in range(1, xselected.shape[0]):
                x = xselected[i, :].reshape(1, -1)
                if cdist(x, xselected[idxs, :]).min() >= tol:
                    idxs.append(i)
            xselected = xselected[idxs, :]
        else:
            ySelected = np.empty((0, objdim))
            out.nit = out.nit + 1
            print(
                "Acquisition function has failed to find a new sample! "
                "Consider modifying it."
            )
            break

        #
        # 7. Evaluate the objective function and update the Pareto front
        #

        batchSize = min(len(xselected), maxeval - out.nfev)
        xselected.resize(batchSize, dim)
        print("Number of new sample points: ", batchSize)

        # Compute f(xselected)
        ySelected = np.asarray(fun(xselected))

        # Update the Pareto front
        out.x = np.concatenate((out.x, xselected), axis=0)
        out.fx = np.concatenate((out.fx, ySelected), axis=0)
        iPareto = find_pareto_front(out.fx)
        out.x = out.x[iPareto, :]
        out.fx = out.fx[iPareto, :]

        # Update sample and fsample in out
        out.sample[out.nfev : out.nfev + batchSize, :] = xselected
        out.fsample[out.nfev : out.nfev + batchSize, :] = ySelected

        # Update the counters
        out.nfev = out.nfev + batchSize
        out.nit = out.nit + 1

        # Call the callback function
        if callback is not None:
            callback(out)

    # Update output
    out.sample.resize(out.nfev, dim)
    out.fsample.resize(out.nfev, objdim)

    # Update surrogate model if it lives outside the function scope
    if return_surrogate:
        t0 = time.time()
        if out.nfev > 0:
            surrogateModel.update(xselected, ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

    return out
