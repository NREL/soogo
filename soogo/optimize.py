"""Optimization algorithms for soogo."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC
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

__authors__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
    "Byron Selvage",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
    "Byron Selvage",
]
__deprecated__ = False

from typing import Callable, Optional
import numpy as np
import time
from copy import deepcopy

# Scipy imports
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

# PyMoo imports
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.pso import PSO

# Local imports
from .model.base import Surrogate
from .acquisition import (
    WeightedAcquisition,
    CoordinatePerturbationOverNondominated,
    EndPointsParetoFront,
    GosacSample,
    MaximizeEI,
    MinimizeMOSurrogate,
    ParetoFront,
    TargetValueAcquisition,
    AcquisitionFunction,
    MinimizeSurrogate,
)
from .utils import find_pareto_front, evaluate_and_log_point, uncertainty_score
from .model import MedianLpfFilter, RbfModel, GaussianProcess
from .sampling import NormalSampler, Sampler, SamplingStrategy, Mitchel91Sampler
from .optimize_result import OptimizeResult
from .termination import UnsuccessfulImprovement, RobustCondition
from .problem import PymooProblem


def surrogate_optimization(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    acquisitionFunc: Optional[AcquisitionFunction] = None,
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
        :class:`RbfModel` model with median low-pass filter is used.
        On exit, if provided, the surrogate model the points used during the
        optimization.
    :param acquisitionFunc: Acquisition function to be used. If None is
        provided, the :class:`TargetValueAcquisition` is used.
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
        acquisitionFunc = TargetValueAcquisition()

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
            surrogateModel, bounds, batchSize, xbest=out.x
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


def multistart_msrs(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    batchSize: int = 1,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a response
    surface model approach with restarts.

    This implementation generalizes the algorithms Multistart LMSRS from [#]_.
    The general algorithm calls :func:`surrogate_optimization()` successive
    times until there are no more function evaluations available. The first
    time :func:`surrogate_optimization()` is called with the given, if any, trained
    surrogate model. Other function calls use an empty surrogate model. This is
    done to enable truly different starting samples each time.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. If None is provided, a
        :class:`RbfModel` model with median low-pass filter is used.
    :param batchSize: Number of new sample points to be generated per iteration.
    :param disp: If True, print information about the optimization process.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result.
    :return: The optimization result.

    References
    ----------
    .. [#] Rommel G Regis and Christine A Shoemaker. A stochastic radial basis
        function method for the global optimization of expensive functions.
        INFORMS Journal on Computing, 19(4):497–509, 2007.
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize output
    out = OptimizeResult(
        x=np.empty(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        sample=np.zeros((maxeval, dim)),
        fsample=np.zeros(maxeval),
    )

    # Copy the surrogate model
    _surrogateModel = deepcopy(surrogateModel)

    # do until max number of f-evals reached
    while out.nfev < maxeval:
        # Acquisition function
        acquisitionFunc = WeightedAcquisition(
            NormalSampler(
                min(1000 * dim, 10000), 0.1, strategy=SamplingStrategy.NORMAL
            ),
            weightpattern=(0.95,),
            termination=RobustCondition(
                UnsuccessfulImprovement(), max(5, dim)
            ),
            sigma_min=0.1 * 0.5**5,
        )
        acquisitionFunc.success_period = maxeval  # to never increase sigma

        # Run local optimization
        out_local = surrogate_optimization(
            fun,
            bounds,
            maxeval - out.nfev,
            surrogateModel=_surrogateModel,
            acquisitionFunc=acquisitionFunc,
            batchSize=batchSize,
            disp=disp,
            callback=callback,
        )

        # Update output
        if out_local.fx < out.fx:
            out.x[:] = out_local.x
            out.fx = out_local.fx
        out.sample[out.nfev : out.nfev + out_local.nfev, :] = out_local.sample
        out.fsample[out.nfev : out.nfev + out_local.nfev] = out_local.fsample
        out.nfev = out.nfev + out_local.nfev

        # Update counters
        out.nit = out.nit + 1

        # Reset the surrogate model
        if _surrogateModel is not None:
            _surrogateModel.reset_data()

    return out


def dycors(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    acquisitionFunc: Optional[WeightedAcquisition] = None,
    batchSize: int = 1,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
):
    """DYCORS algorithm for single-objective optimization

    Implementation of the DYCORS (DYnamic COordinate search using Response
    Surface models) algorithm proposed in [#]_. That is a wrapper to
    :func:`surrogate_optimization()`.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. If None is provided, a
        :class:`RbfModel` model with median low-pass filter is used.
        On exit, if provided, the surrogate model the points used during the
        optimization.
    :param acquisitionFunc: Acquisition function to be used. If None is
        provided, the acquisition function is the one used in DYCORS-LMSRBF from
        Regis and Shoemaker (2012).
    :param batchSize: Number of new sample points to be generated per iteration.
    :param disp: If True, print information about the optimization process.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result.
    :return: The optimization result.

    References
    ----------
    .. [#] Regis, R. G., & Shoemaker, C. A. (2012). Combining radial basis
        function surrogates and dynamic coordinate search in
        high-dimensional expensive black-box optimization.
        Engineering Optimization, 45(5), 529–555.
        https://doi.org/10.1080/0305215X.2012.687731
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize acquisition function
    if acquisitionFunc is None:
        acquisitionFunc = WeightedAcquisition(
            NormalSampler(
                min(100 * dim, 5000), 0.2, strategy=SamplingStrategy.DDS
            ),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            maxeval=maxeval,
            sigma_min=0.2 * 0.5**6,
            sigma_max=0.2,
        )

    return surrogate_optimization(
        fun,
        bounds,
        maxeval,
        surrogateModel=surrogateModel
        if surrogateModel is not None
        else RbfModel(filter=MedianLpfFilter()),
        acquisitionFunc=acquisitionFunc,
        batchSize=batchSize,
        disp=disp,
        callback=callback,
    )


def cptv(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[RbfModel] = None,
    acquisitionFunc: Optional[WeightedAcquisition] = None,
    improvementTol: float = 1e-3,
    consecutiveQuickFailuresTol: int = 0,
    useLocalSearch: bool = False,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using the coordinate
    perturbation and target value strategy.

    This is an implementation of the algorithm desribed in [#]_. The algorithm
    uses a sequence of different acquisition functions as follows:

        1. CP step: :func:`surrogate_optimization()` with `acquisitionFunc`. Ideally,
            this step would use a :class:`WeightedAcquisition` object with a
            :class:`NormalSampler` sampler. The implementation is configured to
            use the acquisition proposed by Müller (2016) by default.

        2. TV step: :func:`surrogate_optimization()` with a
            :class:`TargetValueAcquisition` object.

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
        :class:`RbfModel` model with median low-pass filter is used.
        On exit, if provided, the surrogate model the points used during the
        optimization.
    :param acquisitionFunc: Acquisition function to be used. If None is
        provided, a :class:`WeightedAcquisition` is used following what is
        described by Müller (2016).
    :param improvementTol: Expected improvement in the global optimum per
        iteration.
    :param consecutiveQuickFailuresTol: Number of times that the CP step or the
        TV step fails quickly before the
        algorithm stops. The default is 0, which means the algorithm will stop
        after ``maxeval`` function evaluations. A quick failure is when the
        acquisition function in the CP or TV step does not find any significant
        improvement.
    :param useLocalSearch: If True, the algorithm will perform a continuous
        local search when a significant improvement is not found in a sequence
        of (CP,TV,CP) steps.
    :param disp: If True, print information about the optimization process.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result.
    :return: The optimization result.

    References
    ----------
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Tolerance parameters
    nFailTol = max(5, dim)  # Fail tolerance for the CP step

    # Initialize optional variables
    if surrogateModel is None:
        surrogateModel = RbfModel(filter=MedianLpfFilter())
    if consecutiveQuickFailuresTol == 0:
        consecutiveQuickFailuresTol = maxeval
    if acquisitionFunc is None:
        acquisitionFunc = WeightedAcquisition(
            NormalSampler(
                min(500 * dim, 5000),
                0.2,
                strategy=SamplingStrategy.DDS,
            ),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            rtol=1e-6,
            maxeval=maxeval,
            sigma_min=0.2 * 0.5**6,
            sigma_max=0.2,
            termination=RobustCondition(
                UnsuccessfulImprovement(improvementTol), nFailTol
            ),
        )

    tv_acquisition = TargetValueAcquisition(
        cycleLength=10,
        rtol=acquisitionFunc.rtol,
        termination=RobustCondition(
            UnsuccessfulImprovement(improvementTol), 12
        ),
    )

    # Get index and bounds of the continuous variables
    cindex = [i for i in range(dim) if i not in surrogateModel.iindex]
    cbounds = [bounds[i] for i in cindex]

    # Initialize output
    out = OptimizeResult(
        x=np.empty(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        sample=np.zeros((maxeval, dim)),
        fsample=np.zeros(maxeval),
    )

    # do until max number of f-evals reached
    method = 0
    consecutiveQuickFailures = 0
    localSearchCounter = 0
    k = 0
    while (
        out.nfev < maxeval
        and consecutiveQuickFailures < consecutiveQuickFailuresTol
    ):
        if method == 0:
            # Reset acquisition parameters
            acquisitionFunc.sampler.neval = out.nfev
            acquisitionFunc.sampler.sigma = acquisitionFunc.sigma_max
            acquisitionFunc.best_known_x = np.copy(out.x)
            acquisitionFunc.success_count = 0
            acquisitionFunc.failure_count = 0
            acquisitionFunc.termination.update(out, surrogateModel)
            acquisitionFunc.termination.reset(keep_data_knowledge=True)

            # Run the CP step
            out_local = surrogate_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                surrogateModel=surrogateModel,
                acquisitionFunc=acquisitionFunc,
                disp=disp,
            )

            # Check for quick failure
            if out_local.nit <= nFailTol:
                consecutiveQuickFailures += 1
            else:
                consecutiveQuickFailures = 0

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
            tv_acquisition.termination.update(out, surrogateModel)
            tv_acquisition.termination.reset(keep_data_knowledge=True)

            out_local = surrogate_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                surrogateModel=surrogateModel,
                acquisitionFunc=tv_acquisition,
                disp=disp,
            )

            if out_local.nit <= 12:
                consecutiveQuickFailures += 1
            else:
                consecutiveQuickFailures = 0

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

            out_local = OptimizeResult(
                x=out.x.copy(),
                fx=out_local_.fun,
                nit=out_local_.nit,
                nfev=out_local_.nfev,
                sample=np.array([out.x for i in range(out_local_.nfev)]),
                fsample=np.array([out.fx for i in range(out_local_.nfev)]),
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
    """Wrapper to cptv. See :func:`cptv()`."""
    if "useLocalSearch" in kwargs:
        assert kwargs["useLocalSearch"] is True, (
            "`useLocalSearch` must be True for `cptvl`."
        )
    else:
        kwargs["useLocalSearch"] = True
    return cptv(*args, **kwargs)


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
    :param bounds: List with the limits [x_min,x_max] of each direction x in the search
        space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Multi-target surrogate model to be used. If None is provided, a
        :class:`RbfModel` model is used.
    :param acquisitionFunc: Acquisition function to be used in the CP step. The default is
        WeightedAcquisition(0).
    :param acquisitionFuncGlobal: Acquisition function to be used in the global step. The default is
        WeightedAcquisition(Sampler(0), 0.95).
    :param disp: If True, print information about the optimization process. The default
        is False.
    :param callback: If provided, the callback function will be called after each iteration
        with the current optimization result. The default is None.
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
        t0 = time.time()
        bestCandidates = step3acquisition.optimize(
            surrogateModel, bounds, n=nMax
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
    :param gfun: The constraint function to be minimized. The constraints must be
        formulated as g(x) <= 0.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the search
        space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used for the constraints. If None is provided, a
        :class:`RbfModel` model is used.
    :param disp: If True, print information about the optimization process. The default
        is False.
    :param callback: If provided, the callback function will be called after each iteration
        with the current optimization result. The default is None.
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


def bayesian_optimization(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    acquisitionFunc: Optional[MaximizeEI] = None,
    batchSize: int = 1,
    disp: bool = False,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables via active learning
    of a Gaussian Process model.

    See [#]_ for details.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the search
        space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Gaussian Process surrogate model. The default is GaussianProcess().
        On exit, if provided, the surrogate model the points used during the
        optimization.
    :param acquisitionFunc: Acquisition function to be used.
    :param batchSize: Number of new sample points to be generated per iteration. The default is 1.
    :param disp: If True, print information about the optimization process. The default
        is False.
    :param callback: If provided, the callback function will be called after each iteration
        with the current optimization result. The default is None.
    :return: The optimization result.

    References
    ----------
    .. [#] Che Y, Müller J, Cheng C. Dispersion-enhanced sequential batch
        sampling for adaptive contour estimation. Qual Reliab Eng Int. 2024;
        40: 131–144. https://doi.org/10.1002/qre.3245
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0
    tol = 1e-6 * np.min([abs(b[1] - b[0]) for b in bounds])

    # Initialize optional variables
    return_surrogate = True
    if surrogateModel is None:
        return_surrogate = False
        surrogateModel = GaussianProcess()
    if acquisitionFunc is None:
        acquisitionFunc = MaximizeEI()
    if acquisitionFunc.sampler.n <= 1:
        acquisitionFunc.sampler.n = min(100 * dim, 1000)

    # Initialize output
    out = OptimizeResult()
    out.init(fun, bounds, batchSize, maxeval, surrogateModel)
    out.init_best_values(surrogateModel)

    # Call the callback function
    if callback is not None:
        callback(out)

    # do until max number of f-evals reached or local min found
    xselected = np.copy(out.sample[0 : out.nfev, :])
    ySelected = np.copy(out.fsample[0 : out.nfev])
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

        xselected = np.empty((0, dim))

        # Use the current minimum of the GP in the last iteration
        if out.nfev + batchSize == maxeval:
            t0 = time.time()
            res = differential_evolution(
                lambda x: surrogateModel(np.asarray([x])), bounds
            )
            if res.x is not None:
                if cdist([res.x], surrogateModel.X).min() >= tol:
                    xselected = np.concatenate((xselected, [res.x]), axis=0)
            tf = time.time()
            if disp:
                print(
                    "Time to acquire the minimum of the GP: %f s" % (tf - t0)
                )

        # Acquire new sample points through minimization of EI
        t0 = time.time()
        xMinEI = acquisitionFunc.optimize(
            surrogateModel, bounds, batchSize - len(xselected), ybest=out.fx
        )
        if len(xselected) > 0:
            aux = cdist(xselected, xMinEI)[0]
            xselected = np.concatenate((xselected, xMinEI[aux >= tol]), axis=0)
        else:
            xselected = xMinEI
        tf = time.time()
        if disp:
            print(
                "Time to acquire new sample points using acquisitionFunc: %f s"
                % (tf - t0)
            )

        # Compute f(xselected)
        selectedBatchSize = len(xselected)
        ySelected = np.asarray(fun(xselected))

        # Update best point found so far if necessary
        iSelectedBest = np.argmin(ySelected).item()
        fxSelectedBest = ySelected[iSelectedBest]
        if fxSelectedBest < out.fx:
            out.x[:] = xselected[iSelectedBest, :]
            out.fx = fxSelectedBest

        # Update remaining output variables
        out.sample[out.nfev : out.nfev + selectedBatchSize, :] = xselected
        out.fsample[out.nfev : out.nfev + selectedBatchSize] = ySelected
        out.nfev = out.nfev + selectedBatchSize
        out.nit = out.nit + 1

        # Call the callback function
        if callback is not None:
            callback(out)

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


def fsapso(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    disp: bool = False,
) -> OptimizeResult:
    """
    Minimize a scalar function of one or more variables using the fast
    surrogate-assisted particle swarm optimization (FSAPSO) algorithm
    presented in _[#].

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. If None is provided, a
        :class:`RbfModel` model with cubic kernel is used. On exit, if provided,
        the surrogate model will contain the points used during the
        optimization.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result. The default is
        None.
    :param disp: If True, print information about the optimization process.
        The default is False.

    :return: The optimization result.


    References
    ----------
    .. [#] Li, F., Shen, W., Cai, X., Gao, L., & Gary Wang, G. 2020; A fast
    surrogate-assisted particle swarm optimization algorithm for computationally
    expensive problems. Applied Soft Computing, 92, 106303.
    https://doi.org/10.1016/j.asoc.2020.106303
    """
    # Initialize parameters
    bounds = np.array(bounds)
    dim = len(bounds)

    lb = bounds[:, 0]
    ub = bounds[:, 1]
    vMax = 0.1 * (ub - lb)
    nSwarm = 20
    nInitial = min(max(dim, 20), maxeval)
    tol = np.min([np.sqrt(0.001**2 * dim), 5e-5 * dim * np.min(ub - lb)])

    # Initialize acquisition function(s)
    surrogateMinimizer = MinimizeSurrogate(nCand=1000)

    # Initialize surrogate
    if surrogateModel is None:
        surrogateModel = RbfModel()

    # Reserve space in the surrogate model
    surrogateModel.reserve(maxeval + surrogateModel.ntrain, dim)

    # Initialize output
    out = OptimizeResult(
        x=np.empty(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        sample=np.zeros((maxeval, dim)),
        fsample=np.zeros(maxeval),
    )

    if disp:
        print("Starting FSAPSO optimization...")

    # Initialize surrogate model
    if surrogateModel.ntrain == 0:
        sampler = Sampler(nInitial)
        xInit = sampler.get_slhd_sample(bounds.tolist())

        if disp:
            print(f"Evaluating {len(xInit)} initial points for surrogate...")

        # Evaluate initial points
        for x0 in xInit:
            _ = evaluate_and_log_point(fun, x0, out)

            if disp:
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

        # Build surrogate model
        surrogateModel.update(out.sample[0 : out.nfev], out.fsample[0 : out.nfev])

        if disp:
            print(f"Built surrogate model with {surrogateModel.ntrain} points")

    else:
        # Initialize best point in output
        out.x = surrogateModel.X[np.argmin(surrogateModel.Y)]
        out.fx = np.min(surrogateModel.Y)

        if disp:
            print(f"Using pre-trained surrogate with {surrogateModel.ntrain} points")

    # Select initial swarm
    if surrogateModel.ntrain >= nSwarm:
        # Take 20 best points as initial swarm
        bestIndices = np.argsort(surrogateModel.Y)[:nSwarm]
        swarmInitX = surrogateModel.X[bestIndices]

        if disp:
            print(f"Selected {nSwarm} best training points for initial swarm")

    else:
        # If not enough training data, use random sampling
        if disp:
            print("Not enough training data for initial swarm. Using random sampling to increase population.")

        swarmSampler = Sampler(nSwarm - surrogateModel.ntrain)
        swarmInitX = swarmSampler.get_slhd_sample(bounds.tolist())
        swarmInitX = np.vstack((swarmInitX, surrogateModel.X))

    surrogateProblem = PymooProblem(
        objfunc=lambda x: surrogateModel(x).reshape(-1, 1),
        bounds=bounds
    )

    # Initialize PSO algorithm
    pso = PSO(pop_size=nSwarm, c1=1.491, c2=1.491, max_velocity_rate=vMax, adaptive=False)
    pso.setup(surrogateProblem)

    # Set initial swarm positions
    initialPop = Population()
    for x in swarmInitX:
        ind = Individual(X=x)
        initialPop = Population.merge(initialPop, Population([ind]))

    # Evaluate initial swarm with surrogate
    pso.evaluator.eval(surrogateProblem, initialPop)

    # Set initial swarm population
    pso.pop = initialPop

    if disp:
        print("Starting main FSAPSO loop...")

    # Main FSAPSO loop
    prevGlobalBest = out.fx

    while out.nfev < maxeval and pso.has_next():
        improvedThisIter = False

        # Get minimum of surrogate
        xMin = surrogateMinimizer.optimize(surrogateModel, bounds, n=1)[0]

        # Check xMin is at least tol away from existing points
        if np.min(cdist(xMin.reshape(1, -1), out.sample[: out.nfev])) > tol:
            # Evaluate minimum with true objective
            fMin = evaluate_and_log_point(fun, xMin, out)

            if disp:
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

            # Update surrogate model with new point
            surrogateModel.update(xMin.reshape(1, -1), fMin)

            # If Improved, update PSO's global best
            if fMin < prevGlobalBest:
                improvedThisIter = True
                prevGlobalBest = fMin

                # Update PSO's global best
                pso.opt = Population.create(Individual(X=xMin, F=np.array([fMin])))

        # Update w value
        pso.w = 0.792 - (0.792 - 0.2) * out.nfev / maxeval

        # Update PSO velocities and positions
        swarm = pso.ask()

        # Evaluate particles with cheap surrogate
        pso.evaluator.eval(surrogateProblem, swarm)

        if out.nfev < maxeval:
            # Take swarm best
            fSurr = swarm.get("F")
            bestParticleIdx = np.argmin(fSurr)
            xBestParticle = swarm.get("X")[bestParticleIdx]

            # Evaluate best particle
            if np.min(cdist(xBestParticle.reshape(1, -1), out.sample[: out.nfev])) > tol:
                fBestParticle = evaluate_and_log_point(fun, xBestParticle, out)

                if disp:
                    print("fEvals: %d" % out.nfev)
                    print("Best value: %f" % out.fx)

                # Update surrogate with true evaluation
                surrogateModel.update(xBestParticle.reshape(1, -1), fBestParticle)

                # Update the particle's value in the swarm for PSO
                fUpdated = fSurr.copy()
                fUpdated[bestParticleIdx] = fBestParticle
                swarm.set("F", fUpdated)

                # Check if this improved global best
                if fBestParticle < prevGlobalBest:
                    improvedThisIter = True
                    prevGlobalBest = fBestParticle

                    # Update PSO's global best
                    pso.opt = Population.create(Individual(X=xBestParticle, F=np.array([fBestParticle])))

        # If no improvement, evaluate particle with greatest uncertainty
        if not improvedThisIter and out.nfev < maxeval:
            scores = uncertainty_score(swarm.get("X"), surrogateModel.X, surrogateModel.Y)
            xMostUncertain = swarm.get("X")[np.argmax(scores)]
            fMostUncertain = evaluate_and_log_point(fun, xMostUncertain, out)

            if disp:
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

            # Update surrogate
            surrogateModel.update(xMostUncertain.reshape(1, -1), fMostUncertain)

            # Update particle's fitness
            fFinal = swarm.get("F")
            fFinal[np.argmax(scores)] = fMostUncertain
            swarm.set("F", fFinal)

            # Check if this improved global best
            if fMostUncertain < prevGlobalBest:
                prevGlobalBest = fMostUncertain

                # Update PSO's global best
                pso.opt = Population.create(Individual(X=xMostUncertain, F=np.array([fMostUncertain])))

        # Tell PSO the results
        pso.tell(infills=swarm)

        # Call callback
        if callback is not None:
            callback(out)

    # Remove empty if PSO terminates before maxevals
    out.sample = out.sample[: out.nfev]
    out.fsample = out.fsample[: out.nfev]

    return out


def fake_fsapso(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    batchSize: int = 0,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    disp: bool = False,
) -> OptimizeResult:
    """
    Minimize a scalar function of one or more variables using the fast
    surrogate-assisted particle swarm optimization (FSAPSO) algorithm
    presented in _[#].

    This algorithm provides the batchSize parameter to control the number of
    points evaluated in each iteration. When batchSize < 3, the algorithm
    matches the implementation in the original FSAPSO paper. When batchSize > 3,
    it evaluates all three standard FSAPSO points (surrogate minimum,
    swarm best, and most uncertain) and batchSize - 3 additional points. The
    additional points are generated with mitchel91 sampling to fill the space.

    This is the simplest extension of FSAPSO to a batched algorithm. Using a
    more complex sampling strategy for the additional points could potentially
    improve performance, but this would also increase the complexity of the
    implementation.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. If None is provided, a
        :class:`RbfModel` model with cubic kernel is used. On exit, if provided,
        the surrogate model will contain the points used during the
        optimization.
    :param batchSize: The number of points to evaluate simultaneously.
        If < 3, evaluates surrogate minimum and swarm best as a batch, then
        evaluates most uncertain point only if no improvement was made.
        If >= 3, evaluates all three standard FSAPSO points (surrogate minimum,
        swarm best, and most uncertain) and batchSize - 3 additional points
        are evaluated together in a single batch evaluation. The additional
        points are generated with mitchel91 sampling.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result. The default is
        None.
    :param disp: If True, print information about the optimization process.
        The default is False.
    :return: The optimization result.

    References
    ----------
    .. [#] Li, F., Shen, W., Cai, X., Gao, L., & Gary Wang, G. 2020; A fast
    surrogate-assisted particle swarm optimization algorithm for computationally
    expensive problems. Applied Soft Computing, 92, 106303.
    https://doi.org/10.1016/j.asoc.2020.106303

    """
    # Initialize parameters
    bounds = np.array(bounds)
    dim = len(bounds)

    lb = bounds[:, 0]
    ub = bounds[:, 1]
    vMax = 0.1 * (ub - lb)
    nSwarm = 20

    tol = np.min([np.sqrt(0.001**2 * dim), 5e-5 * dim * np.min(ub - lb)])

    if batchSize < 3:
        batchSize = 2

    # Initialize acquisition function(s)
    surrogateMinimizer = MinimizeSurrogate(nCand=1000)

    # Initialize surrogate
    if surrogateModel is None:
        surrogateModel = RbfModel()

    # Reserve space in the surrogate model
    surrogateModel.reserve(maxeval + surrogateModel.ntrain, dim)

    # Set initial sampling design amount
    nInitial = min(max(dim, 20, batchSize, 2 * surrogateModel.min_design_space_size(dim)), maxeval)

    # Initialize output
    out = OptimizeResult(
        x=np.empty(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        sample=np.zeros((maxeval, dim)),
        fsample=np.zeros(maxeval),
    )

    if disp:
        print("Starting FSAPSO optimization...")

    # Initialize surrogate model
    if surrogateModel.ntrain == 0:

        sampler = Sampler(nInitial)
        xInit = sampler.get_slhd_sample(bounds.tolist())

        if disp:
            print(f"Evaluating {len(xInit)} initial points for surrogate...")

        # Evaluate initial points
        _ = evaluate_and_log_point(fun, xInit, out)

        if disp:
            print("fEvals: %d" % out.nfev)
            print("Best value: %f" % out.fx)

        # Build surrogate model
        surrogateModel.update(out.sample[0 : out.nfev], out.fsample[0 : out.nfev])

        if disp:
            print(f"Built surrogate model with {surrogateModel.ntrain} points")
    else:
        # Initialize best point in output
        out.x = surrogateModel.X[np.argmin(surrogateModel.Y)]
        out.fx = np.min(surrogateModel.Y)

        if disp:
            print(f"Using pre-trained surrogate with {surrogateModel.ntrain} points")

    # Select initial swarm
    if surrogateModel.ntrain >= nSwarm:

        # Take 20 best points as initial swarm
        bestIndices = np.argsort(surrogateModel.Y)[:nSwarm]
        swarmInitX = surrogateModel.X[bestIndices]

        if disp:
            print(f"Selected {nSwarm} best training points for initial swarm")
    else:
        # If not enough training data, use random sampling
        if disp:
            print("Not enough training data for initial swarm. Using random sampling to increase population.")

        swarmSampler = Sampler(nSwarm - surrogateModel.ntrain)
        swarmInitX = swarmSampler.get_slhd_sample(bounds.tolist())
        swarmInitX = np.vstack((swarmInitX, surrogateModel.X))

    surrogateProblem = PymooProblem(
        objfunc=lambda x: surrogateModel(x).reshape(-1, 1),
        bounds=bounds
    )

    # Initialize PSO algorithm
    pso = PSO(pop_size=nSwarm, c1=1.491, c2=1.491, max_velocity_rate=vMax, adaptive=False)
    pso.setup(surrogateProblem)

    # Set initial swarm positions
    initialPop = Population()
    for x in swarmInitX:
        ind = Individual(X=x)
        initialPop = Population.merge(initialPop, Population([ind]))

    # Evaluate initial swarm with surrogate
    pso.evaluator.eval(surrogateProblem, initialPop)

    # Set initial swarm population
    pso.pop = initialPop

    if disp:
        print("Starting main FSAPSO loop...")

    # Main FSAPSO loop
    prevGlobalBest = out.fx

    while out.nfev < maxeval:
        out.nit += 1

        # Update w value
        pso.w = 0.792 - (0.792 - 0.2) * out.nfev / maxeval

        # Update PSO velocities and positions
        swarm = pso.ask()

        # Evaluate particles with cheap surrogate
        pso.evaluator.eval(surrogateProblem, swarm)

        # Collect batch candidates
        batchCandidates = []
        batchCandidateInfo = []

        # Add surrogate minimum to batch
        xMin = surrogateMinimizer.optimize(surrogateModel, bounds, n=1)[0]
        if np.min(cdist(xMin.reshape(1, -1), out.sample[:out.nfev])) > tol:
            batchCandidates.append(xMin)
            batchCandidateInfo.append('surrogate_min')

        # Add swarm best to batch
        if batchSize > 1:
            fSurr = swarm.get("F")
            bestParticleIdx = np.argmin(fSurr)
            xBestParticle = swarm.get("X")[bestParticleIdx]
            if np.min(cdist(xBestParticle.reshape(1, -1), np.concatenate([out.sample[:out.nfev], batchCandidates]))) > tol:
                batchCandidates.append(xBestParticle)
                batchCandidateInfo.append(('swarm_best', bestParticleIdx))

        # If batchSize >= 3, add most uncertain point to batch
        if batchSize >= 3:
            scores = uncertainty_score(swarm.get("X"), surrogateModel.X, surrogateModel.Y)
            mostUncertainIdx = np.argmax(scores)
            xMostUncertain = swarm.get("X")[mostUncertainIdx]
            if np.min(cdist(xMostUncertain.reshape(1, -1), np.concatenate([out.sample[:out.nfev], batchCandidates]))) > tol:
                batchCandidates.append(xMostUncertain)
                batchCandidateInfo.append(('most_uncertain', mostUncertainIdx))

        # If batchSize > 3, fill remaining points using mitchel91 sampling
        if batchSize > 3:

            sampler = Mitchel91Sampler(batchSize - len(batchCandidates))
            mitchel91_samples = sampler.get_mitchel91_sample(bounds.tolist())
            batchCandidates.extend(mitchel91_samples)
            batchCandidateInfo.extend(['mitchel91_fill'] * len(mitchel91_samples))

        # Evaluate the batch
        improvedThisIter = False
        if batchCandidates and out.nfev < maxeval:
            batchX = np.array(batchCandidates)
            batchF = evaluate_and_log_point(fun, batchX, out)

            if disp:
                print(f"fEvals: {out.nfev}")
                print(f"Iterations: {out.nit}")
                print(f"Best value: {out.fx}")

            # Update the surrogate
            surrogateModel.update(batchX, batchF)

            # Ensure batchF is always an array for iteration
            batchF = np.atleast_1d(batchF)

            # Update the swarm best
            for i, (fVal, info) in enumerate(zip(batchF, batchCandidateInfo)):
                if fVal < prevGlobalBest:
                    improvedThisIter = True
                    prevGlobalBest = fVal
                    pso.opt = Population.create(Individual(X=batchCandidates[i], F=np.array([fVal])))

                # Update swarm particles if they correspond to swarm evaluations
                if isinstance(info, tuple):
                    if info[0] == 'swarm_best':
                        fUpdated = swarm.get("F").copy()
                        fUpdated[info[1]] = fVal
                        swarm.set("F", fUpdated)
                    elif info[0] == 'most_uncertain':
                        fFinal = swarm.get("F").copy()
                        fFinal[info[1]] = fVal
                        swarm.set("F", fFinal)

        # If batchSize < 3 and swarm best was not improved, find and evaluate
        # most uncertain point
        if batchSize < 3 and not improvedThisIter and out.nfev < maxeval:
            scores = uncertainty_score(swarm.get("X"), surrogateModel.X, surrogateModel.Y)
            mostUncertainIdx = np.argmax(scores)
            xMostUncertain = swarm.get("X")[mostUncertainIdx]

            if np.min(cdist(xMostUncertain.reshape(1, -1), out.sample[:out.nfev])) > tol:
                fMostUncertain = evaluate_and_log_point(fun, xMostUncertain, out)

                if disp:
                    print(f"fEvals: {out.nfev}")
                    print(f"Best value: {out.fx}")

                # Update surrogate
                surrogateModel.update(xMostUncertain.reshape(1, -1), fMostUncertain)

                # Update particle's fitness
                fFinal = swarm.get("F").copy()
                fFinal[mostUncertainIdx] = fMostUncertain
                swarm.set("F", fFinal)

                # Check if this improved global best
                if fMostUncertain < prevGlobalBest:
                    prevGlobalBest = fMostUncertain
                    pso.opt = Population.create(Individual(X=xMostUncertain, F=np.array([fMostUncertain])))

        # Tell PSO the results
        pso.tell(infills=swarm)

        # Call callback
        if callback is not None:
            callback(out)

        batchSize = min(batchSize, maxeval - out.nfev)

    # Remove empty if PSO terminates before maxevals
    out.sample = out.sample[: out.nfev]
    out.fsample = out.fsample[: out.nfev]

    return out
