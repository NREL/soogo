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
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__deprecated__ = False

from typing import Callable, Optional, Union
import numpy as np
import time
from dataclasses import dataclass
from copy import deepcopy

# Scipy imports
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist

# Local imports
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
    find_pareto_front,
)
from .rbf import MedianLpfFilter, RbfModel
from .gp import GaussianProcess
from .sampling import NormalSampler, Sampler, SamplingStrategy


@dataclass
class OptimizeResult:
    """Optimization result for the global optimizers provided by this
    package."""

    x: Optional[np.ndarray] = None  #: Best sample point found so far
    fx: Union[float, np.ndarray, None] = None  #: Best objective function value
    nit: int = 0  #: Number of active learning iterations
    nfev: int = 0  #: Number of function evaluations taken
    sample: Optional[np.ndarray] = None  #: n-by-dim matrix with all n samples
    fsample: Optional[np.ndarray] = None  #: Vector with all n objective values

    def init(
        self, fun, bounds, mineval: int, maxeval: int, surrogateModel
    ) -> None:
        """Initialize :attr:`nfev` and :attr:`sample` and :attr:`fsample` with
        data about the optimization that is starting.

        This routine calls the objective function :attr:`nfev` times.

        :param fun: The objective function to be minimized.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param mineval: Minimum number of function evaluations to build the
            surrogate model.
        :param maxeval: Maximum number of function evaluations.
        :param surrogateModel: Surrogate model to be used.
        """
        dim = len(bounds)  # Dimension of the problem
        assert dim > 0

        # Local variables
        m0 = surrogateModel.ntrain()  # Number of initial sample points
        m_for_surrogate = surrogateModel.min_design_space_size(
            dim
        )  # Smallest sample for a valid surrogate
        iindex = surrogateModel.get_iindex()  # Integer design variables
        ydim = (
            surrogateModel.ydim()
            if callable(getattr(surrogateModel, "ydim", None))
            else 1
        )  # Dimension of the output

        # Initialize sample arrays in this object
        self.sample = np.empty((maxeval, dim))
        self.sample[:] = np.nan
        self.fsample = np.empty(maxeval if ydim <= 1 else (maxeval, ydim))
        self.fsample[:] = np.nan

        # If the surrogate is empty and no initial sample was given
        if m0 == 0:
            # Create a new sample with SLHD
            m = min(maxeval, max(mineval, 2 * m_for_surrogate))
            self.sample[0:m] = Sampler(m).get_slhd_sample(
                bounds, iindex=iindex
            )
            if m >= 2 * m_for_surrogate:
                count = 0
                while not surrogateModel.check_initial_design(
                    self.sample[0:m]
                ):
                    self.sample[0:m] = Sampler(m).get_slhd_sample(
                        bounds, iindex=iindex
                    )
                    count += 1
                    if count > 100:
                        raise RuntimeError(
                            "Cannot create valid initial design"
                        )

            # Compute f(sample)
            self.fsample[0:m] = fun(self.sample[0:m])
            self.nfev = m

    def init_best_values(self, surrogateModel) -> None:
        """Initialize :attr:`x` and :attr:`fx` based on the best values for the
        surrogate.

        :param surrogateModel: Surrogate model.
        """
        # Initialize self.x and self.fx
        assert self.sample is not None
        assert self.fsample is not None
        assert self.fsample.ndim == 1
        m = self.nfev

        iBest = np.argmin(
            np.concatenate((self.fsample[0:m], surrogateModel.ytrain()))
        ).item()
        if iBest < m:
            self.x = self.sample[iBest].copy()
            self.fx = self.fsample[iBest].item()
        else:
            self.x = surrogateModel.xtrain()[iBest - m].copy()
            self.fx = surrogateModel.ytrain()[iBest - m].item()


def initialize_moo_surrogate(
    fun,
    bounds,
    mineval: int,
    maxeval: int,
    *,
    surrogateModels=(RbfModel(),),
) -> OptimizeResult:
    """Initialize the surrogate model and the output of the optimization.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the search
        space.
    :param mineval: Minimum number of function evaluations to build the surrogate model.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModels: Surrogate models to be used. The default is (RbfModel(),).
    :return: The optimization result.
    """
    dim = len(bounds)  # Dimension of the problem
    objdim = len(surrogateModels)  # Dimension of the objective function
    assert dim > 0 and objdim > 0

    # Initialize output
    out = OptimizeResult(
        x=np.array([]),
        fx=np.array([]),
        nit=0,
        nfev=0,
        sample=np.zeros((maxeval, dim)),
        fsample=np.zeros((maxeval, objdim)),
    )

    # Number of initial sample points
    m0 = surrogateModels[0].ntrain()
    m_for_surrogate = surrogateModels[0].min_design_space_size(
        dim
    )  # Smallest sample for a valid surrogate
    m = 0

    # Add new sample to the surrogate model
    if m0 == 0:
        # Create a new sample with SLHD
        m = min(maxeval, max(mineval, 2 * m_for_surrogate))
        out.sample[0:m] = Sampler(m).get_slhd_sample(
            bounds, iindex=surrogateModels[0].iindex
        )
        if m >= 2 * m_for_surrogate:
            count = 0
            while not surrogateModels[0].check_initial_design(out.sample[0:m]):
                out.sample[0:m] = Sampler(m).get_slhd_sample(
                    bounds, iindex=surrogateModels[0].iindex
                )
                count += 1
                if count > 100:
                    raise RuntimeError("Cannot create valid initial design")

        # Compute f(sample)
        out.fsample[0:m] = fun(out.sample[0:m])
        out.nfev = m

        # Update surrogate
        for i in range(objdim):
            surrogateModels[i].update(out.sample[0:m], out.fsample[0:m, i])

    # Create the pareto front
    fallpoints = np.concatenate(
        (
            np.transpose(
                [surrogateModels[i].ytrain()[:m0] for i in range(objdim)]
            ),
            out.fsample[0:m, :],
        ),
        axis=0,
    )
    iPareto = find_pareto_front(fallpoints)
    out.x = surrogateModels[0].xtrain()[iPareto, :].copy()
    out.fx = fallpoints[iPareto, :]

    return out


def initialize_surrogate_constraints(
    fun,
    gfun,
    bounds,
    mineval: int,
    maxeval: int,
    *,
    surrogateModels=(RbfModel(),),
) -> OptimizeResult:
    """Initialize the surrogate models for the constraints.

    :param fun: The objective function to be minimized.
    :param gfun: The constraint functions. Each constraint function must return a scalar
        value. If the constraint function returns a value greater than zero, it
        is considered a violation of the constraint.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the search
        space.
    :param mineval: Minimum number of function evaluations to build the surrogate model.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModels: Surrogate models to be used. The default is (RbfModel(),).
    :return: The optimization result.
    """
    dim = len(bounds)  # Dimension of the problem
    gdim = len(surrogateModels)  # Number of constraints
    assert dim > 0 and gdim > 0

    # Initialize output
    out = OptimizeResult(
        x=np.array([]),
        fx=np.array([]),
        nit=0,
        nfev=0,
        sample=np.zeros((maxeval, dim)),
        fsample=np.zeros((maxeval, 1 + gdim)),
    )
    bestfx = np.Inf

    # Number of initial sample points
    m0 = surrogateModels[0].ntrain()
    m_for_surrogate = surrogateModels[0].min_design_space_size(
        dim
    )  # Smallest sample for a valid surrogate

    # Add new sample to the surrogate model
    if m0 == 0:
        # Create a new sample with SLHD
        m = min(maxeval, max(mineval, 2 * m_for_surrogate))
        out.sample[0:m] = Sampler(m).get_slhd_sample(
            bounds, iindex=surrogateModels[0].iindex
        )
        if m >= 2 * m_for_surrogate:
            count = 0
            while not surrogateModels[0].check_initial_design(out.sample[0:m]):
                out.sample[0:m] = Sampler(m).get_slhd_sample(
                    bounds, iindex=surrogateModels[0].iindex
                )
                count += 1
                if count > 100:
                    raise RuntimeError("Cannot create valid initial design")

        # Compute f(sample) and g(sample)
        out.fsample[0:m, 0] = bestfx
        out.fsample[0:m, 1:] = gfun(out.sample[0:m])
        out.nfev = m

        # Update surrogate
        for i in range(gdim):
            surrogateModels[i].update(out.sample[0:m], out.fsample[0:m, i + 1])

        # Update best point found so far
        for i in range(m):
            if np.max(out.fsample[i, 1:]) <= 0:
                out.fsample[i, 0] = fun(out.sample[i, :].reshape(1, -1))
                if out.x.size == 0 or out.fsample[i, 0] < bestfx:
                    out.x = out.sample[i, :].copy()
                    out.fx = out.fsample[i, :].copy()
                    bestfx = out.fsample[i, 0]

    return out


def surrogate_optimization(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel=None,
    acquisitionFunc: Optional[AcquisitionFunction] = None,
    batchSize: int = 1,
    improvementTol: float = 1e-3,
    nSuccTol: int = 3,
    nFailTol: int = 5,
    performContinuousSearch: bool = True,
    termination=None,
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
    surrogateModel.reserve(surrogateModel.ntrain() + maxeval, dim)

    # Initialize output
    out = OptimizeResult()
    out.init(fun, bounds, batchSize, maxeval, surrogateModel)
    out.init_best_values(surrogateModel)

    # Call the callback function
    if callback is not None:
        callback(out)

    # counters
    failctr = 0  # number of consecutive unsuccessful iterations
    succctr = 0  # number of consecutive successful iterations
    remainingCountinuousSearch = (
        0  # number of consecutive iterations with continuous search remaining
    )

    # tolerance parameters
    if nSuccTol == 0:
        nSuccTol = maxeval
    if nFailTol == 0:
        nFailTol = maxeval

    # Continuous local search
    nMaxContinuousSearch = 0
    if performContinuousSearch:
        if isinstance(acquisitionFunc, WeightedAcquisition):
            if isinstance(acquisitionFunc.sampler, NormalSampler):
                nMaxContinuousSearch = len(acquisitionFunc.weightpattern)

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
        xselected = acquisitionFunc.acquire(
            surrogateModel,
            bounds,
            batchSize,
            xbest=out.x,
            countinuousSearch=(remainingCountinuousSearch > 0),
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

        # determine if significant improvement
        iSelectedBest = np.argmin(ySelected).item()
        fxSelectedBest = ySelected[iSelectedBest]
        if (out.fx - fxSelectedBest) >= improvementTol * (
            out.fsample.max() - out.fx
        ):
            # "significant" improvement
            failctr = 0
            if remainingCountinuousSearch == 0:
                succctr = succctr + 1
            elif performContinuousSearch:
                remainingCountinuousSearch = nMaxContinuousSearch
        elif remainingCountinuousSearch == 0:
            failctr = failctr + 1
            succctr = 0
        else:
            remainingCountinuousSearch = (
                remainingCountinuousSearch - selectedBatchSize
            )

        # determine best one of newly sampled points
        modifiedCoordinates = [False] * dim
        if fxSelectedBest < out.fx:
            modifiedCoordinates = [
                xselected[iSelectedBest, i] != out.x[i] for i in range(dim)
            ]
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

        if remainingCountinuousSearch == 0:
            # Activate continuous search if an integer variables have changed and
            # a significant improvement was found
            if failctr == 0 and performContinuousSearch:
                intCoordHasChanged = False
                for i in surrogateModel.iindex:
                    if modifiedCoordinates[i]:
                        intCoordHasChanged = True
                        break
                if intCoordHasChanged:
                    remainingCountinuousSearch = nMaxContinuousSearch

            # Update counters and acquisition
            if isinstance(acquisitionFunc, WeightedAcquisition):
                if isinstance(acquisitionFunc.sampler, NormalSampler):
                    if failctr >= nFailTol:
                        acquisitionFunc.sampler.sigma *= 0.5
                        if (
                            acquisitionFunc.sampler.sigma
                            < acquisitionFunc.sampler.sigma_min
                        ):
                            # Algorithm is probably in a local minimum!
                            acquisitionFunc.sampler.sigma = (
                                acquisitionFunc.sampler.sigma_min
                            )
                        else:
                            failctr = 0
                    elif succctr >= nSuccTol:
                        acquisitionFunc.sampler.sigma = min(
                            2 * acquisitionFunc.sampler.sigma,
                            acquisitionFunc.sampler.sigma_max,
                        )
                        succctr = 0

        # Check for convergence
        if failctr >= nFailTol and termination == "nFailTol":
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
    surrogateModel=None,
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

    # do until max number of f-evals reached
    while out.nfev < maxeval:
        # Run local optimization
        out_local = surrogate_optimization(
            fun,
            bounds,
            maxeval - out.nfev,
            surrogateModel=deepcopy(surrogateModel),
            acquisitionFunc=WeightedAcquisition(
                NormalSampler(
                    min(1000 * dim, 10000),
                    0.1,
                    sigma_min=0.1 * 0.5**5,
                    strategy=SamplingStrategy.NORMAL,
                ),
                weightpattern=(0.95,),
            ),
            batchSize=batchSize,
            nSuccTol=maxeval,
            nFailTol=max(5, dim),
            performContinuousSearch=True,
            termination="nFailTol",
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

    return out


def dycors(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel=None,
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
                min(100 * dim, 5000),
                0.2,
                sigma_min=0.2 * 0.5**6,
                sigma_max=0.2,
                strategy=SamplingStrategy.DDS,
            ),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            maxeval=maxeval,
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
        nSuccTol=3,
        nFailTol=max(dim, 5),
        performContinuousSearch=True,
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
                sigma_min=0.2 * 0.5**6,
                sigma_max=0.2,
                strategy=SamplingStrategy.DDS,
            ),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            rtol=1e-6,
            maxeval=maxeval,
        )

    # Tolerance parameters
    nFailTol = max(5, dim)  # Fail tolerance for the CP step
    tol = acquisitionFunc.tol(bounds)  # Tolerance for excluding points

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
            out_local = surrogate_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                surrogateModel=surrogateModel,
                acquisitionFunc=acquisitionFunc,
                performContinuousSearch=True,
                improvementTol=improvementTol,
                nSuccTol=3,
                nFailTol=nFailTol,
                termination="nFailTol",
                disp=disp,
            )

            # Reset perturbation range
            acquisitionFunc.sampler.sigma = acquisitionFunc.sampler.sigma_max

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
            out_local = surrogate_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                surrogateModel=surrogateModel,
                acquisitionFunc=TargetValueAcquisition(
                    cycleLength=10, rtol=acquisitionFunc.rtol
                ),
                improvementTol=improvementTol,
                nFailTol=12,
                termination="nFailTol",
                disp=disp,
            )

            if out_local.nit <= 12:
                consecutiveQuickFailures += 1
            else:
                consecutiveQuickFailures = 0

            # Update neval in the CP acquisition function
            acquisitionFunc.neval += out_local.nfev

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
                return fun(x_)[0]

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

            # Update neval in the CP acquisition function
            acquisitionFunc.neval += out_local.nfev

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
    surrogateModels,
    *,
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
    :param surrogateModels: Surrogate models to be used.
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
    objdim = len(surrogateModels)  # Number of objective functions
    assert dim > 0 and objdim > 1

    # Initialize optional variables
    if acquisitionFunc is None:
        acquisitionFunc = WeightedAcquisition(NormalSampler(0, 0.1))
    if acquisitionFuncGlobal is None:
        acquisitionFuncGlobal = WeightedAcquisition(Sampler(0), 0.95)

    # Use a number of candidates that is greater than 1
    if acquisitionFunc.sampler.n <= 1:
        acquisitionFunc.sampler.n = min(500 * dim, 5000)
    if acquisitionFuncGlobal.sampler.n <= 1:
        acquisitionFuncGlobal.sampler.n = min(500 * dim, 5000)

    # Reserve space for the surrogate model to avoid repeated allocations
    for s in surrogateModels:
        s.reserve(s.ntrain() + maxeval, dim)

    # Initialize output
    out = initialize_moo_surrogate(
        fun,
        bounds,
        0,
        maxeval,
        surrogateModels=surrogateModels,
    )
    assert isinstance(out.fx, np.ndarray)

    # Define acquisition functions
    tol = acquisitionFunc.tol(bounds)
    step1acquisition = ParetoFront()
    step2acquisition = CoordinatePerturbationOverNondominated(acquisitionFunc)
    step3acquisition = EndPointsParetoFront(rtol=acquisitionFunc.rtol)
    step5acquisition = MinimizeMOSurrogate(rtol=acquisitionFunc.rtol)

    # do until max number of f-evals reached or local min found
    xselected = np.empty((0, dim))
    ySelected = np.copy(out.fsample[0 : out.nfev, :])
    while out.nfev < maxeval:
        nMax = maxeval - out.nfev
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % out.nfev)

        # Update surrogate models
        t0 = time.time()
        if out.nfev > 0:
            for i in range(objdim):
                surrogateModels[i].update(xselected, ySelected[:, i])
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        #
        # 0. Adjust parameters in the acquisition
        #
        acquisitionFunc.neval = acquisitionFunc.maxeval - (maxeval - out.nfev)

        #
        # 1. Define target values to fill gaps in the Pareto front
        #
        t0 = time.time()
        xselected = step1acquisition.acquire(
            surrogateModels,
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
        bestCandidates = step2acquisition.acquire(
            surrogateModels,
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
        bestCandidates = step3acquisition.acquire(
            surrogateModels, bounds, n=nMax
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
        bestCandidates = acquisitionFuncGlobal.acquire(
            surrogateModels, bounds, 1
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
        bestCandidates = step5acquisition.acquire(
            surrogateModels, bounds, n=min(nMax, 2 * objdim)
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
    t0 = time.time()
    if out.nfev > 0:
        for i in range(objdim):
            surrogateModels[i].update(xselected, ySelected[:, i])
    tf = time.time()
    if disp:
        print("Time to update surrogate model: %f s" % (tf - t0))

    return out


def gosac(
    fun,
    gfun,
    bounds,
    maxeval: int,
    surrogateModels,
    *,
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
    :param surrogateModels: Surrogate models to be used.
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
    gdim = len(surrogateModels)  # Number of constraints
    assert dim > 0 and gdim > 0

    # Reserve space for the surrogate model to avoid repeated allocations
    for s in surrogateModels:
        s.reserve(s.ntrain() + maxeval, dim)

    # Initialize output
    out = initialize_surrogate_constraints(
        fun,
        gfun,
        bounds,
        0,
        maxeval,
        surrogateModels=surrogateModels,
    )
    assert isinstance(out.fx, np.ndarray)

    # Acquisition functions
    rtol = 1e-3
    acquisition1 = MinimizeMOSurrogate(rtol=rtol)
    acquisition2 = GosacSample(fun, rtol=rtol)

    xselected = np.empty((0, dim))
    ySelected = np.copy(out.fsample[0 : out.nfev, 1:])

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
            for i in range(gdim):
                surrogateModels[i].update(xselected, ySelected[:, i])
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Solve the surrogate multiobjective problem
        t0 = time.time()
        bestCandidates = acquisition1.acquire(surrogateModels, bounds, n=0)
        tf = time.time()
        if disp:
            print(
                "Solving the surrogate multiobjective problem: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        # Evaluate the surrogate at the best candidates
        sCandidates = np.empty((len(bestCandidates), gdim))
        for i in range(gdim):
            sCandidates[:, i], _ = surrogateModels[i](bestCandidates)

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
            out.fsample[out.nfev, 0] = np.Inf

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
            for i in range(gdim):
                surrogateModels[i].update(xselected, ySelected[:, i])
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
            for i in range(gdim):
                surrogateModels[i].update(xselected, ySelected[:, i])
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Solve cheap problem with multiple constraints
        t0 = time.time()
        xselected = acquisition2.acquire(surrogateModels, bounds)
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
            out.fsample[out.nfev, 0] = np.Inf

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
    t0 = time.time()
    if out.nfev > 0:
        for i in range(gdim):
            surrogateModels[i].update(xselected, ySelected[:, i])
    tf = time.time()
    if disp:
        print("Time to update surrogate model: %f s" % (tf - t0))

    return out


def bayesian_optimization(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel=None,
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
                lambda x: surrogateModel(np.asarray([x]))[0], bounds
            )
            if res.x is not None:
                if cdist([res.x], surrogateModel.xtrain()).min() >= tol:
                    xselected = np.concatenate((xselected, [res.x]), axis=0)
            tf = time.time()
            if disp:
                print(
                    "Time to acquire the minimum of the GP: %f s" % (tf - t0)
                )

        # Acquire new sample points through minimization of EI
        t0 = time.time()
        xMinEI = acquisitionFunc.acquire(
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
