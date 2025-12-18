"""Multistart LMSRS optimization routine."""

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

from copy import deepcopy
from typing import Callable, Optional

import numpy as np

from ..acquisition import WeightedAcquisition
from ..model import Surrogate
from .utils import OptimizeResult
from ..sampling import NormalSampler, SamplingStrategy
from ..termination import RobustCondition, UnsuccessfulImprovement
from .surrogate_optimization import surrogate_optimization


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
    The general algorithm calls :func:`.surrogate_optimization()` successive
    times until there are no more function evaluations available. The first
    time :func:`.surrogate_optimization()` is called with the given, if
    any, trained surrogate model. Other function calls use an empty
    surrogate model. This is done to enable truly different starting samples
    each time.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. Only used as input, not
        updated. If None is provided, :func:`.surrogate_optimization()` will
        choose a default model.
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
