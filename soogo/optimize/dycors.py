"""DYCORS optimization wrapper."""

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

from typing import Callable, Optional

from ..acquisition import WeightedAcquisition
from ..model import MedianLpfFilter, RbfModel
from ..model import Surrogate
from .utils import OptimizeResult
from ..sampling import NormalSampler, SamplingStrategy
from .surrogate_optimization import surrogate_optimization


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
