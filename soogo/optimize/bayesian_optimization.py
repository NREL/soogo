"""Bayesian optimization routine using Gaussian Processes."""

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

from typing import Optional

from ..acquisition import MaximizeEI
from ..model import GaussianProcess
from ..model import Surrogate
from .utils import OptimizeResult
from .surrogate_optimization import surrogate_optimization


def bayesian_optimization(
    *args,
    surrogateModel: Optional[Surrogate] = None,
    acquisitionFunc: Optional[MaximizeEI] = None,
    **kwargs,
) -> OptimizeResult:
    """Wrapper for surrogate_optimization() using a Gaussian Process surrogate
    model and the Expected Improvement acquisition function.

    :param \\*args: Positional arguments passed to surrogate_optimization().
    :param surrogateModel: Gaussian Process surrogate model. The default is GaussianProcess().
        On exit, if provided, the surrogate model the points used during the
        optimization.
    :param acquisitionFunc: Acquisition function to be used.
    :param \\*\\*kwargs: Keyword arguments passed to surrogate_optimization().
    """
    # Initialize optional variables
    if surrogateModel is None:
        surrogateModel = GaussianProcess(normalize_y=True)
    if acquisitionFunc is None:
        acquisitionFunc = MaximizeEI()

    return surrogate_optimization(
        *args,
        surrogateModel=surrogateModel,
        acquisitionFunc=acquisitionFunc,
        **kwargs,
    )
