"""Bayesian optimization routine using Gaussian Processes."""

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

from ..acquisition import MaximizeEI
from ..model import GaussianProcess
from .utils import OptimizeResult
from .surrogate_optimization import surrogate_optimization


def bayesian_optimization(*args, **kwargs) -> OptimizeResult:
    """Wrapper for :func:`.surrogate_optimization()` using a Gaussian Process
    surrogate model and the Expected Improvement acquisition function.
    """
    # Initialize optional variables
    if "surrogateModel" not in kwargs or kwargs["surrogateModel"] is None:
        kwargs["surrogateModel"] = GaussianProcess(normalize_y=True)
    if "acquisitionFunc" not in kwargs or kwargs["acquisitionFunc"] is None:
        kwargs["acquisitionFunc"] = MaximizeEI()

    return surrogate_optimization(*args, **kwargs)
