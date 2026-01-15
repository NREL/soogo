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

import numpy as np
from typing import Optional

from ..acquisition import MaximizeEI, Acquisition
from ..model import GaussianProcess
from .utils import OptimizeResult
from .surrogate_optimization import surrogate_optimization


def bayesian_optimization(
    *args,
    surrogateModel: Optional[GaussianProcess] = None,
    acquisitionFunc: Optional[Acquisition] = None,
    seed=None,
    **kwargs,
) -> OptimizeResult:
    """Wrapper for :func:`.surrogate_optimization()` using a Gaussian Process
    surrogate model and the Expected Improvement acquisition function.

    :param surrogateModel: Surrogate model to be used. If None, a Gaussian
        Process model is used.
    :param acquisitionFunc: Acquisition function to be used. If None, the
        Expected Improvement acquisition function is used.
    :param seed: Seed for random number generator.
    """
    # Initialize optional variables
    rng = np.random.default_rng(seed)
    if surrogateModel is None:
        surrogateModel = GaussianProcess(
            normalize_y=True,
            random_state=rng.integers(np.iinfo(np.int32).max).item(),
        )
    if acquisitionFunc is None:
        acquisitionFunc = MaximizeEI(seed=rng)

    return surrogate_optimization(
        *args,
        surrogateModel=surrogateModel,
        acquisitionFunc=acquisitionFunc,
        seed=seed,
        **kwargs,
    )
