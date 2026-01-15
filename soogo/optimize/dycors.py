"""DYCORS optimization wrapper."""

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

import numpy as np
from typing import Optional

from ..acquisition import CoordinatePerturbation, Acquisition, BoundedParameter
from .surrogate_optimization import surrogate_optimization


def dycors(
    *args, acquisitionFunc: Optional[Acquisition] = None, seed=None, **kwargs
):
    """DYCORS algorithm for single-objective optimization implemented as a
    wrapper to :func:`.surrogate_optimization()`.

    Implementation of the DYCORS (DYnamic COordinate search using Response
    Surface models) algorithm proposed in [#]_. The acquisition function, if not
    provided, is the one used in DYCORS-LMSRBF from Regis and Shoemaker (2012).

    :param acquisitionFunc: Acquisition function to be used. If None, the
        DYCORS acquisition function is used.
    :param seed: Seed for random number generator.

    References
    ----------
    .. [#] Regis, R. G., & Shoemaker, C. A. (2012). Combining radial basis
        function surrogates and dynamic coordinate search in
        high-dimensional expensive black-box optimization.
        Engineering Optimization, 45(5), 529–555.
        https://doi.org/10.1080/0305215X.2012.687731
    """
    bounds = args[1] if len(args) > 1 else kwargs["bounds"]

    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize acquisition function
    if acquisitionFunc is None:
        rng = np.random.default_rng(seed)
        acquisitionFunc = CoordinatePerturbation(
            pool_size=min(100 * dim, 5000),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            sigma=BoundedParameter(0.2, 0.2 * 0.5**6, 0.2),
            n_continuous_search=4,
            seed=rng.integers(np.iinfo(np.int32).max).item(),
        )

    return surrogate_optimization(
        *args, acquisitionFunc=acquisitionFunc, seed=seed, **kwargs
    )
