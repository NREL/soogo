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

from ..acquisition import WeightedAcquisition
from ..sampling import NormalSampler, SamplingStrategy
from .surrogate_optimization import surrogate_optimization


def dycors(*args, **kwargs):
    """DYCORS algorithm for single-objective optimization implemented as a
    wrapper to :func:`.surrogate_optimization()`.

    Implementation of the DYCORS (DYnamic COordinate search using Response
    Surface models) algorithm proposed in [#]_. The acquisition function, if not
    provided, is the one used in DYCORS-LMSRBF from Regis and Shoemaker (2012).

    References
    ----------
    .. [#] Regis, R. G., & Shoemaker, C. A. (2012). Combining radial basis
        function surrogates and dynamic coordinate search in
        high-dimensional expensive black-box optimization.
        Engineering Optimization, 45(5), 529–555.
        https://doi.org/10.1080/0305215X.2012.687731
    """
    bounds = args[1] if len(args) > 1 else kwargs["bounds"]
    maxeval = args[2] if len(args) > 2 else kwargs["maxeval"]

    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize acquisition function
    if "acquisitionFunc" not in kwargs or kwargs["acquisitionFunc"] is None:
        kwargs["acquisitionFunc"] = WeightedAcquisition(
            NormalSampler(
                min(100 * dim, 5000), 0.2, strategy=SamplingStrategy.DDS
            ),
            weightpattern=(0.3, 0.5, 0.8, 0.95),
            maxeval=maxeval,
            sigma_min=0.2 * 0.5**6,
            sigma_max=0.2,
        )

    return surrogate_optimization(*args, **kwargs)
