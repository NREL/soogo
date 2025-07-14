"""soogo (Surrogate-based 0-th Order Global Optimization)"""

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

__all__ = [
    "acquisition",
    "optimize",
    "sampling",
    "rbf",
    "gp",
    "surrogate_optimization",
    "multistart_msrs",
    "dycors",
    "cptv",
    "cptvl",
    "socemo",
    "gosac",
    "bayesian_optimization",
    "OptimizeResult",
]
__version__ = "1.2.1"

from . import acquisition
from . import optimize
from . import sampling
from . import rbf
from . import gp

# Expose optimization routines
from .optimize import (
    surrogate_optimization,
    multistart_msrs,
    dycors,
    cptv,
    cptvl,
    socemo,
    gosac,
    bayesian_optimization,
    OptimizeResult,
)
