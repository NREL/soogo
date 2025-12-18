"""acquisition module"""

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

from .base import Acquisition
from .utils import (
    weighted_score,
    argmin_weighted_score,
    select_weighted_candidates,
)
from .weighted_acquisition import WeightedAcquisition
from .target_value_acquisition import TargetValueAcquisition
from .minimize_surrogate import MinimizeSurrogate
from .pareto_front import ParetoFront
from .endpoints_pareto_front import EndPointsParetoFront
from .minimize_mo_surrogate import MinimizeMOSurrogate
from .coordinate_perturbation_over_nondominated import (
    CoordinatePerturbationOverNondominated,
)
from .gosac_sample import GosacSample
from .maximize_ei import MaximizeEI
from .transition_search import TransitionSearch
from .maximize_distance import MaximizeDistance
from .alternated_acquisition import AlternatedAcquisition
from .multiple_acquisition import MultipleAcquisition

__all__ = [
    "Acquisition",
    "weighted_score",
    "argmin_weighted_score",
    "select_weighted_candidates",
    "WeightedAcquisition",
    "TargetValueAcquisition",
    "MinimizeSurrogate",
    "ParetoFront",
    "EndPointsParetoFront",
    "MinimizeMOSurrogate",
    "CoordinatePerturbationOverNondominated",
    "GosacSample",
    "MaximizeEI",
    "TransitionSearch",
    "MaximizeDistance",
    "AlternatedAcquisition",
    "MultipleAcquisition",
]
