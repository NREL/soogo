"""Data class for the problem definition."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC
# Copyright (C) 2013 Cornell University

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

__authors__ = ["Juliane Mueller", "Christine A. Shoemaker", "Haoyu Jia"]
__contact__ = "juliane.mueller@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__version__ = "1.0.1"
__deprecated__ = False

from dataclasses import dataclass
from collections.abc import Callable
import numpy as np


@dataclass
class Data:
    """Data class for the problem definition.

    Attributes
    ----------
    xlow : np.ndarray
        Lower bounds of the input space.
    xup : np.ndarray
        Upper bounds of the input space.
    objfunction : callable
        Objective function.
    dim : int
        Dimension of the input space.
    iindex : tuple = ()
        Indices of the input space that are integer.
    """

    xlow: np.ndarray
    xup: np.ndarray
    objfunction: Callable[[np.ndarray], float]
    dim: int
    iindex: tuple[int, ...] = ()

    def is_valid(self) -> bool:
        # Dimension must be positive integer
        if self.dim <= 0:
            return False
        # Vector length of lower and upper bounds must equal problem dimension
        if self.xlow.size != self.dim or self.xup.size != self.dim:
            return False
        # Lower bounds must be lower than upper bounds
        if np.any(self.xlow > self.xup):
            return False
        # Integrality tuple must be valid
        if not all(
            self.iindex[i] in range(self.dim) for i in range(len(self.iindex))
        ):
            return False
        return True
