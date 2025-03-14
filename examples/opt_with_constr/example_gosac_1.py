"""Example of using GOSAC to optimize a function with constraints."""

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

__authors__ = ["Weslley S. Pereira"]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = ["Juliane Mueller", "Weslley S. Pereira"]
__deprecated__ = False

from blackboxoptim import gosac
from blackboxoptim.rbf import RbfModel

import numpy as np


def objfun(x):
    """Test problem G4 by Koziel and Michalewicz [1999], integrality constraints
    added.
    """
    return (
        5.3578547 * x[:, 2] ** 2
        + 0.8356891 * x[:, 0] * x[:, 4]
        + 37.293239 * x[:, 0]
        - 40792.141
    )


def gfun(x):
    C = np.empty((len(x), 6))
    C[:, 0] = (
        85.334407
        + 0.0056858 * x[:, 1] * x[:, 4]
        + 0.0006262 * x[:, 0] * x[:, 3]
        - 0.0022053 * x[:, 2] * x[:, 4]
        - 92
    )
    C[:, 1] = (
        -85.334407
        - 0.0056858 * x[:, 1] * x[:, 4]
        - 0.0006262 * x[:, 0] * x[:, 3]
        + 0.0022053 * x[:, 2] * x[:, 4]
    )
    C[:, 2] = (
        80.51249
        + 0.0071317 * x[:, 1] * x[:, 4]
        + 0.0029955 * x[:, 0] * x[:, 1]
        + 0.0021813 * x[:, 2] ** 2
        - 110
    )
    C[:, 3] = (
        -80.51249
        - 0.0071317 * x[:, 1] * x[:, 4]
        - 0.0029955 * x[:, 0] * x[:, 1]
        - 0.0021813 * x[:, 2] ** 2
        + 90
    )
    C[:, 4] = (
        9.300961
        + 0.0047026 * x[:, 2] * x[:, 4]
        + 0.0012547 * x[:, 0] * x[:, 2]
        + 0.0019085 * x[:, 2] * x[:, 3]
        - 25
    )
    C[:, 5] = (
        -9.300961
        - 0.0047026 * x[:, 2] * x[:, 4]
        - 0.0012547 * x[:, 0] * x[:, 2]
        - 0.0019085 * x[:, 2] * x[:, 3]
        + 20
    )
    return C


if __name__ == "__main__":
    bounds = ((78, 102), (33, 45), (27, 45), (27, 45), (27, 45))
    np.random.seed(3)

    iindex = (0, 1)
    s = [RbfModel(iindex=iindex) for _ in range(6)]

    maxeval = 100
    res = gosac(objfun, gfun, bounds, maxeval, surrogateModels=s, disp=True)
