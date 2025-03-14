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
__version__ = "1.0.1"
__deprecated__ = False

from blackboxoptim.optimize import gosac
from blackboxoptim.rbf import RbfModel

import numpy as np


def objfun(x):
    return (
        2 * x[:, 0] + 3 * x[:, 1] + 1.5 * x[:, 2] + 2 * x[:, 3] - 0.5 * x[:, 4]
    )


def gfun(x):
    C = np.empty((len(x), 5))
    C[:, 0] = x[:, 0] ** 2 + x[:, 2] - 1.25
    C[:, 1] = x[:, 1] ** 1.5 + 1.5 * x[:, 3] - 3
    C[:, 2] = x[:, 0] + x[:, 2] - 1.6
    C[:, 3] = 1.333 * x[:, 1] + x[:, 3] - 3
    C[:, 4] = -x[:, 2] - x[:, 3] + x[:, 4]
    return C


if __name__ == "__main__":
    bounds = ((0, 10), (0, 10), (0, 10), (0, 1), (0, 1))
    np.random.seed(2)

    iindex = (0, 1)
    s = [RbfModel(iindex=iindex) for _ in range(5)]

    maxeval = 100
    res = gosac(objfun, gfun, bounds, maxeval, surrogateModels=s, disp=True)
