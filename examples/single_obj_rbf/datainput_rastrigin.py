"""Data class for the Rastigrin problem definition."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC
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

import numpy as np
from data import Data


def datainput_rastrigin():
    n = 100
    return Data(
        xlow=-np.ones(n) * 10, xup=np.ones(n) * 10, objfunction=myfun, dim=n
    )


def myfun(x):
    n = 100
    y = 10 * n + np.sum(
        np.power(x, 2) - 10 * np.cos(2 * np.pi * np.asarray(x)), axis=1
    )

    return y


if __name__ == "__main__":
    print(myfun([[0.5, 0.9, 0.3]]))
