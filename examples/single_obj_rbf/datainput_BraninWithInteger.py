"""Data class for the Branin problem definition."""

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
__deprecated__ = False

import numpy as np
from data import Data


def datainput_BraninWithInteger():
    return Data(
        xlow=np.array([-5, 0]),
        xup=np.array([10, 15]),
        objfunction=myfun,
        dim=2,
        iindex=(0,),
    )


def myfun(x):
    X = np.asarray(x if x.ndim > 1 else [x])
    y = [
        pow(
            xi[1]
            - 5.1 * pow(xi[0], 2) / (4 * pow(np.pi, 2))
            + 5 * xi[0] / np.pi
            - 6,
            2,
        )
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(xi[0])
        + 10
        for xi in X
    ]
    return y


if __name__ == "__main__":
    print(myfun([[3, 2.4013], [3, 2.26325204]]))
    print(myfun([[3, 2.38247443]]))
