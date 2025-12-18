"""Data class for the Hartman3 problem definition."""

# Copyright (c) 2025 Alliance for Energy Innovation, LLC
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
import numpy.matlib as matlib
from data import Data


def datainput_hartman3():
    return Data(xlow=np.zeros(3), xup=np.ones(3), objfunction=myfun, dim=3)


def myfun(x):
    X = np.asarray(x if x.ndim > 1 else [x])
    c = np.array([1, 1.2, 3, 3.2])
    A = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]])
    P = np.array(
        [
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.0382, 0.5743, 0.8828],
        ]
    )
    y = [
        -sum(
            c * np.exp(-np.sum(A * (matlib.repmat(xi, 4, 1) - P) ** 2, axis=1))
        )
        for xi in X
    ]
    return y


if __name__ == "__main__":
    print(myfun(np.array([0.5, 0.9, 0.3])))
