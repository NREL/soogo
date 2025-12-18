"""GOSAC benchmark problems and test functions."""

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

__authors__ = ["Weslley S. Pereira"]

from copy import copy
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


def fRana(x: np.ndarray) -> np.ndarray:
    # Source: http://infinity77.net/global_optimization/test_functions_nd_R.html#go_benchmark.Rana
    x1 = x[:, 0]
    return np.sum(
        x.T
        * np.sin(np.sqrt(np.abs(x1 - x.T + 1)))
        * np.cos(np.sqrt(np.abs(x1 + x.T + 1)))
        + (x1 + 1)
        * np.sin(np.sqrt(np.abs(x1 + x.T + 1)))
        * np.cos(np.sqrt(np.abs(x1 - x.T + 1))),
        axis=0,
    )


def fWeierstrass(x: np.ndarray) -> np.ndarray:
    # Source: http://infinity77.net/global_optimization/test_functions_nd_W.html#go_benchmark.Weierstrass
    n = x.shape[1]
    kmax = 20
    a = 0.5
    b = 3
    return np.sum(
        sum(
            [
                (a**k) * np.cos(2 * np.pi * (b**k) * (x + 0.5))
                for k in range(kmax + 1)
            ]
        )
        - n * sum([(a**k) * np.cos(np.pi * (b**k)) for k in range(10 + 1)]),
        axis=1,
    )


@dataclass
class Problem:
    """A class to represent a problem for the GOSAC benchmark.

    Attributes
    ----------
    objf : Callable[[np.ndarray], np.ndarray]
        The objective function. Receives a 2D array of shape (n, dim) and
        returns a 1D array of shape (n,).
    gfun : Callable[[np.ndarray], np.ndarray]
        The constraint function. Receives a 2D array of shape (n, dim) and
        returns a 2D array of shape (n, gdim).
    iindex : tuple[int, ...]
        The indices of the integer variables.
    bounds : tuple[tuple[float, float], ...]
        The bounds of the variables.
    xmin : tuple[float, ...] | None
        The known minimum of the objective function. If None, the minimum is
        unknown.
    fmin : float | None
        The value of the objective function at the known minimum. If None, the
        minimum is unknown.
    """

    objf: Callable[[np.ndarray], np.ndarray]
    gfun: Callable[[np.ndarray], np.ndarray]
    iindex: tuple[int, ...]
    bounds: tuple[tuple[float, float], ...]
    xmin: Optional[tuple[float, ...]] = None
    fmin: Optional[float] = None


# Problems from the GOSAC benchmark
gosac_p: list[Problem] = []

# Problem 1
gosac_p.append(
    Problem(
        lambda x: 5 * (x[:, 1] - 0.2) ** 2 + 0.8 - 0.7 * x[:, 0],
        lambda x: np.transpose(
            [
                -np.exp(x[:, 1] - 0.2) - x[:, 2],
                x[:, 2] + 1.1 * x[:, 0] + 1,
                x[:, 1] - 1.2 * x[:, 0],
            ]
        ),
        (0,),
        ((0.0, 1.0), (0.2, 1.0), (-2.22554, -1.0)),
        (1, 0.9419, -2.1),
        2.8524,
    )
)

# Problem 2
gosac_p.append(
    Problem(
        lambda x: 2 * x[:, 0]
        + 3 * x[:, 1]
        + 1.5 * x[:, 2]
        + 2 * x[:, 3]
        - 0.5 * x[:, 4],
        lambda x: np.transpose(
            [
                x[:, 0] ** 2 + x[:, 2] - 1.25,
                x[:, 1] ** 1.5 + 1.5 * x[:, 3] - 3,
                x[:, 0] + x[:, 2] - 1.6,
                1.333 * x[:, 1] + x[:, 3] - 3,
                -x[:, 2] - x[:, 3] + x[:, 4],
            ]
        ),
        (0, 1),
        ((0, 10), (0, 10), (0, 10), (0, 1), (0, 1)),
        (0, 0, 0, 0, 0),
        0,
    )
)

# Problem 3
gosac_p.append(
    Problem(
        lambda x: 2 * x[:, 0]
        + 3 * x[:, 1]
        + 1.5 * x[:, 2]
        + 2 * x[:, 3]
        - 0.5 * x[:, 4],
        lambda x: np.transpose(
            [
                x[:, 0] + x[:, 2] - 1.6,
                1.333 * x[:, 1] + x[:, 3] - 3,
                -x[:, 2] - x[:, 3] + x[:, 4],
            ]
        ),
        (0, 1, 2),
        ((0, 10), (0, 10), (0, 10), (0, 1), (0, 1)),
        (0, 0, 0, 0, 0),
        0,
    )
)

# Problem 4
gosac_p.append(
    Problem(
        lambda x: (x[:, 4] - 1) ** 2
        + (x[:, 5] - 2) ** 2
        + (x[:, 6] - 1) ** 2
        - np.log(x[:, 7] + 1)
        + (x[:, 8] - 1) ** 2
        + (x[:, 9] - 2) ** 2
        + (x[:, 10] - 3) ** 2,
        lambda x: np.transpose(
            [
                x[:, 0] + x[:, 1] + x[:, 2] + x[:, 8] + x[:, 9] + x[:, 10] - 5,
                x[:, 6] ** 2
                + x[:, 8] ** 2
                + x[:, 9] ** 2
                + x[:, 10] ** 2
                - 5.5,
                x[:, 0] + x[:, 8] - 1.2,
                x[:, 1] + x[:, 9] - 1.8,
                x[:, 2] + x[:, 10] - 2.5,
                x[:, 3] + x[:, 8] - 1.2,
                x[:, 5] ** 2 + x[:, 9] ** 2 - 1.64,
                x[:, 6] ** 2 + x[:, 10] ** 2 - 4.25,
                x[:, 5] ** 2 + x[:, 10] ** 2 - 4.64,
                x[:, 4] - x[:, 0],
                x[:, 5] - x[:, 1],
                x[:, 6] - x[:, 2],
                x[:, 7] - x[:, 3],
            ]
        ),
        (0, 1, 2, 3),
        (
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 10),
            (0, 10),
            (0, 10),
        ),
        (1, 1, 0, 1, 1, 1, 0, 1, 0.2, 0.8, 1.9079),
        4.5796,
    )
)

# Problem 5
gosac_p.append(
    Problem(
        lambda x: -np.abs(
            (
                np.sum(np.cos(x) ** 4, axis=1)
                - 2 * np.prod(np.cos(x) ** 2, axis=1)
            )
            / np.sqrt(np.sum(np.arange(1, x.shape[1] + 1) * x**2, axis=1))
        ),
        lambda x: np.transpose(
            [
                0.75 - np.prod(x, axis=1),
                np.sum(x, axis=1) - 7.5 * x.shape[1],
            ]
        ),
        (0, 1, 2, 3, 4, 5),
        ((0, 10),) * 25,
        (
            3,
            3,
            3,
            3,
            3,
            3,
            3.021616897,
            2.998077169,
            0.3770080798,
            0.3780122315,
            0.3737440086,
            2.900804325,
            2.871928856,
            2.843098301,
            2.806423432,
            0.3625403532,
            0.3622042429,
            0.3593120409,
            0.3589796805,
            0.3556172311,
            0.3542916697,
            0.3534646982,
            0.3511441992,
            0.3492637643,
            0.3467394526,
        ),
        -0.73904,
    )
)

# Problem 6
gosac_p.append(
    Problem(
        # 5.3578547x23 + 0.8356891x1 x5 + 37.293239x1 − 40792.141
        lambda x: 5.3578547 * x[:, 2] ** 2
        + 0.8356891 * x[:, 0] * x[:, 4]
        + 37.293239 * x[:, 0]
        - 40792.141,
        lambda x: np.transpose(
            [
                -85.334407
                - 0.0056858 * x[:, 1] * x[:, 4]
                - 0.0006262 * x[:, 0] * x[:, 3]
                + 0.0022053 * x[:, 2] * x[:, 4],
                85.334407
                + 0.0056858 * x[:, 1] * x[:, 4]
                + 0.0006262 * x[:, 0] * x[:, 3]
                - 0.0022053 * x[:, 2] * x[:, 4]
                - 92,
                90
                - 80.51249
                - 0.0071317 * x[:, 1] * x[:, 4]
                - 0.0029955 * x[:, 0] * x[:, 1]
                - 0.0021813 * x[:, 2] ** 2,
                80.51249
                + 0.0071317 * x[:, 1] * x[:, 4]
                + 0.0029955 * x[:, 0] * x[:, 1]
                + 0.0021813 * x[:, 2] ** 2
                - 110,
                20
                - 9.300961
                - 0.0047026 * x[:, 2] * x[:, 4]
                - 0.0012547 * x[:, 0] * x[:, 2]
                - 0.0019085 * x[:, 2] * x[:, 3],
                9.300961
                + 0.0047026 * x[:, 2] * x[:, 4]
                + 0.0012547 * x[:, 0] * x[:, 2]
                + 0.0019085 * x[:, 2] * x[:, 3]
                - 25,
            ]
        ),
        (0, 1),
        ((78, 102), (33, 45), (27, 45), (27, 45), (27, 45)),
        (78, 33, 29.9953, 45, 36.7758),
        -30665.54,
    )
)

# Problem 7
gosac_p.append(
    Problem(
        lambda x: (x[:, 0] - 10) ** 2
        + 5 * (x[:, 1] - 12) ** 2
        + x[:, 2] ** 4
        + 3 * (x[:, 3] - 11) ** 2
        + 10 * x[:, 4] ** 6
        + 7 * x[:, 5] ** 2
        + x[:, 6] ** 4
        - 4 * x[:, 5] * x[:, 6]
        - 10 * x[:, 5]
        - 8 * x[:, 6],
        lambda x: np.transpose(
            [
                2 * x[:, 0] ** 2
                + 3 * x[:, 1] ** 4
                + x[:, 2]
                + 4 * x[:, 3] ** 2
                + 5 * x[:, 4]
                - 127,
                7 * x[:, 0]
                + 3 * x[:, 1]
                + 10 * x[:, 2] ** 2
                + x[:, 3]
                - x[:, 4]
                - 282,
                23 * x[:, 0]
                + x[:, 1] ** 2
                + 6 * x[:, 5] ** 2
                - 8 * x[:, 6]
                - 196,
                4 * x[:, 0] ** 2
                + x[:, 1] ** 2
                - 3 * x[:, 0] * x[:, 1]
                + 2 * x[:, 2] ** 2
                + 5 * x[:, 5]
                - 11 * x[:, 6],
            ]
        ),
        (0, 1, 2),
        ((-10, 10),) * 7,
        (
            2,
            2,
            -1,
            4.32299345363447,
            -0.550622364659113,
            1.16531693494537,
            1.46201272171894,
        ),
        682.936407277722,
    )
)

# Problem 8
gosac_p.append(
    Problem(
        lambda x: np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1),
        lambda x: np.reshape(
            2 - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1), (-1, 1)
        ),
        (0, 1, 2),
        ((-12, 12),) * 10,
        (
            -11,
            -11,
            -11,
            -11.0949,
            -11.0943,
            -11.0951,
            -11.1007,
            -11.1086,
            -11.0945,
            -11.0874,
        ),
        121.3460,
    )
)

# Problem 9
gosac_p.append(
    Problem(
        lambda x: 1
        - (1 / 30) * np.sum(np.cos(10 * x) * np.exp(-0.5 * x**2), axis=1),
        lambda x: np.reshape(
            30 - np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1), (-1, 1)
        ),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
        ((-3, 3),) * 15 + ((-np.pi, np.pi),) * 15,
        (
            -2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -2,
            2,
            2,
            0,
            -2,
            -1.2966,
            -1.9140,
            0.6706,
            1.8452,
            1.8257,
            1.2247,
            2.1688,
            1.1877,
            -1.1559,
            1.8825,
            2.0568,
            0.1094,
            -1.9155,
            1.2508,
            1.8362,
        ),
        0.5242,
    )
)

# Problem 10
gosac_p.append(
    Problem(
        # Bird test function
        lambda x: (x[:, 0] - x[:, 1]) ** 2
        + np.exp((1 - np.sin(x[:, 0])) ** 2) * np.cos(x[:, 1])
        + np.exp((1 - np.cos(x[:, 1])) ** 2) * np.sin(x[:, 0]),
        # Rana test function
        lambda x: np.reshape(fRana(x) - 5, (-1, 1)),
        (0,),
        ((-9, 9), (-3 * np.pi, 3 * np.pi)),
        # (-8, -9.4142),
        # -104.3309,
    )
)

# Problem 11
gosac_p.append(copy(gosac_p[1]))
gosac_p[-1].iindex = ()

# Problem 12
gosac_p.append(copy(gosac_p[0]))
gosac_p[-1].iindex = ()
gosac_p[-1].xmin = (0.5752, 0.6903, -1.6327)
gosac_p[-1].fmin = 1.5991

# Problem 13
gosac_p.append(copy(gosac_p[4]))
gosac_p[-1].iindex = ()
gosac_p[-1].xmin = (
    3.15511326,
    3.130995963,
    3.107959557,
    3.085973799,
    3.062816464,
    3.040233783,
    3.020947118,
    2.994815207,
    0.3872600413,
    0.365723082,
    0.3691262415,
    2.891160109,
    2.862145914,
    2.829570976,
    2.789894285,
    0.3515813893,
    0.3492115401,
    0.3572330093,
    0.3515194382,
    0.3478836621,
    0.3494086305,
    0.3570991925,
    0.3472568487,
    0.3465191214,
    0.341666913,
)
gosac_p[-1].fmin = -0.7432

# Problem 14
gosac_p.append(copy(gosac_p[5]))
gosac_p[-1].iindex = ()

# Problem 15
gosac_p.append(copy(gosac_p[6]))
gosac_p[-1].iindex = ()
gosac_p[-1].xmin = (
    2.33052957014494,
    1.95136812551608,
    -0.477596518954723,
    4.36572821435469,
    -0.624470180441579,
    1.03811416325727,
    1.59426489591394,
)
gosac_p[-1].fmin = 680.630056404647

# Problem 16
gosac_p.append(copy(gosac_p[3]))
gosac_p[-1].iindex = ()
# gosac_p[-1].xmin = (
#     0.4849,
#     0.8024,
#     0.4851,
#     0.0690,
#     0.4849,
#     0.8024,
#     0.4851,
#     0.9690,
#     0.2310,
#     0.9976,
#     1.9990,
# )
# gosac_p[-1].fmin = 3.8853

# Problem 17
gosac_p.append(copy(gosac_p[2]))
gosac_p[-1].iindex = ()

# Problem 18
gosac_p.append(copy(gosac_p[9]))
gosac_p[-1].iindex = ()
gosac_p[-1].bounds = ((-3 * np.pi, 3 * np.pi),) * 2
gosac_p[-1].xmin = (-1.5821, -3.1302)
gosac_p[-1].fmin = -106.7645

# Problem 19
gosac_p.append(
    Problem(
        lambda x: 10
        * x[:, 4]
        * x[:, 6]
        * x[:, 8]
        * x[:, 9]
        * x[:, 13]
        * x[:, 14]
        * x[:, 15]
        + 7 * x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
        + x[:, 2] * x[:, 3] * x[:, 5] * x[:, 6] * x[:, 7]
        + 12 * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
        + 8 * x[:, 5] * x[:, 6] * x[:, 7] * x[:, 11]
        + 3 * x[:, 5] * x[:, 6] * x[:, 8] * x[:, 13] * x[:, 15]
        + x[:, 8] * x[:, 9] * x[:, 13] * x[:, 15]
        + 5 * x[:, 4] * x[:, 9] * x[:, 13] * x[:, 14] * x[:, 15]
        + 3 * x[:, 0] * x[:, 1] * x[:, 10] * x[:, 11],
        lambda x: np.transpose(
            [
                3
                * x[:, 4]
                * x[:, 6]
                * x[:, 8]
                * x[:, 9]
                * x[:, 13]
                * x[:, 14]
                * x[:, 15]
                - 12
                * x[:, 0]
                * x[:, 1]
                * x[:, 2]
                * x[:, 3]
                * x[:, 7]
                * x[:, 10]
                - 8 * x[:, 2] * x[:, 3] * x[:, 5] * x[:, 6] * x[:, 7]
                + x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - 7 * x[:, 0] * x[:, 1] * x[:, 10] * x[:, 11]
                + 2 * x[:, 12] * x[:, 13] * x[:, 14] * x[:, 15]
                + 2,
                x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - 10 * x[:, 2] * x[:, 3] * x[:, 5] * x[:, 6] * x[:, 7]
                - 5 * x[:, 5] * x[:, 6] * x[:, 7] * x[:, 11]
                + x[:, 5] * x[:, 6] * x[:, 8] * x[:, 13] * x[:, 15]
                + 7 * x[:, 8] * x[:, 9] * x[:, 13] * x[:, 15]
                + x[:, 4] * x[:, 9] * x[:, 13] * x[:, 14] * x[:, 15]
                + 1,
                5
                * x[:, 4]
                * x[:, 6]
                * x[:, 8]
                * x[:, 9]
                * x[:, 13]
                * x[:, 14]
                * x[:, 15]
                - 3
                * x[:, 0]
                * x[:, 1]
                * x[:, 2]
                * x[:, 3]
                * x[:, 7]
                * x[:, 10]
                - x[:, 2] * x[:, 3] * x[:, 5] * x[:, 6] * x[:, 7]
                - 2 * x[:, 4] * x[:, 9] * x[:, 13] * x[:, 14] * x[:, 15]
                + x[:, 12] * x[:, 13] * x[:, 14] * x[:, 15]
                + 1,
                3 * x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - 5
                * x[:, 4]
                * x[:, 6]
                * x[:, 8]
                * x[:, 9]
                * x[:, 13]
                * x[:, 14]
                * x[:, 15]
                + x[:, 2] * x[:, 3] * x[:, 5] * x[:, 6] * x[:, 7]
                + 2 * x[:, 4] * x[:, 9] * x[:, 13] * x[:, 14] * x[:, 15]
                - x[:, 12] * x[:, 13] * x[:, 14] * x[:, 15]
                - 1,
                -4 * x[:, 2] * x[:, 3] * x[:, 5] * x[:, 6] * x[:, 7]
                - 2 * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - 5 * x[:, 5] * x[:, 6] * x[:, 8] * x[:, 13] * x[:, 15]
                + x[:, 8] * x[:, 9] * x[:, 13] * x[:, 15]
                - 9 * x[:, 4] * x[:, 9] * x[:, 13] * x[:, 14] * x[:, 15]
                - 2 * x[:, 0] * x[:, 1] * x[:, 10] * x[:, 11]
                + 3,
                9 * x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - 12 * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - 7 * x[:, 5] * x[:, 6] * x[:, 7] * x[:, 11]
                + 6 * x[:, 5] * x[:, 6] * x[:, 8] * x[:, 13] * x[:, 15]
                + 2 * x[:, 4] * x[:, 9] * x[:, 13] * x[:, 14] * x[:, 15]
                - 15 * x[:, 0] * x[:, 1] * x[:, 10] * x[:, 11]
                + 3 * x[:, 12] * x[:, 13] * x[:, 14] * x[:, 15]
                + 7,
                5 * x[:, 0] * x[:, 1] * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - 8
                * x[:, 4]
                * x[:, 6]
                * x[:, 8]
                * x[:, 9]
                * x[:, 13]
                * x[:, 14]
                * x[:, 15]
                + 2 * x[:, 2] * x[:, 3] * x[:, 5] * x[:, 6] * x[:, 7]
                - 7 * x[:, 2] * x[:, 3] * x[:, 7] * x[:, 10]
                - x[:, 5] * x[:, 6] * x[:, 7] * x[:, 11]
                - 5 * x[:, 8] * x[:, 9] * x[:, 13] * x[:, 15]
                - 10 * x[:, 0] * x[:, 1] * x[:, 10] * x[:, 11]
                + 1,
            ]
        ),
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
        ((0, 1),) * 16,
        (1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1),
        13,
    )
)

# Problem 20
gosac_p.append(copy(gosac_p[1]))
gosac_p[-1].iindex = tuple(range(5))

# Problem 21
gosac_p.append(copy(gosac_p[4]))
gosac_p[-1].iindex = tuple(range(25))
gosac_p[-1].xmin = (
    8,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
    3,
)
gosac_p[-1].fmin = -0.4218

# Problem 22
gosac_p.append(copy(gosac_p[5]))
gosac_p[-1].iindex = tuple(range(5))
gosac_p[-1].xmin = (81, 33, 30, 45, 36)
gosac_p[-1].fmin = -30512.45

# Problem 23
gosac_p.append(copy(gosac_p[6]))
gosac_p[-1].iindex = tuple(range(7))
gosac_p[-1].xmin = (2, 2, 0, 4, 0, 1, 2)
gosac_p[-1].fmin = 700

# Problem 24
gosac_p.append(copy(gosac_p[3]))
gosac_p[-1].iindex = tuple(range(11))
gosac_p[-1].xmin = (1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 2)
gosac_p[-1].fmin = 7.3069

# Problem 25
gosac_p.append(copy(gosac_p[2]))
gosac_p[-1].iindex = tuple(range(5))

# Problem 26
gosac_p.append(
    Problem(
        # Weierstrass test function
        fWeierstrass,
        # Vicent test function
        lambda x: np.reshape(-np.sum(np.sin(10 * np.log(x)), axis=1), (-1, 1)),
        (),
        ((0.25, np.pi),) * 10,
        # (3, 2, 3, 2, 3, 2, 3, 3, 1, 2),
        # 1.1783,
    )
)

# Problem 27
gosac_p.append(copy(gosac_p[7]))
gosac_p[-1].iindex = tuple(range(20))
gosac_p[-1].bounds = ((-12, 12),) * 20
gosac_p[-1].xmin = (
    -11,
    11,
    11,
    11,
    -11,
    -12,
    -11,
    -11,
    -11,
    -11,
    -11,
    -11,
    11,
    -11,
    -11,
    11,
    -11,
    11,
    -12,
    -11,
)
gosac_p[-1].fmin = 219.8758
