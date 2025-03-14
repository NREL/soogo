"""Python interface to the Virtual Library of Simulation Experiments."""

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
    "r",
    "rfuncs",
    "get_function_domain",
    "optRfuncs",
    "get_min_function",
]
__authors__ = ["Weslley S. Pereira"]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = ["Sonja Surjanovic", "Derek Bingham", "Weslley S. Pereira"]
__deprecated__ = False

from rpy2.robjects import r
import os

# The following dictionary contains the name of the R function and the number of
# arguments it takes. If the function takes a variable number of arguments, the
# value is a tuple with the minimum and maximum number of arguments. If the
# function takes a fixed number of arguments, the value is an integer. -1 is used
# to indicate that the function takes a variable number of arguments, but the
# maximum number is unknown.
rfuncs = {
    "limetal02non": 2,
    "hump": 2,
    "grlee08": 2,
    "grlee12": 1,
    # "curretal88explc": 2,
    "zhouetal11": 2,
    "curretal91": 2,
    "ishigami": 3,
    "hart4": 4,
    "wingweight": 10,
    "moon10hd": 20,
    "powell": (4, -1),
    "levy": (1, -1),
    "dejong5": 2,
    "matya": 2,
    "boha3": 2,
    "rosen": (2, -1),
    # "moonetal12": 31,
    "loepetal13": 10,
    "dixonpr": (2, -1),
    "crossit": 2,
    "goldprsc": 2,
    "oakoh022d": 2,
    "camel3": 2,
    "chsan10": 2,
    "easom": 2,
    "moon10hdc2": 20,
    "permdb": 2,
    "trid": (2, -1),
    "forretal08": (1, -1),
    "schaffer2": 2,
    "copeak": (1, -1),
    "gfunc": (1, -1),
    # "forretal08lc": (1,-1),
    "limetal02pol": 2,
    "spherefmod": 6,
    "rastr": (1, -1),
    "hart3": 3,
    "powersum": 4,
    "braninsc": 2,
    "griewank": (1, -1),
    "rosensc": (4, -1),
    "borehole": 8,
    "linketal06sin": 10,
    "willetal06": 3,
    "moon10mix": 3,
    "steelcol": 9,
    "braninmodif": 2,
    "environ": 4,
    "booth": 2,
    "detpep10exp": 3,
    "moon10low": 3,
    "cont": (1, -1),
    "moon10hdc1": 20,
    "sumsqu": (1, -1),
    "schwef": (1, -1),
    "hart6": 6,
    "beale": 2,
    "park91a": 4,
    "morcaf95b": (1, -1),
    "webetal96": 2,
    "morretal06": 30,
    "boha1": 2,
    "langer": 2,
    "franke2d": 2,
    "roosarn63": (1, -1),
    "holsetal13sin": (1, -1),
    "hanetal09": 2,
    "oscil": (1, -1),
    "prpeak": (1, -1),
    "michal": 2,
    "bukin6": 2,
    "ackley": (1, -1),
    "eldetal07ratio": 2,
    "stybtang": (1, -1),
    "linketal06dec": 10,
    "oakoh04": 15,
    "otlcircuit": 6,
    "qianetal08": 2,
    "rothyp": (1, -1),
    "spheref": (1, -1),
    "holder": 2,
    "shubert": 2,
    "park91b": 4,
    "hart6sc": 6,
    "grlee09": 6,
    "morcaf95a": (1, -1),
    # "marthe": 20,
    # "park91alc": 4,
    "zhou98": (1, -1),
    "gaussian": (1, -1),
    "fried": 5,
    "goldpr": 2,
    "egg": 2,
    "sulf": 9,
    "bratleyetal92": (1, -1),
    "colville": 4,
    "detpep108d": 8,
    "robot": 8,
    "soblev99": (1, 20),
    "schaffer4": 2,
    "welchetal92": 20,
    "linketal06simple": 10,
    # "park91blc": 4,
    "boha2": 2,
    "perm0db": 2,
    "shortcol": 3,
    "mccorm": 2,
    "zakharov": (1, -1),
    "santetal03dc": (1, -1),
    "drop": 2,
    "disc": (2, -1),
    "moon10hdc3": 20,
    "detpep10curv": 3,
    "branin": 2,
    "camel6": 2,
    "piston": 7,
    "shekel": 4,
    "linketal06nosig": 10,
    "levy13": 2,
    "sumpow": (1, -1),
}


def get_function_domain(func: str, d: int = 2) -> list:
    """Get the domain of the function func.

    Parameters
    ----------
    func : str
        Name of the function.
    d : int
        Dimension of the input space. Default is 2. Note that some functions
        have a fixed predefined dimension.

    Returns
    -------
    list
        List with the domain boundaries for each dimension.
    """
    from math import pi

    if func == "hump":
        return [[-5.0, 5.0] for i in range(2)]  # Same as camel3
    if func == "ishigami":
        return [[-pi, pi] for i in range(3)]
    if func == "boha1" or func == "boha2" or func == "boha3":
        return [[-100.0, 100.0] for i in range(2)]
    if func == "steelcol":
        return [None] * 9
    if func == "webetal96":
        return [[1.0, 10.0], None]
    if func == "eldetal07ratio":
        return [None] * 2
    if func == "oakoh04":
        return [None] * 15
    if func == "shortcol":
        return [None] * 3
    if func == "disc":
        return [[0.0, 1.0] for i in range(d)]
    if func == "linketal06nosig":
        return [[0.0, 1.0] for i in range(10)]
    if func == "camel3":
        return [[-5.0, 5.0] for i in range(2)]
    if func == "soblev99":
        return [[0.0, 1.0] for i in range(d)]
    if func == "colville":
        return [[-10.0, 10.0] for i in range(4)]
    if func == "perm0db":
        return [[-d * 1.0, d * 1.0] for i in range(d)]
    if func == "environ":
        return [[7.0, 13.0], [0.02, 0.12], [0.01, 3.0], [30.01, 30.295]]
    if func == "levy":
        return [[-10.0, 10.0] for i in range(d)]
    if func == "curretal88sin":
        return [0.0, 1.0]
    if func == "curretal91":
        return [[0.0, 1.0] for i in range(2)]
    if func == "detpep10exp":
        return [[0.0, 1.0] for i in range(3)]
    if func == "prpeak":
        return [[0.0, 1.0] for i in range(d)]
    if func == "piston":
        return [
            [30.0, 60.0],
            [0.005, 0.02],
            [0.002, 0.01],
            [1000.0, 5000.0],
            [90000.0, 110000.0],
            [290.0, 296.0],
            [340.0, 360.0],
        ]
    if func == "linketal06sin":
        return [[0.0, 1.0] for i in range(10)]
    if func == "dejong5":
        return [[-65.536, 65.536] for i in range(2)]
    if func == "branin" or func == "braninsc" or func == "braninmodif":
        return [[-5.0, 10.0], [0.0, 15.0]]
    if func == "drop":
        return [[-5.12, 5.12] for i in range(2)]
    if func == "zhouetal11":
        return [[0.0, 1.0], [1, 3]]
    if func == "stybtang":
        return [[-5.0, 5.0] for i in range(d)]
    if func == "forretal08":
        return [0.0, 1.0]
    if func == "park91b":
        return [[0.0, 1.0] for i in range(4)]
    if func == "limetal02non":
        return [[0.0, 1.0] for i in range(2)]
    if func == "holder":
        return [[-10.0, 10.0] for i in range(2)]
    if func == "moon10mix":
        return [[0.0, 1.0], [0.0, 1.0], [1, 2]]
    if func == "moonetal12":
        return [
            # "Borehole Function": [
            [0.05, 0.15],
            [100.0, 50000.0],
            [63070.0, 115600.0],
            [990.0, 1110.0],
            [63.1, 116.0],
            [700.0, 820.0],
            [1120.0, 1680.0],
            [9855.0, 12045.0],
            # ],
            # "Wing Weight Function": [
            [150.0, 200.0],
            [220.0, 300.0],
            [6.0, 10.0],
            [-10.0, 10.0],
            [16.0, 45.0],
            [0.5, 1.0],
            [0.08, 0.18],
            [2.5, 6.0],
            [1700.0, 2500.0],
            [0.025, 0.08],
            # ],
            # "OTL Circuit Function": [
            [50.0, 150.0],
            [25.0, 70.0],
            [0.5, 3.0],
            [1.2, 2.5],
            [0.25, 1.2],
            [50.0, 300.0],
            # ],
            # "Piston Simulation Function": [
            [30.0, 60.0],
            [0.005, 0.02],
            [0.002, 0.01],
            [1000.0, 5000.0],
            [90000.0, 110000.0],
            [290.0, 296.0],
            [340.0, 360.0],
            # ],
        ]
    if func == "hanetal09":
        return [[0.0, 1.0], [1, 3]]
    if func == "schaffer2":
        return [[-100.0, 100.0] for i in range(2)]
    if func == "sulf":
        return [None] * 9
    if func == "bratleyetal92":
        return [[0.0, 1.0] for i in range(d)]
    if func == "morcaf95b":
        return [[0.0, 1.0] for i in range(d)]
    if func == "dixonpr":
        return [[-10.0, 10.0] for i in range(d)]
    if func == "grlee09":
        return [[0.0, 1.0] for i in range(6)]
    if func == "matya":
        return [[-10.0, 10.0] for i in range(2)]
    if func == "rastr":
        return [[-5.12, 5.12] for i in range(d)]
    if func == "goldpr" or func == "goldprsc":
        return [[-2.0, 2.0] for i in range(2)]
    if func == "shekel":
        return [[0.0, 10.0] for i in range(4)]
    if func == "sumpow":
        return [[-1.0, 1.0] for i in range(d)]
    if func == "zhou98":
        return [[0.0, 1.0] for i in range(d)]
    if func == "sumsqu":
        return [[-10.0, 10.0] for i in range(d)]
    if func == "permdb":
        return [[-d * 1.0, d * 1.0] for i in range(d)]
    if func == "hig02":
        return [0.0, 10.0]
    if func == "borehole":
        return [
            [0.05, 0.15],
            [100.0, 50000.0],
            [63070.0, 115600.0],
            [990.0, 1110.0],
            [63.1, 116.0],
            [700.0, 820.0],
            [1120.0, 1680.0],
            [9855.0, 12045.0],
        ]
    if func == "michal":
        return [[0.0, pi] for i in range(d)]
    if func == "linketal06dec":
        return [[0.0, 1.0] for i in range(10)]
    if func == "wingweight":
        return [
            [150.0, 200.0],
            [220.0, 300.0],
            [6.0, 10.0],
            [-10.0, 10.0],
            [16.0, 45.0],
            [0.5, 1.0],
            [0.08, 0.18],
            [2.5, 6.0],
            [1700.0, 2500.0],
            [0.025, 0.08],
        ]
    if func == "schaffer4":
        return [[-100.0, 100.0] for i in range(2)]
    if func == "gfunc":
        return [[0.0, 1.0] for i in range(d)]
    if func == "franke2d":
        return [[0.0, 1.0] for i in range(2)]
    if func == "powersum":
        return [[0.0, 4.0] for i in range(4)]
    if func == "gaussian":
        return [[0.0, 1.0] for i in range(d)]
    if func == "ackley":
        return [[-32.768, 32.768] for i in range(d)]
    if func == "oscil":
        return [[0.0, 1.0] for i in range(d)]
    if func == "grlee12":
        return [0.5, 2.5]
    if func == "robot":
        return [
            [0.0, 2 * pi],
            [0.0, 2 * pi],
            [0.0, 2 * pi],
            [0.0, 2 * pi],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    if func == "morretal06":
        return [[0.0, 1.0] for i in range(30)]
    if (
        func == "moon10hd"
        or func == "moon10hdc1"
        or func == "moon10hdc2"
        or func == "moon10hdc3"
    ):
        return [[0.0, 1.0] for i in range(20)]
    if func == "marthe":
        return [
            [1.0, 15.0],
            [5.0, 20.0],
            [1.0, 15.0],
            [1.0, 15.0],
            [1.0, 15.0],
            [1.0, 15.0],
            [1.0, 15.0],
            [0.05, 2.0],
            [0.05, 2.0],
            [0.05, 2.0],
            [0.0005, 0.2],
            [0.0005, 0.2],
            [0.0005, 0.2],
            None,
            None,
            None,
            [0.3, 0.37],
            [0.0, 0.0001],
            [0.0, 0.01],
            [0.0, 0.1],
        ]
    if func == "otlcircuit":
        return [
            [50.0, 150.0],
            [25.0, 70.0],
            [0.5, 3.0],
            [1.2, 2.5],
            [0.25, 1.2],
            [50.0, 300.0],
        ]
    if func == "hart6" or func == "hart6sc":
        return [[0.0, 1.0] for i in range(6)]
    if func == "mccorm":
        return [[-1.5, 4.0], [-3.0, 4.0]]
    if func == "moon10low":
        return [[0.0, 1.0] for i in range(3)]
    if func == "grlee08":
        return [[-2.0, 6.0] for i in range(2)]
    if func == "willetal06":
        return [[0.0, 1.0] for i in range(3)]
    if func == "bukin6":
        return [[-15.0, -5.0], [-3.0, 3.0]]
    if func == "oakoh022d":
        return [[-0.01, 0.01] for i in range(2)]
    if func == "langer":
        return [[0.0, 10.0] for i in range(d)]
    if func == "curretal88exp":
        return [[0.0, 1.0] for i in range(2)]
    if func == "roosarn63":
        return [[0.0, 1.0] for i in range(d)]
    if func == "zakharov":
        return [[-5.0, 10.0] for i in range(d)]
    if func == "cont":
        return [[0.0, 1.0] for i in range(d)]
    if func == "chsan10":
        return [[0.0, 1.0] for i in range(2)]
    if func == "welchetal92":
        return [[-0.5, 0.5] for i in range(20)]
    if func == "easom":
        return [[-100.0, 100.0] for i in range(2)]
    if func == "spheref" or func == "spherefmod":
        return [[-5.12, 5.12] for i in range(d)]
    if func == "egg":
        return [[-512.0, 512.0] for i in range(2)]
    if func == "loepetal13":
        return [[0.0, 1.0] for i in range(10)]
    if func == "camel6":
        return [[-3.0, 3.0], [-2.0, 2.0]]
    if func == "qianetal08":
        return [[0.0, 1.0], [1, 2]]
    if func == "holsetal13sin":
        return [0.0, 10.0]
    if func == "linketal06simple":
        return [[0.0, 1.0] for i in range(10)]
    if func == "beale":
        return [[-4.5, 4.5] for i in range(2)]
    if func == "rosen" or func == "rosensc":
        return [[-5.0, 10.0] for i in range(d)]
    if func == "shubert":
        return [[-10.0, 10.0] for i in range(2)]
    if func == "morcaf95a":
        return [[0.0, 1.0] for i in range(d)]
    if func == "booth":
        return [[-10.0, 10.0] for i in range(2)]
    if func == "hart4":
        return [[0.0, 1.0] for i in range(4)]
    if func == "santetal03dc":
        return [0.0, 1.0]
    if func == "schwef":
        return [[-500.0, 500.0] for i in range(d)]
    if func == "copeak":
        return [[0.0, 1.0] for i in range(d)]
    if func == "levy13":
        return [[-10.0, 10.0] for i in range(2)]
    if func == "limetal02pol":
        return [[0.0, 1.0] for i in range(2)]
    if func == "powell":
        return [[-4.0, 5.0] for i in range(d)]
    if func == "trid":
        return [[-(1.0 * d**2), 1.0 * d**2] for i in range(d)]
    if func == "fried":
        return [[0.0, 1.0] for i in range(5)]
    if func == "crossit":
        return [[-10.0, 10.0] for i in range(2)]
    if func == "hig02grlee08":
        return [0.0, 20.0]
    if func == "park91a":
        return [[0.0, 1.0] for i in range(4)]
    if func == "hart3":
        return [[0.0, 1.0] for i in range(3)]
    if func == "rothyp":
        return [[-65.536, 65.536] for i in range(d)]
    if func == "curretal88sur":
        return [0.0, 1.0]
    if func == "detpep108d":
        return [[0.0, 1.0] for i in range(8)]
    if func == "detpep10curv":
        return [[0.0, 1.0] for i in range(3)]
    if func == "griewank":
        return [[-600.0, 600.0] for i in range(d)]
    if func == "holsetal13log":
        return [0.0, 5.0]
    else:
        return [None] * d


# The following tuple contains the names of the R functions that are used in the
# tests. The functions are divided into groups based on the number of arguments
optRfuncs = (
    # Many local minima
    "ackley",
    "bukin6",
    "crossit",
    "drop",
    "egg",
    "grlee12",
    "griewank",
    "holder",
    "langer",
    "levy",
    "levy13",
    "rastr",
    "schaffer2",
    "schaffer4",
    "schwef",
    "shubert",
    # Bowl-Shaped
    "boha1",
    "boha2",
    "boha3",
    "perm0db",
    "rothyp",
    "spheref",
    "spherefmod",
    "sumpow",
    "sumsqu",
    "trid",
    # Plate-Shaped
    "booth",
    "matya",
    "mccorm",
    "powersum",
    "zakharov",
    # Valley-Shaped
    "camel3",
    "camel6",
    "dixonpr",
    "rosen",
    "rosensc",
    # Steep Ridges/Drops
    "dejong5",
    "easom",
    "michal",
    # Other
    "beale",
    "branin",
    "braninsc",
    "braninmodif",
    "colville",
    "forretal08",
    "goldpr",
    "goldprsc",
    "hart3",
    "hart4",
    "hart6",
    "hart6sc",
    "permdb",
    "powell",
    "shekel",
    "stybtang",
)


def get_min_function(func: str, d: int = 2) -> float:
    from numpy import inf

    # Many local minima
    if func == "ackley":
        return 0.0
    if func == "bukin6":
        return 0.0
    if func == "crossit":
        return -2.06261
    if func == "drop":
        return -1.0
    if func == "egg":
        return -959.6407
    if func == "griewank":
        return 0.0
    if func == "holder":
        return -19.2085
    if func == "levy":
        return 0.0
    if func == "levy13":
        return 0.0
    if func == "rastr":
        return 0.0
    if func == "schaffer2":
        return 0.0
    if func == "schwef":
        return 0.0
    if func == "shubert":
        return -186.7309
    # Bowl-Shaped
    if func == "boha1" or func == "boha2" or func == "boha3":
        return 0.0
    if func == "perm0db":
        return 0.0
    if func == "rothyp":
        return 0.0
    if func == "spheref" or func == "spherefmod":
        return 0.0
    if func == "sumpow":
        return 0.0
    if func == "sumsqu":
        return 0.0
    if func == "trid":
        return (-d * (d + 4) * (d - 1)) / 6
    # Plate-Shaped
    if func == "booth":
        return 0.0
    if func == "matya":
        return 0.0
    if func == "mccorm":
        return -1.9133
    if func == "zakharov":
        return 0.0
    # Valley-Shaped
    if func == "camel3":
        return 0.0
    if func == "camel6":
        return -1.0316
    if func == "dixonpr":
        return 0
    if func == "rosen" or func == "rosensc":
        return 0.0
    # Steep Ridges/Drops
    if func == "easom":
        return -1.0
    # Other
    if func == "beale":
        return 0.0
    if func == "branin" or func == "braninsc" or func == "braninmodif":
        return 0.397887
    if func == "colville":
        return 0.0
    if func == "goldpr" or func == "goldprsc":
        return 3.0
    if func == "hart3":
        return -3.86278
    if func == "hart6":
        return -3.04245876289
    if func == "hart6sc":
        return -3.32237
    if func == "permdb":
        return 0.0
    if func == "powell":
        return 0.0
    if func == "stybtang":
        return -39.16599 * d

    # Obtained using scipy differential_evolution with
    # TODO: Run multiple times to check this is accurate
    # maxiter=10000 and tol=1e-15
    if func == "grlee12":
        return -0.8690111349894997
    if func == "schaffer4":
        return 0.29257863203598045
    if func == "powersum":
        return 0.0  # Solution at (1,2,2,4)
    if func == "dejong5":
        return 0.99800383779445
    if func == "forretal08":
        return -6.020740055767083
    if func == "hart4":
        return -3.1344941412223988
    if func == "shekel":
        return -10.53644315348353

    # Obtained from the paper:
    # Certified Global Minima for a Benchmark of Difficult Optimization Problems
    # by Charlie Vanaret et al. 2020
    # See https://arxiv.org/pdf/2003.09867
    if func == "michal":
        if d == 10:
            return -9.6601517
        if d == 15:
            return -14.6464002
        if d == 20:
            return -19.6370136
        if d == 25:
            return -24.6331947
        if d == 30:
            return -29.6308839
        if d == 35:
            return -34.6288550
        if d == 40:
            return -39.6267489
        if d == 45:
            return -44.6256251
        if d == 50:
            return -49.6248323
        if d == 55:
            return -54.6240533
        if d == 60:
            return -59.6231462
        if d == 65:
            return -64.6226167
        if d == 70:
            return -69.6222202
        if d == 75:
            return -74.6218112

    return inf


# Load the R functions
benchPath = os.path.dirname(os.path.realpath(__file__))
for rfile, _ in rfuncs.items():
    r.source(os.path.join(benchPath, rfile + ".r"))

# Clean up the namespace
del benchPath, rfile
del os

if __name__ == "__main__":
    print("VLSE benchmark loaded.")
    print("Available functions:")

    for func in rfuncs:
        try:
            print(
                func,
                "(" + str(rfuncs[func]) + " arguments) in domain",
                get_function_domain(
                    func,
                    rfuncs[func]
                    if isinstance(rfuncs[func], int)
                    else rfuncs[func][0],
                ),
            )
        except Exception as _:
            print(func, "(" + str(rfuncs[func]) + " arguments)")
    print("")

    from rpy2 import robjects
    import numpy as np

    hart6 = getattr(r, "hart6")
    x = np.array((0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573))
    y = np.array(
        (
            0.40464078,
            0.88244549,
            0.84609725,
            0.57397974,
            0.13891472,
            0.03850067,
        )
    )
    print(np.array(hart6(robjects.FloatVector(x.reshape(-1, 1))))[0])
    print(np.array(hart6(robjects.FloatVector(y.reshape(-1, 1))))[0])
