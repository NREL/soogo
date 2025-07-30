"""Run the optimization on the VLSE benchmark."""

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
__credits__ = ["Weslley S. Pereira"]
__deprecated__ = False

import os
import numpy as np
import pickle
import time
from benchmark import *

from soogo import optimize, acquisition, sampling, OptimizeResult
from pathlib import Path
from copy import deepcopy

from soogo.model import gp, rbf


def run_optimizer(
    objf,
    maxEval: int,
    algo,
    nRuns: int,
    *,
    bounds=None,
    disp: bool = False,
) -> list[OptimizeResult]:
    """Runs the algorithm `algo` to find the minimum of the function `objf`.

    In the ith run, the random seed is set to i. This assures every method uses
    the same set of `2 * (n + 1)` initial sample points, where `n` is the
    dimension of the input space.

    If the `bounds` for a variable are two integers, that variable is considered
    an integer automatically.

    :param objf: Objective function.
    :param maxEval: Maximum number of function evaluations allowed.
    :param algo: Optimization algorithm.
    :param nRuns: Number of times that optimization will be performed.
    :param bounds: Bounds for the optimization. Default: the bounds for the
        domain of `objf`.
    :param disp: Whether or not to display information about the optimization
        iterations.
    :return: List of results for each of the `nRuns` optimization runs.
    """
    bounds = objf.domain() if bounds is None else bounds
    nArgs = len(bounds)

    assert len(bounds) == len(objf.domain())

    # integrality constraints
    model = deepcopy(algo["model"])
    if isinstance(model, rbf.RbfModel):
        model.iindex = tuple(
            i
            for i in range(nArgs)
            if isinstance(bounds[i][0], int) and isinstance(bounds[i][1], int)
        )

    # Update acquisition strategy, using maxEval and nArgs for the problem
    acquisitionFunc = deepcopy(algo["acquisition"])
    if hasattr(acquisitionFunc, "maxeval"):
        acquisitionFunc.maxeval = maxEval
    if hasattr(acquisitionFunc, "sampler"):
        acquisitionFunc.sampler.n = min(100 * nArgs, 5000)

    # Find the minimum
    optimizer = algo["optimizer"]
    optres = []
    for i in range(nRuns):
        # Create initial sample
        np.random.seed(i)
        sample0 = sampling.Sampler(2 * (nArgs + 1)).get_slhd_sample(bounds)
        fsample0 = objf(sample0)

        # Create initial surrogate
        modelIter = deepcopy(model)
        modelIter.update(sample0, fsample0)

        if (
            optimizer == optimize.surrogate_optimization
            or optimizer == optimize.dycors
        ):
            acquisitionFuncIter = deepcopy(acquisitionFunc)
            res = optimizer(
                objf,
                bounds=bounds,
                maxeval=maxEval - 2 * (nArgs + 1),
                surrogateModel=modelIter,
                acquisitionFunc=acquisitionFuncIter,
                disp=disp,
            )
        else:
            res = optimizer(
                objf,
                bounds=bounds,
                maxeval=maxEval - 2 * (nArgs + 1),
                surrogateModel=modelIter,
                disp=disp,
            )
        optres.append(res)

        print(res.x)
        print(res.fx)

    return optres


# Functions that can be used in the tests
myFuncs = {
    "branin": Branin(),
    "hart3": Hart3(),
    "hart6": Hart6(),
    "shekel": Shekel(),
    "ackley": Ackley(15),
    "levy": Levy(20),
    "powell": Powell(24),
    "michal": Michal(20),
    "spheref": Spheref(27),
    "rastr": Rastr(30),
    "mccorm": Mccorm(),
    "bukin6": Bukin6(),
    "camel6": Camel6(),
    "crossit": Crossit(),
    "drop": Drop(),
    "egg": Egg(),
    "griewank": Griewank(2),
    "holder": Holder(),
    "levy13": Levy13(),
}

# Algorithms that can be used in the tests
algorithms = {}
algorithms["SRS"] = {
    "model": rbf.RbfModel(filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.multistart_msrs,
    "acquisition": None,
}
algorithms["DYCORS"] = {
    "model": rbf.RbfModel(filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.dycors,
    "acquisition": None,
}
algorithms["CPTV"] = {
    "model": rbf.RbfModel(filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.cptv,
    "acquisition": None,
}
algorithms["CPTVl"] = {
    "model": rbf.RbfModel(filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.cptvl,
    "acquisition": None,
}
algorithms["MLSL"] = {
    "model": rbf.RbfModel(filter=rbf.MedianLpfFilter()),
    "optimizer": optimize.surrogate_optimization,
    "acquisition": acquisition.MinimizeSurrogate(1, 0.005 * np.sqrt(2.0)),
}
algorithms["GP"] = {
    "model": gp.GaussianProcess(normalize_y=True),
    "optimizer": optimize.bayesian_optimization,
    "acquisition": acquisition.MaximizeEI(),
}

# Maximum number of evaluations per function. 100*n, where n is the input dimension
maxEvals = {key: 100 * (len(f.domain()) + 1) for key, f in myFuncs.items()}

# Program that runs the benchmark
if __name__ == "__main__":
    import argparse

    # Arguments for command line
    parser = argparse.ArgumentParser(
        description="Run given algorithm and problem from the vlse benchmark"
    )
    parser.add_argument(
        "-a", "--algorithm", choices=algorithms.keys(), default="CPTVl"
    )
    parser.add_argument(
        "-p", "--problem", choices=myFuncs.keys(), default="branin"
    )
    parser.add_argument("-n", "--ntrials", type=int, default=3)
    parser.add_argument(
        "-b",
        "--bounds",
        metavar="[low,high]",
        type=float,
        nargs="+",
        help="Pass in order: low0, high0, low1, high1, ...",
    )
    args = parser.parse_args()

    # Process bounds
    if args.bounds is not None:
        bounds = [
            [args.bounds[2 * i], args.bounds[2 * i + 1]]
            for i in range(len(args.bounds) // 2)
        ]
    else:
        bounds = None

    # Print params
    print(args.algorithm)
    print(args.problem)
    print(bounds)
    print(args.ntrials)

    # Run optimization and record time
    t0 = time.time()
    optres = run_optimizer(
        myFuncs[args.problem],
        maxEvals[args.problem],
        algorithms[args.algorithm],
        args.ntrials,
        bounds=bounds,
        disp=True,
    )
    tf = time.time()

    # Save the results
    folder = os.path.dirname(os.path.abspath(__file__)) + "/pickle"
    Path(folder).mkdir(parents=True, exist_ok=True)
    filepath = (
        folder
        + "/"
        + args.problem
        + "_"
        + args.algorithm
        + "_"
        + ("bounds" if bounds else "default")
        + ".pkl"
    )
    with open(filepath, "wb") as f:
        pickle.dump(
            [
                len(myFuncs[args.problem].domain()),
                maxEvals[args.problem],
                args.ntrials,
                optres,
                (tf - t0),
                bounds,
            ],
            f,
        )
