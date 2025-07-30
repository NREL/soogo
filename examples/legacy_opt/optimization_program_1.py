"""Example with optimization and plot."""

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

__authors__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__deprecated__ = False


from copy import deepcopy
import importlib
from math import sqrt
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

import soogo
from soogo import sampling, OptimizeResult
from soogo.acquisition import (
    WeightedAcquisition,
    TargetValueAcquisition,
    AcquisitionFunction,
    MinimizeSurrogate,
)
from soogo.model.rbf_kernel import RadialBasisFunction
from soogo.model import (
    RbfModel,
    MedianLpfFilter,
    CubicRadialBasisFunction,
    ThinPlateRadialBasisFunction,
)

from data import Data


def read_and_run(
    data_file: str,
    acquisitionFunc: Optional[AcquisitionFunction] = None,
    maxeval: int = 0,
    Ntrials: int = 0,
    batchSize: int = 0,
    rbf_type: RadialBasisFunction = CubicRadialBasisFunction(),
    PlotResult: bool = True,
    optim_func=soogo.multistart_msrs,
    seeds=None,
) -> list[OptimizeResult]:
    """Perform the optimization and plot the solution if asked.

    Parameters
    ----------
    data_file : str
        Path for the data file.
    acquisitionFunc : soogo.acquisition.AcquisitionFunction, optional
        Sampler to be used. Default: None, i.e., defined by the optimizer.
    maxeval : int, optional
        Maximum number of allowed function evaluations per trial.
        Default: 0, i.e., defined by the :func:`check_set_parameters()`.
    Ntrials : int, optional
        Number of trials.
        Default: 0, i.e., defined by the :func:`check_set_parameters()`.
    batchSize : int, optional
        Number of new sample points per step of the optimization algorithm.
        Default: 0, i.e., defined by the :func:`check_set_parameters()`.
    rbf_type : RadialBasisFunction, optional
        Type of RBF kernel to be used. Default: CubicRadialBasisFunction.
    PlotResult : bool, optional
        If True, plot the results. Default: False.
    optim_func : optional
        Optimizer to be used. Default :func:`soogo.multistart_msrs()`.
    seeds: optional
        Sequence of random seeds to be used. One per trial.
        Default: None, defined inside this function.

    Returns
    -------
    optres : list[OptimizeResult]
        List of OptimizeResult objects with the optimization results.
    """
    # Define seeds
    if seeds is None:
        np.random.seed(3)
        seeds = [np.random.randint(9999) for _ in range(Ntrials)]

    ## Start input check
    data = read_check_data_file(data_file)
    maxeval, Ntrials, batchSize = check_set_parameters(
        data, maxeval, Ntrials, batchSize
    )
    ## End input check

    ## Optimization
    optres = []
    for j in range(Ntrials):
        np.random.seed(seeds[j])

        # Create empty RBF model
        rbfModel = RbfModel(rbf_type, data.iindex, filter=MedianLpfFilter())
        acquisitionFuncIter = deepcopy(acquisitionFunc)

        # Call the surrogate optimization function
        if (
            optim_func == soogo.surrogate_optimization
            or optim_func == soogo.dycors
        ):
            opt = optim_func(
                data.objfunction,
                bounds=tuple(
                    (data.xlow[i], data.xup[i]) for i in range(data.dim)
                ),
                maxeval=maxeval,
                surrogateModel=rbfModel,
                acquisitionFunc=acquisitionFuncIter,
                batchSize=batchSize,
                disp=True,
            )
        elif (
            optim_func == soogo.surrogate_optimization
            or optim_func == soogo.multistart_msrs
            or optim_func == soogo.cptv
            or optim_func == soogo.cptvl
        ):
            opt = optim_func(
                data.objfunction,
                bounds=tuple(
                    (data.xlow[i], data.xup[i]) for i in range(data.dim)
                ),
                maxeval=maxeval,
                surrogateModel=rbfModel,
                disp=True,
            )
        else:
            raise ValueError("Invalid optimization function.")
        optres.append(opt)
    ## End Optimization

    ## Plot Result
    if PlotResult:
        plot_results(optres, "RBFPlot.png")
    ## End Plot Result

    return optres


def plot_results(optres: list[OptimizeResult], filename: str):
    """Plot the results.

    Parameters
    ----------
    optres: list[OptimizeResult]
        List of OptimizeResult objects with the optimization results.
    filename : str
        Path for the plot file.
    """
    Ntrials = len(optres)
    maxeval = min([len(optres[i].fsample) for i in range(Ntrials)])
    Y_cur_best = np.empty((maxeval, Ntrials))
    for ii in range(Ntrials):  # go through all trials
        Y_cur = optres[ii].fsample
        Y_cur_best[0, ii] = Y_cur[0]
        for j in range(1, maxeval):
            if Y_cur[j] < Y_cur_best[j - 1, ii]:
                Y_cur_best[j, ii] = Y_cur[j]
            else:
                Y_cur_best[j, ii] = Y_cur_best[j - 1, ii]
    # compute means over matrix of current best values (Y_cur_best has dimension
    # maxeval x Ntrials)
    Ymean = np.mean(Y_cur_best, axis=1)

    plt.rcParams.update({"font.size": 16})

    plt.plot(np.arange(1, maxeval + 1), Ymean)
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Average best function value")
    plt.yscale("log")
    plt.grid(True, linestyle="dashed")
    plt.draw()
    # show()
    plt.savefig(filename)


def read_check_data_file(data_file: str) -> Data:
    """Read and check the data file for correctness.

    Parameters
    ----------
    data_file : str
        Path for the data file.

    Returns
    -------
    data : Data
        Valid data object with the problem definition.
    """
    module = importlib.import_module(data_file)
    data = getattr(module, data_file)()

    if data.is_valid() is False:
        raise ValueError(
            """The data file is not valid. Please, look at the documentation of\
            \n\tthe class Data for more information."""
        )

    return data


def check_set_parameters(
    data: Data,
    maxeval: int = 0,
    Ntrials: int = 0,
    batchSize: int = 0,
):
    """Check and set the parameters for the optimization if not provided.

    Parameters
    ----------
    data : Data
        Data object with the problem definition.
    maxeval : int, optional
        Maximum number of allowed function evaluations per trial.
    Ntrials : int, optional
        Number of trials.
    batchSize : int, optional
        Number of new sample points per step of the optimization algorithm.

    Returns
    -------
    maxeval : int
        Maximum number of allowed function evaluations per trial.
    Ntrials : int
        Number of trials.
    batchSize : int
        Number of new sample points per step of the optimization algorithm.
    """

    if maxeval == 0:
        print(
            """No maximal number of allowed function evaluations given.\
                \n\tI use default value maxeval = 20 * dimension."""
        )
        maxeval = 20 * data.dim
    if not isinstance(maxeval, int) or maxeval <= 0:
        raise ValueError(
            "Maximal number of allowed function evaluations must be positive integer.\n"
        )

    if Ntrials == 0:
        print(
            """No maximal number of trials given.\
                \n\tI use default value NumberOfTrials=1."""
        )
        Ntrials = 1
    if not isinstance(Ntrials, int) or Ntrials <= 0:
        raise ValueError(
            "Maximal number of trials must be positive integer.\n"
        )

    if batchSize == 0:
        print(
            """No number of desired new sample sites given.\
                \n\tI use default value batchSize=1."""
        )
        batchSize = 1
    if not isinstance(batchSize, int) or batchSize < 0:
        raise ValueError(
            "Number of new sample sites must be positive integer.\n"
        )

    return maxeval, Ntrials, batchSize


def main(config: int) -> list[OptimizeResult]:
    """
    Main workflow

    :param int config: Defines problem to solve and optimizer to use.
    :return: List of optimization results.
    """
    if config == 1:
        optres = read_and_run(
            data_file="datainput_Branin",
            acquisitionFunc=WeightedAcquisition(
                sampling.NormalSampler(
                    1000,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**5,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.NORMAL,
                ),
                [
                    0.95,
                ],
                maxeval=200,
            ),
            maxeval=200,
            Ntrials=3,
            batchSize=1,
            PlotResult=True,
        )
    elif config == 2:
        optres = read_and_run(
            data_file="datainput_hartman3",
            acquisitionFunc=WeightedAcquisition(
                sampling.NormalSampler(
                    300,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**6,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS,
                ),
                [0.3, 0.5, 0.8, 0.95],
                maxeval=200,
            ),
            maxeval=200,
            Ntrials=1,
            batchSize=1,
            PlotResult=True,
        )
    elif config == 3:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=None,
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=ThinPlateRadialBasisFunction(),
            PlotResult=True,
            optim_func=soogo.dycors,
        )
    elif config == 4:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=WeightedAcquisition(
                sampling.NormalSampler(
                    200,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**5,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS_UNIFORM,
                ),
                [0.3, 0.5, 0.8, 0.95],
                maxeval=100,
            ),
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=ThinPlateRadialBasisFunction(),
            PlotResult=True,
            optim_func=soogo.surrogate_optimization,
        )
    elif config == 5:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=TargetValueAcquisition(),
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=ThinPlateRadialBasisFunction(),
            PlotResult=True,
            optim_func=soogo.surrogate_optimization,
        )
    elif config == 6:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=ThinPlateRadialBasisFunction(),
            PlotResult=True,
            optim_func=soogo.cptv,
        )
    elif config == 7:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=ThinPlateRadialBasisFunction(),
            PlotResult=True,
            optim_func=soogo.cptvl,
        )
    elif config == 8:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=MinimizeSurrogate(100, 0.005 * sqrt(2)),
            maxeval=100,
            Ntrials=3,
            batchSize=10,
            rbf_type=ThinPlateRadialBasisFunction(),
            PlotResult=True,
            optim_func=soogo.surrogate_optimization,
        )
    else:
        raise ValueError("Invalid configuration number.")

    return optres


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the optimization and plot the results."
    )
    parser.add_argument(
        "--config",
        type=int,
        help="Configuration number to be used.",
        default=1,
    )
    args = parser.parse_args()

    optres = main(args.config)
    Ntrials = len(optres)

    print("BestValues", [optres[i].fx for i in range(Ntrials)])
    print("BestPoints", [optres[i].x for i in range(Ntrials)])
    print("NumFuncEval", [optres[i].nfev for i in range(Ntrials)])
    print("NumberOfIterations", [optres[i].nit for i in range(Ntrials)])
