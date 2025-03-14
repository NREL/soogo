"""Compare different learning strategies for RBF surrogate optimization."""

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


from copy import deepcopy
import importlib
from math import sqrt
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from blackboxoptim import rbf, optimize, sampling
from blackboxoptim.acquisition import (
    WeightedAcquisition,
    AcquisitionFunction,
    MinimizeSurrogate,
)
from data import Data


def read_and_run(
    data_file: str,
    acquisitionFunc: Optional[AcquisitionFunction] = None,
    maxeval: int = 0,
    Ntrials: int = 0,
    batchSize: int = 0,
    rbf_type: rbf.RbfKernel = rbf.RbfKernel.CUBIC,
    filter: rbf.RbfFilter = rbf.MedianLpfFilter(),
    optim_func=optimize.multistart_msrs,
    seeds=None,
) -> list[optimize.OptimizeResult]:
    """Perform the optimization, save the solution and plot.

    If PlotResult is True, it also plots the results and saves the plot to
    "RBFPlot.png".

    Parameters
    ----------
    data_file : str
        Path for the data file.
    sampler : sampling.NormalSampler, optional
        Sampler to be used.
    weightpattern : list[float], optional
        List of weights for the weighted RBF.
    maxeval : int, optional
        Maximum number of allowed function evaluations per trial.
    Ntrials : int, optional
        Number of trials.
    batchSize : int, optional
        Number of new sample points per step of the optimization algorithm.
    rbf_type : rbf.RbfKernel, optional
        Type of RBF to be used.

    Returns
    -------
    optres : list[optimize.OptimizeResult]
        List of optimize.OptimizeResult objects with the optimization results.
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
        rbfModel = rbf.RbfModel(rbf_type, data.iindex, filter=filter)
        acquisitionFuncIter = deepcopy(acquisitionFunc)

        # # Uncomment to compare with Surrogates.jl
        # rbfModel.update_xtrain(
        #     np.array(
        #         [
        #             [0.3125, 0.8125, 0.8125],
        #             [0.6875, 0.0625, 0.4375],
        #             [0.4375, 0.5625, 0.6875],
        #             [0.9375, 0.6875, 0.3125],
        #             [0.5625, 0.3125, 0.5625],
        #             [0.0625, 0.9375, 0.1875],
        #             [0.8125, 0.1875, 0.9375],
        #             [0.1875, 0.4375, 0.0625],
        #         ]
        #     )
        # )

        # Call the surrogate optimization function
        if (
            optim_func == optimize.surrogate_optimization
            or optim_func == optimize.dycors
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
            optim_func == optimize.surrogate_optimization
            or optim_func == optimize.multistart_msrs
            or optim_func == optimize.cptv
            or optim_func == optimize.cptvl
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

    return optres


def get_meanPerNEval(optres: list[optimize.OptimizeResult]) -> np.ndarray:
    """Get mean of the best function values per number of function evaluations.

    Parameters
    ----------
    optres: list[optimize.OptimizeResult]
        List of optimize.OptimizeResult objects with the optimization results.
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
    return np.mean(Y_cur_best, axis=1)


def read_check_data_file(data_file: str) -> Data:
    """Read and check the data file.

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
    """Check and set the parameters for the optimization.

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


if __name__ == "__main__":
    comparisonList = [3, 5, 6, 7, 8]
    optresList = {}
    strategyName = []

    if 1 in comparisonList:
        optresList[1] = read_and_run(
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
            filter=rbf.MedianLpfFilter(),
            maxeval=200,
            Ntrials=3,
            batchSize=1,
        )
    if 2 in comparisonList:
        optresList[2] = read_and_run(
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
            filter=rbf.MedianLpfFilter(),
            maxeval=200,
            Ntrials=1,
            batchSize=1,
        )
    if 3 in comparisonList:
        strategyName.append("DYCORS")
        optresList[3] = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=None,
            filter=rbf.MedianLpfFilter(),
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=rbf.RbfKernel.THINPLATE,
            optim_func=optimize.dycors,
        )
    if 4 in comparisonList:
        optresList[4] = read_and_run(
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
            filter=rbf.MedianLpfFilter(),
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=rbf.RbfKernel.THINPLATE,
            optim_func=optimize.surrogate_optimization,
        )
    if 5 in comparisonList:
        strategyName.append("TargetValue")
        optresList[5] = read_and_run(
            data_file="datainput_BraninWithInteger",
            filter=rbf.MedianLpfFilter(),
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=rbf.RbfKernel.THINPLATE,
            optim_func=optimize.surrogate_optimization,
        )
    if 6 in comparisonList:
        strategyName.append("CPTV")
        optresList[6] = read_and_run(
            data_file="datainput_BraninWithInteger",
            filter=rbf.MedianLpfFilter(),
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=rbf.RbfKernel.THINPLATE,
            optim_func=optimize.cptv,
        )
    if 7 in comparisonList:
        strategyName.append("CPTVl")
        optresList[7] = read_and_run(
            data_file="datainput_BraninWithInteger",
            filter=rbf.MedianLpfFilter(),
            maxeval=100,
            Ntrials=3,
            batchSize=1,
            rbf_type=rbf.RbfKernel.THINPLATE,
            optim_func=optimize.cptvl,
        )
    if 8 in comparisonList:
        strategyName.append("MinimizeSurrogate")
        optresList[8] = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=MinimizeSurrogate(100, 0.005 * sqrt(2)),
            filter=rbf.MedianLpfFilter(),
            maxeval=100,
            Ntrials=3,
            batchSize=10,
            rbf_type=rbf.RbfKernel.THINPLATE,
            optim_func=optimize.surrogate_optimization,
        )

    ## Plot Results
    plt.rcParams.update({"font.size": 16})
    j = 0
    for i in comparisonList:
        Ymean = get_meanPerNEval(optresList[i])
        plt.plot(np.arange(1, Ymean.size + 1), Ymean, label=strategyName[j])
        j += 1
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Average best function value")
    plt.legend()
    plt.yscale("log")
    plt.draw()
    plt.savefig("RBFComparePlot.png")
