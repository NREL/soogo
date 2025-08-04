"""Simple example for the usage of the minimize function with no restarts."""

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

from examples.legacy_opt.optimization_program_1 import read_check_data_file
from soogo import RbfModel, MedianLpfFilter, surrogate_optimization
from soogo.sampling import NormalSampler, Sampler, SamplingStrategy
from soogo.acquisition import WeightedAcquisition
from soogo.termination import RobustCondition, UnsuccessfulImprovement

if __name__ == "__main__":
    np.random.seed(3)

    print("This is the test for LocalStochRBFstop")

    data_file = "datainput_hartman3"
    maxeval = 200
    Ntrials = 3
    PlotResult = 1
    batchSize = 2
    data = read_check_data_file(data_file)
    nCand = 500 * data.dim
    m = 2 * (data.dim + 1)
    numstart = (
        0  # collect all objective function values of the current trial here
    )
    Y_all = []  # collect all sample points of the current trial here
    S_all = []  # best objective function value found so far in the current trial
    value = (
        np.inf
    )  # best objective function value found so far in the current trial
    numevals = 0  # number of function evaluations done so far

    bounds = (
        (data.xlow[0], data.xup[0]),
        (data.xlow[1], data.xup[1]),
        (data.xlow[2], data.xup[2]),
    )

    rank_P = 0
    while rank_P != data.dim + 1:
        sample = Sampler(m).get_slhd_sample(bounds)
        P = np.concatenate((np.ones((m, 1)), sample), axis=1)
        rank_P = np.linalg.matrix_rank(P)
    sample = np.array(
        [
            [0.0625, 0.3125, 0.0625],
            [0.1875, 0.5625, 0.8125],
            [0.3125, 0.8125, 0.3125],
            [0.4375, 0.0625, 0.5625],
            [0.5625, 0.9375, 0.4375],
            [0.6875, 0.1875, 0.6875],
            [0.8125, 0.4375, 0.1875],
            [0.9375, 0.6875, 0.9375],
        ]
    )
    sample = np.add(np.multiply(data.xup - data.xlow, sample), data.xlow)

    print(data.xlow)
    print(data.xup)
    print(data.objfunction)
    print(data.dim)
    print(nCand)
    print(sample)
    print("LocalStochRBFstop Start")

    rbfModel = RbfModel(filter=MedianLpfFilter())
    rbfModel.update(sample, data.objfunction(sample))

    optres = surrogate_optimization(
        data.objfunction,
        bounds=bounds,
        maxeval=maxeval - numevals,
        surrogateModel=rbfModel,
        acquisitionFunc=WeightedAcquisition(
            NormalSampler(
                nCand,
                sigma=0.2,
                strategy=SamplingStrategy.NORMAL,
            ),
            weightpattern=[0.3, 0.5],
            rtol=1e-3,
            maxeval=maxeval - numevals,
            sigma_min=0.2 * 0.5**5,
            sigma_max=0.2,
            termination=RobustCondition(UnsuccessfulImprovement(0.001), 5),
        ),
        batchSize=batchSize,
    )

    print("Results")
    print("xlow", data.xlow)
    print("xup", data.xup)
    print("S", optres.sample.shape)
    print("m", optres.sample.shape[0])
    print("Y", optres.fsample.shape)
    print("xbest", optres.x)
    print("Fbest", optres.fx)
    print("lambda", rbfModel.ntrain)
    print("NumberFevals", optres.nfev)
