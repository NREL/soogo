"""Test the sampling functions."""

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

import numpy as np
import pytest
from soogo.sampling import (
    Mitchel91Sampler,
    NormalSampler,
    Sampler,
    SamplingStrategy,
)


@pytest.mark.parametrize("dim", [1, 2, 3, 10])
@pytest.mark.parametrize(
    "strategy", [SamplingStrategy.UNIFORM, SamplingStrategy.SLHD]
)
def test_sampler(dim: int, strategy: SamplingStrategy):
    n = 2 * (dim + 1)
    bounds = [(-1, 1)] * dim

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        sample = Sampler(n, strategy=strategy).get_sample(bounds)

        # Check if the shape is correct
        assert sample.shape == (n, dim)

        # Check if the values are within the bounds
        for j in range(dim):
            assert np.all(sample[:, j] >= -1)
            assert np.all(sample[:, j] <= 1)

        # Check that the values do not repeat in the slhd case
        if strategy == SamplingStrategy.SLHD:
            for j in range(dim):
                u, c = np.unique(sample[:, j], return_counts=True)
                assert u[c > 1].size == 0


@pytest.mark.parametrize("dim", [1, 2, 3, 10])
@pytest.mark.parametrize(
    "strategy",
    [
        SamplingStrategy.UNIFORM,
        SamplingStrategy.SLHD,
        SamplingStrategy.NORMAL,
        SamplingStrategy.DDS,
        SamplingStrategy.DDS_UNIFORM,
    ],
)
def test_normal_sampler(dim: int, strategy: SamplingStrategy):
    n = 2 * (dim + 1)
    bounds = [(-1, 1)] * dim
    sigma = 0.1
    probability = 0.5
    mu = np.array([b[0] + (b[1] - b[0]) / 2 for b in bounds])

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        sample = NormalSampler(n, sigma, strategy=strategy).get_sample(
            bounds, mu=mu, probability=probability
        )

        # Check if the shape is correct
        assert sample.shape == (n, dim)

        # Check if the values are within the bounds
        for j in range(dim):
            assert np.all(sample[:, j] >= -1)
            assert np.all(sample[:, j] <= 1)


@pytest.mark.parametrize("dim", [1, 2, 3, 10])
@pytest.mark.parametrize("n0", [0, 1, 10])
def test_mitchel91_sampler(dim: int, n0: int):
    n = 2 * (dim + 1)
    bounds = [(-1, 1)] * dim
    sample0 = np.random.rand(n0, dim)

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        sample = Mitchel91Sampler(n).get_sample(bounds, current_sample=sample0)

        # Check if the shape is correct
        assert sample.shape == (n, dim)

        # Check if the values are within the bounds
        for j in range(dim):
            assert np.all(sample[:, j] >= -1)
            assert np.all(sample[:, j] <= 1)


@pytest.mark.parametrize("boundx", [(0, 1), (-1, 1), (-6, 5)])
@pytest.mark.parametrize(
    "strategy",
    [
        SamplingStrategy.UNIFORM,
        SamplingStrategy.SLHD,
        SamplingStrategy.DDS,
        SamplingStrategy.DDS_UNIFORM,
    ],
)
def test_iindex_sampler(boundx, strategy: SamplingStrategy):
    dim = 10
    n = 2 * (dim + 1)
    bounds = [boundx] * dim
    sigma = 0.1
    probability = 0.5

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        iindex = np.random.choice(dim, size=dim // 2)
        mu = np.array([b[0] + (b[1] - b[0]) / 2 for b in bounds])
        mu[iindex] = np.round(mu[iindex])

        sample = NormalSampler(n, sigma, strategy=strategy).get_sample(
            bounds, iindex=iindex, mu=mu, probability=probability
        )

        # Check if the sample has integer values in the iindex
        for i in iindex:
            assert np.all(sample[:, i] - np.round(sample[:, i]) == 0)


@pytest.mark.parametrize("boundx", [(0, 1), (-1, 1), (0, 10)])
def test_slhd(boundx):
    dim = 10
    bounds = [boundx] * dim

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        iindex = np.random.choice(dim, size=dim // 2)

        for n in (boundx[1] - boundx[0], boundx[1] - boundx[0] + 1):
            sample = Sampler(
                n, strategy=SamplingStrategy.SLHD
            ).get_slhd_sample(bounds, iindex=iindex)

            # Check if the sample has integer values in the iindex
            for i in iindex:
                assert np.all(sample[:, i] - np.round(sample[:, i]) == 0)

            # Check if the sample has repeated values
            for i in range(dim):
                u, c = np.unique(sample[:, i], return_counts=True)
                assert u[c > 1].size == 0


@pytest.mark.parametrize("boundx", [(0, 1), (-1, 1), (-6, 5)])
@pytest.mark.parametrize("n0", [0, 1, 10])
def test_iindex_mitchel91_sampler(boundx, n0: int):
    dim = 10
    n = 2 * (dim + 1)
    bounds = [boundx] * dim
    sample0 = np.random.rand(n0, dim)

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        iindex = np.random.choice(dim, size=dim // 2)

        sample = Mitchel91Sampler(n).get_sample(
            bounds, iindex=iindex, current_sample=sample0
        )

        # Check if the sample has integer values in the iindex
        for i in iindex:
            assert np.all(sample[:, i] - np.round(sample[:, i]) == 0)
