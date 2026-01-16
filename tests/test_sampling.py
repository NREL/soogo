"""Test the sampling functions."""

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

import numpy as np
import pytest
from scipy.stats.qmc import LatinHypercube

from soogo.sampling import (
    random_sample,
    SymmetricLatinHypercube,
    SpaceFillingSampler,
    truncnorm_sample,
    dds_sample,
    dds_uniform_sample,
)


@pytest.mark.parametrize("dim", [1, 2, 3, 10])
@pytest.mark.parametrize("rng_tag", ["", "default", "lhd", "slhd"])
def test_sampler(dim: int, rng_tag: str):
    n = 2 * (dim + 1)
    bounds = [(-1, 1)] * dim

    if rng_tag == "default":
        rng = np.random.default_rng()
    elif rng_tag == "lhd":
        rng = LatinHypercube(d=dim)
    elif rng_tag == "slhd":
        rng = SymmetricLatinHypercube(d=dim)
    else:
        rng = None

    for i in range(3):
        sample = random_sample(n, bounds, seed=rng)

        # Check if the shape is correct
        assert sample.shape == (n, dim)

        # Check if the values are within the bounds
        for j in range(dim):
            assert np.all(sample[:, j] >= -1)
            assert np.all(sample[:, j] <= 1)

        # Check that the values do not repeat in the slhd case
        if rng_tag == "lhd" or rng_tag == "slhd":
            for j in range(dim):
                u, c = np.unique(sample[:, j], return_counts=True)
                assert u[c > 1].size == 0


@pytest.mark.parametrize("dim", [1, 2, 3, 10])
@pytest.mark.parametrize(
    "normal_sampler", [truncnorm_sample, dds_sample, dds_uniform_sample]
)
def test_normal_sampler(dim: int, normal_sampler):
    n = 2 * (dim + 1)
    bounds = [(-1, 1)] * dim
    sigma = 0.1
    probability = 0.5
    mu = np.array([b[0] + (b[1] - b[0]) / 2 for b in bounds])

    # Set seed to 5 for reproducibility
    rng = np.random.default_rng(5)

    for i in range(3):
        if normal_sampler == truncnorm_sample:
            sample = normal_sampler(
                n,
                bounds,
                mu=mu,
                sigma_ref=sigma,
                seed=rng,
            )
        else:
            sample = normal_sampler(
                n,
                bounds,
                mu=mu,
                sigma_ref=sigma,
                probability=probability,
                seed=rng,
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

    for i in range(3):
        sample = SpaceFillingSampler().generate(
            n, bounds, current_sample=sample0
        )

        # Check if the shape is correct
        assert sample.shape == (n, dim)

        # Check if the values are within the bounds
        for j in range(dim):
            assert np.all(sample[:, j] >= -1)
            assert np.all(sample[:, j] <= 1)


@pytest.mark.parametrize("boundx", [(0, 1), (-1, 1), (-6, 5)])
@pytest.mark.parametrize(
    "normal_sampler", [truncnorm_sample, dds_sample, dds_uniform_sample]
)
def test_iindex_sampler(boundx, normal_sampler):
    dim = 10
    n = 2 * (dim + 1)
    bounds = [boundx] * dim
    sigma = 0.1
    probability = 0.5

    # Set seed to 5 for reproducibility
    rng = np.random.default_rng(5)

    for i in range(3):
        iindex = np.random.choice(dim, size=dim // 2)
        mu = np.array([b[0] + (b[1] - b[0]) / 2 for b in bounds])
        mu[iindex] = np.round(mu[iindex])

        if normal_sampler == truncnorm_sample:
            sample = normal_sampler(
                n,
                bounds,
                mu=mu,
                sigma_ref=sigma,
                seed=rng,
                iindex=iindex,
            )
        else:
            sample = normal_sampler(
                n,
                bounds,
                mu=mu,
                sigma_ref=sigma,
                probability=probability,
                seed=rng,
                iindex=iindex,
            )

        # Check if the sample has integer values in the iindex
        for i in iindex:
            assert np.all(sample[:, i] - np.round(sample[:, i]) == 0)


@pytest.mark.parametrize("boundx", [(0, 1), (-1, 1), (0, 10)])
@pytest.mark.parametrize("rng_tag", ["lhd", "slhd"])
def test_slhd(boundx, rng_tag: str):
    dim = 10
    bounds = [boundx] * dim

    if rng_tag == "lhd":
        rng = LatinHypercube(d=dim, scramble=False)
    else:
        rng = SymmetricLatinHypercube(d=dim, scramble=False)

    for i in range(3):
        iindex = np.random.choice(dim, size=dim // 2).tolist()

        for n in (boundx[1] - boundx[0], boundx[1] - boundx[0] + 1):
            sample = random_sample(n, bounds, seed=rng, iindex=iindex)

            # Check if the sample has integer values in the iindex
            for i in iindex:
                assert np.all(sample[:, i] - np.round(sample[:, i]) == 0)

            # Check if the sample has repeated values
            for i in range(dim):
                u, c = np.unique(sample[:, i], return_counts=True)
                assert u[c > 1].size == 0, f"{n}: {sample[:, i]}"


@pytest.mark.parametrize("boundx", [(0, 1), (-1, 1), (-6, 5)])
@pytest.mark.parametrize("n0", [0, 1, 10])
def test_iindex_mitchel91_sampler(boundx, n0: int):
    dim = 10
    n = 2 * (dim + 1)
    bounds = [boundx] * dim
    sample0 = np.random.rand(n0, dim)

    for i in range(3):
        iindex = np.random.choice(dim, size=dim // 2).tolist()

        sample = SpaceFillingSampler().generate(
            n, bounds, iindex=iindex, current_sample=sample0
        )

        # Check if the sample has integer values in the iindex
        for i in iindex:
            assert np.all(sample[:, i] - np.round(sample[:, i]) == 0)
