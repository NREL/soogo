"""Test the Gaussian Process model and helpers."""

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
from soogo.model.gp import GaussianProcess, gp_expected_improvement


@pytest.mark.parametrize("n", (10, 100))
@pytest.mark.parametrize("copy_X_train", (True, False))
def test_X(n: int, copy_X_train: bool):
    gp = GaussianProcess(copy_X_train=copy_X_train)

    X0 = np.random.rand(n, 3)
    y = np.random.rand(n)
    gp.update(X0, y)
    assert np.isclose(X0, gp.X).all()

    X1 = np.random.rand(n, 3)
    y = np.random.rand(n)
    gp.update(X1, y)
    assert np.isclose(np.concatenate((X0, X1), axis=0), gp.X).all()


def test_expected_improvement():
    # Test case 1: Mu is at the minimum
    mu = 0.0
    sigma = 1.0
    ybest = 0.0
    expected = 0.39894
    result = gp_expected_improvement(ybest - mu, sigma)
    assert np.isclose(result, expected, rtol=1e-4), (
        f"Test case 1 failed: {result} != {expected}"
    )

    # Test case 2: Mu is above the minimum
    mu = 1.0
    sigma = 1.0
    ybest = 0.0
    expected = 0.083315
    result = gp_expected_improvement(ybest - mu, sigma)
    assert np.isclose(result, expected, rtol=1e-4), (
        f"Test case 2 failed: {result} != {expected}"
    )

    # Test case 3: Mu is below the minimum
    mu = -1.0
    sigma = 1.0
    ybest = 0.0
    expected = 1.0833
    result = gp_expected_improvement(ybest - mu, sigma)
    assert np.isclose(result, expected, rtol=1e-4), (
        f"Test case 3 failed: {result} != {expected}"
    )

    # Test case 4: Uncertainty is high
    mu = 0.0
    sigma = 10.0
    ybest = 0.0
    expected = 3.9894
    result = gp_expected_improvement(ybest - mu, sigma)
    assert np.isclose(result, expected, rtol=1e-4), (
        f"Test case 4 failed: {result} != {expected}"
    )

    # Test case 5: Uncertainty is low
    mu = 0.0
    sigma = 0.1
    ybest = 0.0
    expected = 0.039894
    result = gp_expected_improvement(ybest - mu, sigma)
    assert np.isclose(result, expected, rtol=1e-4), (
        f"Test case 5 failed: {result} != {expected}"
    )
