"""Test the acquisition functions."""

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
from soogo.acquisition import expected_improvement


def test_expected_improvement():
    # Test case 1: Mu is at the minimum
    mu = 0.0
    sigma = 1.0
    ybest = 0.0
    expected = 0.39894
    result = expected_improvement(ybest - mu, sigma)
    assert np.isclose(
        result, expected, rtol=1e-4
    ), f"Test case 1 failed: {result} != {expected}"

    # Test case 2: Mu is above the minimum
    mu = 1.0
    sigma = 1.0
    ybest = 0.0
    expected = 0.083315
    result = expected_improvement(ybest - mu, sigma)
    assert np.isclose(
        result, expected, rtol=1e-4
    ), f"Test case 2 failed: {result} != {expected}"

    # Test case 3: Mu is below the minimum
    mu = -1.0
    sigma = 1.0
    ybest = 0.0
    expected = 1.0833
    result = expected_improvement(ybest - mu, sigma)
    assert np.isclose(
        result, expected, rtol=1e-4
    ), f"Test case 3 failed: {result} != {expected}"

    # Test case 4: Uncertainty is high
    mu = 0.0
    sigma = 10.0
    ybest = 0.0
    expected = 3.9894
    result = expected_improvement(ybest - mu, sigma)
    assert np.isclose(
        result, expected, rtol=1e-4
    ), f"Test case 4 failed: {result} != {expected}"

    # Test case 5: Uncertainty is low
    mu = 0.0
    sigma = 0.1
    ybest = 0.0
    expected = 0.039894
    result = expected_improvement(ybest - mu, sigma)
    assert np.isclose(
        result, expected, rtol=1e-4
    ), f"Test case 5 failed: {result} != {expected}"
