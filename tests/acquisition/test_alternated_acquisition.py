"""Test the AlternatedAcquisition class."""

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

__authors__ = ["Weslley S. Pereira", "Byron Selvage"]

import numpy as np

from soogo.model import Surrogate
from soogo.acquisition import (
    Acquisition,
    CoordinatePerturbation,
    MaximizeDistance,
    AlternatedAcquisition,
)
from soogo.termination import IterateNTimes
from soogo import OptimizeResult
from tests.acquisition.utils import MockSurrogateModel


class TestAlternatedAcquisition:
    """Test suite for the AlternatedAcquisition class."""

    def test_alternated_acquisition(self):
        """
        Test that the AlternatedAcquisition class correctly alternates
        between acquisition functions.
        """

        # Create mock acquisition functions
        class MockAcquisition(Acquisition):
            def __init__(self, n: int):
                super().__init__(termination=IterateNTimes(n))

            def optimize(
                self, model: Surrogate, bounds: np.ndarray, n: int = 1
            ) -> np.ndarray:
                return np.array([[0.5, 0.5]])

        # Create a list of mock acquisition functions
        acquisition_funcs = [
            MockAcquisition(1),
            MockAcquisition(2),
            MockAcquisition(1),
        ]

        # Initialize the AlternatedAcquisition with the mock functions
        alternated_acq = AlternatedAcquisition(acquisition_funcs)

        # Simulate an optimization result
        out = OptimizeResult()
        out.nfev = 1
        out.fx = np.array([0.1])
        out.fsample = np.array([[0.1]])
        out.nobj = 1

        for i in range(12):
            # Check that it alternates in the pattern: 1st, 2nd, 2nd, 3rd
            expected_pattern = [0, 1, 1, 2]
            assert (
                alternated_acq.idx
                == expected_pattern[i % len(expected_pattern)]
            )

            # Update the alternated acquisition with the mock result
            alternated_acq.update(out, None)

    def test_optimize_generates_expected_points(self, dims=[2, 5, 25]):
        """
        Test that the optimize() method generates point that are the
        correct shape and within bounds while alternating between acquisition
        functions.
        """
        # Use coordinate perturbation search and maximize distance as
        # acquisition functions
        cpAcquisition = CoordinatePerturbation(termination=IterateNTimes(1))
        maximizeDistance = MaximizeDistance(termination=IterateNTimes(2))

        # Create a list of mock acquisition functions
        acquisition_funcs = [cpAcquisition, maximizeDistance]

        # Initialize the AlternatedAcquisition with the mock functions
        alternated_acq = AlternatedAcquisition(acquisition_funcs)

        # Mock optimization result
        out = OptimizeResult()
        out.nfev = 1
        out.fx = np.array([0.1])
        out.fsample = np.array([[0.1]])
        out.nobj = 1

        for dim in dims:
            out.x = np.array([0.1 for _ in range(dim)])
            bounds = np.array([[0, 1] for _ in range(dim)])
            X_train = np.array([[0.5 for _ in range(dim)]])
            Y_train = np.array([0.0])
            mock_surrogate = MockSurrogateModel(X_train, Y_train)

            result = alternated_acq.optimize(
                mock_surrogate, bounds, n=1, weightpattern=0.5
            )
            assert result.shape == (1, dim)
            assert np.all(result >= bounds[:, 0]) and np.all(
                result <= bounds[:, 1]
            )

            alternated_acq.update(out, mock_surrogate)

            assert alternated_acq.idx in [0, 1]
