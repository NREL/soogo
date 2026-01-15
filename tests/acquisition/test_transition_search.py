"""Test the TransitionSearch acquisition function."""

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
import pytest

from soogo.acquisition import CoordinatePerturbation
from tests.acquisition.utils import MockSurrogateModel


class TestTransitionSearch:
    """Test suite for the TransitionSearch acquisition function."""

    @pytest.mark.parametrize(["n_points", "dims"], [([1, 5], [2, 5, 25])])
    def test_optimize_generates_expected_points(self, dims, n_points):
        """
        Test the output points of optimize().

        Ensures that the generated points are:
        - Within the specified bounds.
        - Have the expected shape (n_points, dims).
        - The amount requested.
        """
        for dim in dims:
            for n in n_points:
                bounds = np.array([[0, 1] for _ in range(dim)])
                X_train = np.array([[0.5 for _ in range(dim)]])
                Y_train = np.array([0.0])
                mock_surrogate = MockSurrogateModel(X_train, Y_train)
                cp_search = CoordinatePerturbation()

                result = cp_search.optimize(
                    mock_surrogate, bounds, n=n, weightpattern=0.5
                )
                assert result.shape == (n, dim)
                assert np.all(result >= bounds[:, 0]) and np.all(
                    result <= bounds[:, 1]
                )

    def test_generate_candidates(self):
        """
        Tests that the generate_candidates() method:
        - Generates the expected number of candidates.
        - All candidates are within the specified bounds.
        """
        nCand = [200, 1000, 100000]
        bounds = np.array([[0, 10], [0, 10]])

        for n in nCand:
            cp_search = CoordinatePerturbation(pool_size=n)
            candidates = cp_search.generate_candidates(
                bounds, mu=np.array([5, 5])
            )

            # Should generate n candidates
            assert len(candidates) == n

            # All candidates should be within bounds
            assert np.all(candidates >= bounds[:, 0])
            assert np.all(candidates <= bounds[:, 1])

    # def test_select_candidates(self):
    #     """
    #     Test that the select_candidates() method:

    #     - Chooses the candidate further from evaluated points when function
    #       values are the same.
    #     - Chooses the candidate with lower function value when distances are
    #       the same.
    #     - Removes candidates that are below the evaluability threshold.

    #     """
    #     X_train = np.array([[5, 5]])
    #     Y_train = np.array([0.0])
    #     bounds = np.array([[0, 10], [0, 10]])

    #     mock_surrogate = MockSurrogateModel(X_train, Y_train)
    #     mock_evaluability = MockEvaluabilitySurrogate(X_train, Y_train)
    #     cycle_search = CoordinatePerturbation()

    #     # Both tests would return [0.0, 0.0] if the evaluability filter fails
    #     # Test case 1: Same function values, different distances
    #     candidates = np.array([[0.0, 0.0], [9.0, 1.0], [4.0, 6.0]])
    #     point = cycle_search.select_candidates(
    #         mock_surrogate,
    #         candidates,
    #         bounds,
    #         n=1,
    #         scoreWeight=0.5,
    #         evaluabilitySurrogate=mock_evaluability,
    #     )
    #     assert np.allclose(point, np.array([[9.0, 1.0]]))

    #     # Test case 2: Same distances, different function values
    #     candidates = np.array([[0.0, 0.0], [3.0, 5.0], [7.0, 5.0]])
    #     point = cycle_search.select_candidates(
    #         mock_surrogate,
    #         candidates,
    #         bounds,
    #         n=1,
    #         scoreWeight=0.5,
    #         evaluabilitySurrogate=mock_evaluability,
    #     )
    #     assert np.allclose(point, np.array([[3.0, 5.0]]))

    #     # Test case 3: Weighted sum
    #     X_train = np.array([[5.0, 5.0], [6.0, 6.0], [3.0, 4.0]])
    #     Y_train = np.array([0.0, 1.0, 0.5])
    #     mock_surrogate = MockSurrogateModel(X_train, Y_train)
    #     mock_evaluability = MockEvaluabilitySurrogate(X_train, Y_train)
    #     candidates = np.array([[0.0, 0.0], [2.0, 6.0], [7.0, 0.5]])
    #     point = cycle_search.select_candidates(
    #         mock_surrogate,
    #         candidates,
    #         bounds,
    #         n=1,
    #         scoreWeight=0.75,
    #         evaluabilitySurrogate=mock_evaluability,
    #     )
    #     assert np.allclose(point, np.array([[7.0, 0.5]]))
