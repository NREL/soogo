"""Test the MaximizeDistance acquisition function."""

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

from soogo.acquisition import MaximizeDistance
from tests.acquisition.utils import MockSurrogateModel


class TestMaximizeDistance:
    """Test suite for the MaximizeDistance acquisition function."""

    def test_optimize_generates_expected_points(self, dims=[2, 5, 25]):
        """
        Test the output points of optimize().

        Ensures that the generated points are:
        - Within the specified bounds.
        - The expected shape (n_points, dims).
        - The amount requested.
        """
        for dim in dims:
            bounds = np.array([[0, 1] for _ in range(dim)])
            X_train = np.array([[0.5 for _ in range(dim)]])
            Y_train = np.array([0.0])
            mock_surrogate = MockSurrogateModel(X_train, Y_train)
            maximize_distance = MaximizeDistance()

            result = maximize_distance.optimize(mock_surrogate, bounds, n=1)
            assert result.shape == (1, dim)
            assert np.all(result >= bounds[:, 0]) and np.all(
                result <= bounds[:, 1]
            )

    def test_optimize_maximizes_min_distance(self):
        """
        Test that the optimize() method maximizes the minimum distance
        between points. Checks that the points returned are distinct
        and that they match expected values in simple scenarios.
        """
        bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
        maximize_distance = MaximizeDistance()

        # # Test 1: Only existing point is in corner of bounds
        # X_train = np.array([[0.0, 0.0]])
        # Y_train = np.array([0.0])
        # mock_surrogate = MockSurrogateModel(X_train, Y_train)
        # points = maximize_distance.optimize(mock_surrogate, bounds, n=4)
        # expected_points = np.array(
        #     [[10.0, 10.0], [10.0, 0.0], [0.0, 10.0], [5.0, 5.0]]
        # )

        # # Check that each point is different
        # assert len(np.unique(points, axis=0)) == 4

        # # Check that each returned point is one of the expected points
        # for point in points:
        #     assert np.any(np.all(np.isclose(expected_points, point), axis=1))

        # Test 2: Multiple existing points spread out
        x_train = np.array(
            [[5.0, 6.0], [2.0, 3.0], [8.0, 1.0], [1.0, 9.0], [7.0, 8.5]]
        )
        y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mock_surrogate = MockSurrogateModel(x_train, y_train)
        point = maximize_distance.optimize(mock_surrogate, bounds, n=1)

        # Check that the point is correct
        # Expected point was calculated with wolframalpha
        assert np.allclose(point, np.array([10.0, 5.0833]))

    def test_optimize_with_mixedint(self):
        """
        Test that the optimize() method works with mixed integer bounds.
        """
        bounds = np.array([[0.0, 10.0], [0, 10]])
        X_train = np.array([[5.0, 5], [6.0, 6], [3.0, 4]])
        Y_train = np.array([0.0, 1.0, 0.5])
        iindex = np.array([1])
        mock_surrogate = MockSurrogateModel(X_train, Y_train, iindex=iindex)
        maximize_distance = MaximizeDistance()

        result = maximize_distance.optimize(mock_surrogate, bounds, n=1)

        # Check that we get the expected number of points
        assert result.shape == (1, 2)

        # Check that all points are within bounds
        assert np.all(result >= np.array([bounds[:, 0]]))
        assert np.all(result <= np.array([bounds[:, 1]]))

        # Check that integer dimension values are actually integers
        integer_dim_values = result[
            :, 1
        ]  # Second dimension is integer (index 1)
        assert np.all(integer_dim_values == np.round(integer_dim_values))

        # Check that points are different from the training points
        for point in result:
            assert not np.any(np.all(np.isclose(point, X_train), axis=1))
