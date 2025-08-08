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
import pytest
from typing import Union, Tuple, Optional

from soogo.model.base import Surrogate
from soogo.acquisition import CycleSearch, MaximizeDistance

class TestCycleSearch:
    """Test suite for the CycleSearch acquisition function."""

    class MockSurrogateModel(Surrogate):
        def __init__(self, X_train: np.ndarray, Y_train: np.ndarray):
            self._X = X_train.copy()
            self._Y = Y_train.copy()
            self._iindex = ()

        @property
        def X(self) -> np.ndarray:
            return self._X

        @property
        def Y(self) -> np.ndarray:
            return self._Y

        @property
        def iindex(self) -> np.ndarray:
            return self._iindex

        def reserve(self, n: int, dim: int, ntarget: int = 1) -> None:
            pass

        def __call__(
            self, x: np.ndarray, i: int = -1, **kwargs
        ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
            """
            Return sum of coords (x + y).
            """
            x = np.atleast_2d(x)
            result = np.sum(x, axis=1)
            return result if len(result) > 1 else result[0]

        def update(self, x: np.ndarray, y: np.ndarray) -> None:
            pass

        def min_design_space_size(self, dim: int) -> int:
            pass

        def check_initial_design(self, sample: np.ndarray) -> bool:
            pass

        def eval_kernel(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
            pass

        def reset_data(self) -> None:
            pass

    class MockEvaluabilitySurrogate(Surrogate):
        def __init__(self, X_train: np.ndarray, Y_train: np.ndarray):
            self._X = X_train.copy()
            self._Y = Y_train.copy()
            self._iindex = ()

        @property
        def X(self) -> np.ndarray:
            return self._X

        @property
        def Y(self) -> np.ndarray:
            return self._Y

        @property
        def iindex(self) -> np.ndarray:
            return self._iindex

        def reserve(self, n: int, dim: int, ntarget: int = 1) -> None:
            pass

        def __call__(
            self, x: np.ndarray, i: int = -1, **kwargs
        ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
            """
            Return 1.0 except for the first coord which returns 0.1.
            """
            x = np.atleast_2d(x)
            result = np.ones(x.shape[0])
            result[0] = 0.1
            return result if len(result) > 1 else result[0]

        def update(self, x: np.ndarray, y: np.ndarray) -> None:
            pass

        def min_design_space_size(self, dim: int) -> int:
            pass

        def check_initial_design(self, sample: np.ndarray) -> bool:
            pass

        def eval_kernel(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
            pass

        def reset_data(self) -> None:
            pass

    @pytest.mark.parametrize(["n_points", "dims"], [([1], [2, 5, 25])])
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
                mock_surrogate = self.MockSurrogateModel(X_train, Y_train)
                cycle_search = CycleSearch()

                result = cycle_search.optimize(
                    mock_surrogate, bounds, n=n, scoreWeight=0.5)
                assert result.shape == (n, dim)
                assert np.all(result >= bounds[:, 0]) and np.all(result <= bounds[:, 1])

    def test_generate_candidates(self):
        """
        Tests that the generate_candidates() method:
        - Generates the expected number of candidates.
        - All candidates are within the specified bounds.
        """
        nCand = [200, 1000, 100000]
        bounds = np.array([[0, 10], [0, 10]])
        X_train = np.array([[5, 5]])
        Y_train = np.array([0.0])

        mock_surrogate = self.MockSurrogateModel(X_train, Y_train)
        cycle_search = CycleSearch()

        for n in nCand:
            candidates = cycle_search.generate_candidates(mock_surrogate, bounds, nCand=n)

            # Should generate 2 * nCand candidates (perturbations + uniform)
            expected_count = 2 * n
            assert len(candidates) == expected_count

            # All candidates should be within bounds
            assert np.all(candidates >= bounds[:, 0])
            assert np.all(candidates <= bounds[:, 1])

    def test_select_candidates(self):
        """
        Test that the select_candidates() method:
        - Chooses the candidate further from evaluated points when function
            values are the same.
        - Chooses the candidate with lower function value when distances are
            the same.
        - Removes candidates that are below the evaluability threshold.
        """
        X_train = np.array([[5, 5]])
        Y_train = np.array([0.0])
        bounds = np.array([[0, 10], [0, 10]])

        mock_surrogate = self.MockSurrogateModel(X_train, Y_train)
        mock_evaluability = self.MockEvaluabilitySurrogate(X_train, Y_train)
        cycle_search = CycleSearch()

        # Both tests would return [0.0, 0.0] if the evaluability filter fails
        # Test case 1: Same function values, different distances
        candidates = np.array([[0.0, 0.0], [9.0, 1.0], [4.0, 6.0]])
        point = cycle_search.select_candidates(mock_surrogate, candidates, bounds, n=1, scoreWeight=0.5, evaluabilitySurrogate=mock_evaluability)
        assert np.allclose(point, np.array([[9.0, 1.0]]))

        # Test case 2: Same distances, different function values
        candidates = np.array([[0.0, 0.0], [3.0, 5.0], [7.0, 5.0]])
        point = cycle_search.select_candidates(mock_surrogate, candidates, bounds, n=1, scoreWeight=0.5, evaluabilitySurrogate=mock_evaluability)
        assert np.allclose(point, np.array([[3.0, 5.0]]))

        # Test case 3: Weighted sum
        X_train = np.array([[5.0, 5.0], [6.0, 6.0], [3.0, 4.0]])
        Y_train = np.array([0.0, 1.0, 0.5])
        mock_surrogate = self.MockSurrogateModel(X_train, Y_train)
        mock_evaluability = self.MockEvaluabilitySurrogate(X_train, Y_train)
        candidates = np.array([[0.0, 0.0], [2.0, 6.0], [7.0, 0.5]])
        point = cycle_search.select_candidates(mock_surrogate, candidates, bounds, n=1, scoreWeight=0.75, evaluabilitySurrogate=mock_evaluability)
        assert np.allclose(point, np.array([[7.0, 0.5]]))

class TestMaximizeDistance:
    """Test suite for the MaximizeDistance acquisition function."""

    class MockSurrogateModel(Surrogate):
        def __init__(self, X_train: np.ndarray, Y_train: np.ndarray, iindex: np.ndarray = ()):
            self._X = X_train.copy()
            self._Y = Y_train.copy()
            self._iindex = iindex

        @property
        def X(self) -> np.ndarray:
            return self._X

        @property
        def Y(self) -> np.ndarray:
            return self._Y

        @property
        def iindex(self) -> np.ndarray:
            return self._iindex

        def reserve(self, n: int, dim: int, ntarget: int = 1) -> None:
            pass

        def __call__(
            self, x: np.ndarray, i: int = -1, **kwargs
        ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
            """
            Return sum of coords (x + y).
            """
            x = np.atleast_2d(x)
            result = np.sum(x, axis=1)
            return result if len(result) > 1 else result[0]

        def update(self, x: np.ndarray, y: np.ndarray) -> None:
            pass

        def min_design_space_size(self, dim: int) -> int:
            pass

        def check_initial_design(self, sample: np.ndarray) -> bool:
            pass

        def eval_kernel(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
            pass

        def reset_data(self) -> None:
            pass

    @pytest.mark.parametrize(["n_points", "dims"], [([1], [2, 5, 25])])
    def test_optimize_generates_expected_points(self, dims, n_points):
        """
        Test the output points of optimize().

        Ensures that the generated points are:
        - Within the specified bounds.
        - The expected shape (n_points, dims).
        - The amount requested.
        """
        for dim in dims:
            for n in n_points:
                bounds = np.array([[0, 1] for _ in range(dim)])
                X_train = np.array([[0.5 for _ in range(dim)]])
                Y_train = np.array([0.0])
                mock_surrogate = self.MockSurrogateModel(X_train, Y_train)
                maximize_distance = MaximizeDistance()

                result = maximize_distance.optimize(
                    mock_surrogate, bounds, n=n, scoreWeight=0.5)
                assert result.shape == (n, dim)
                assert np.all(result >= bounds[:, 0]) and np.all(result <= bounds[:, 1])

    def test_optimize_maximizes_min_distance(self):
        """
        Test that the optimize() method maximizes the minimum distance
        between points. Checks that the points returned are distinct
        and that they match expected values in simple scenarios.
        """
        bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
        maximize_distance = MaximizeDistance()

        # Test 1: Only existing point is in corner of bounds
        X_train = np.array([[0.0, 0.0]])
        Y_train = np.array([0.0])
        mock_surrogate = self.MockSurrogateModel(X_train, Y_train)
        points = maximize_distance.optimize(mock_surrogate, bounds, n=4)
        expected_points = np.array([[10.0, 10.0], [10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])

        # Check that each point is different
        assert len(np.unique(points, axis=0)) == 4

        # Check that each returned point is one of the expected points
        for point in points:
            assert np.any(np.all(np.isclose(expected_points, point), axis=1))

        # Test 2: Multiple existing points spread out
        x_train = np.array([[5.0, 6.0], [2.0, 3.0], [8.0, 1.0], [1.0, 9.0], [7.0, 8.5]])
        y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        mock_surrogate = self.MockSurrogateModel(x_train, y_train)
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
        mock_surrogate = self.MockSurrogateModel(X_train, Y_train, iindex=iindex)
        maximize_distance = MaximizeDistance()

        result = maximize_distance.optimize(mock_surrogate, bounds, n=2)

        # Check that we get the expected number of points
        assert result.shape == (2, 2)

        # Check that all points are within bounds
        assert np.all(result >= np.array([bounds[:, 0]]))
        assert np.all(result <= np.array([bounds[:, 1]]))

        # Check that integer dimension values are actually integers
        integer_dim_values = result[:, 1]  # Second dimension is integer (index 1)
        assert np.all(integer_dim_values == np.round(integer_dim_values))

        # Check that points are different from the training points
        for point in result:
            assert not np.any(np.all(np.isclose(point, X_train), axis=1))
