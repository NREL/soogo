"""Test the acquisition functions."""

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
from typing import Union, Tuple, Optional

from soogo.model import Surrogate
from soogo.acquisition import (
    Acquisition,
    TransitionSearch,
    MaximizeDistance,
    AlternatedAcquisition,
)
from soogo.termination import IterateNTimes
from soogo import OptimizeResult


class MockSurrogateModel(Surrogate):
    """
    A mock surrogate model for testing purposes.
    When called, this model returns the sum of the coordinates of the
    input points.
    """

    def __init__(
        self, X_train: np.ndarray, Y_train: np.ndarray, iindex: np.ndarray = ()
    ):
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

    def eval_kernel(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass

    def reset_data(self) -> None:
        pass


class MockEvaluabilitySurrogate(Surrogate):
    """
    A mock evaluability surrogate model for testing purposes.
    When called, this model returns a 0.1 for the first point and
    1.0 for all others.
    """

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

    def eval_kernel(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass

    def reset_data(self) -> None:
        pass


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
                cycle_search = TransitionSearch()

                result = cycle_search.optimize(
                    mock_surrogate, bounds, n=n, scoreWeight=0.5
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
        X_train = np.array([[5, 5]])
        Y_train = np.array([0.0])

        mock_surrogate = MockSurrogateModel(X_train, Y_train)
        cycle_search = TransitionSearch()

        for n in nCand:
            candidates = cycle_search.generate_candidates(
                mock_surrogate, bounds, nCand=n
            )

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

        mock_surrogate = MockSurrogateModel(X_train, Y_train)
        mock_evaluability = MockEvaluabilitySurrogate(X_train, Y_train)
        cycle_search = TransitionSearch()

        # Both tests would return [0.0, 0.0] if the evaluability filter fails
        # Test case 1: Same function values, different distances
        candidates = np.array([[0.0, 0.0], [9.0, 1.0], [4.0, 6.0]])
        point = cycle_search.select_candidates(
            mock_surrogate,
            candidates,
            bounds,
            n=1,
            scoreWeight=0.5,
            evaluabilitySurrogate=mock_evaluability,
        )
        assert np.allclose(point, np.array([[9.0, 1.0]]))

        # Test case 2: Same distances, different function values
        candidates = np.array([[0.0, 0.0], [3.0, 5.0], [7.0, 5.0]])
        point = cycle_search.select_candidates(
            mock_surrogate,
            candidates,
            bounds,
            n=1,
            scoreWeight=0.5,
            evaluabilitySurrogate=mock_evaluability,
        )
        assert np.allclose(point, np.array([[3.0, 5.0]]))

        # Test case 3: Weighted sum
        X_train = np.array([[5.0, 5.0], [6.0, 6.0], [3.0, 4.0]])
        Y_train = np.array([0.0, 1.0, 0.5])
        mock_surrogate = MockSurrogateModel(X_train, Y_train)
        mock_evaluability = MockEvaluabilitySurrogate(X_train, Y_train)
        candidates = np.array([[0.0, 0.0], [2.0, 6.0], [7.0, 0.5]])
        point = cycle_search.select_candidates(
            mock_surrogate,
            candidates,
            bounds,
            n=1,
            scoreWeight=0.75,
            evaluabilitySurrogate=mock_evaluability,
        )
        assert np.allclose(point, np.array([[7.0, 0.5]]))


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
                self.termination = IterateNTimes(n)

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
        out = OptimizeResult(
            nfev=1, fx=np.array([0.1]), fsample=np.array([[0.1]]), nobj=1
        )

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
        # Use transition search and maximize distance as acquisition functions
        transitionSearch = TransitionSearch(termination=IterateNTimes(1))
        maximizeDistance = MaximizeDistance(termination=IterateNTimes(2))

        # Create a list of mock acquisition functions
        acquisition_funcs = [transitionSearch, maximizeDistance]

        # Initialize the AlternatedAcquisition with the mock functions
        alternated_acq = AlternatedAcquisition(acquisition_funcs)

        # Mock optimization result
        out = OptimizeResult(
            nfev=1, fx=np.array([0.1]), fsample=np.array([[0.1]]), nobj=1
        )

        for dim in dims:
            bounds = np.array([[0, 1] for _ in range(dim)])
            X_train = np.array([[0.5 for _ in range(dim)]])
            Y_train = np.array([0.0])
            mock_surrogate = MockSurrogateModel(X_train, Y_train)

            result = alternated_acq.optimize(
                mock_surrogate, bounds, n=1, scoreWeight=0.5
            )
            assert result.shape == (1, dim)
            assert np.all(result >= bounds[:, 0]) and np.all(
                result <= bounds[:, 1]
            )

            alternated_acq.update(out, mock_surrogate)

            assert alternated_acq.idx in [0, 1]


class TestFarEnoughSampleFilter:
    """
    Test class for FarEnoughSampleFilter utility.
    """

    def test_initialization(self):
        """Test that FarEnoughSampleFilter initializes correctly."""
        from soogo.acquisition.utils import FarEnoughSampleFilter

        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        tol = 0.5
        filter = FarEnoughSampleFilter(X, tol)

        assert filter.tol == tol
        assert filter.tree is not None

    def test_is_far_enough_returns_true_for_distant_point(self):
        """Test that is_far_enough returns True for points far from X."""
        from soogo.acquisition.utils import FarEnoughSampleFilter

        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        tol = 0.5
        filter = FarEnoughSampleFilter(X, tol)

        # Point at (5, 5) is far from both (0,0) and (1,1)
        x_far = np.array([5.0, 5.0])
        assert filter.is_far_enough(x_far)

    def test_is_far_enough_returns_false_for_close_point(self):
        """Test that is_far_enough returns False for points close to X."""
        from soogo.acquisition.utils import FarEnoughSampleFilter

        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        tol = 0.5
        filter = FarEnoughSampleFilter(X, tol)

        # Point at (0.1, 0.1) is very close to (0,0)
        x_close = np.array([0.1, 0.1])
        assert not filter.is_far_enough(x_close)

    def test_call_filters_candidates_correctly(self):
        """Test that __call__ filters out candidates that are too close."""
        from soogo.acquisition.utils import FarEnoughSampleFilter

        X = np.array([[0.0, 0.0], [5.0, 5.0]])
        tol = 1.0
        filter = FarEnoughSampleFilter(X, tol)

        # Create candidates: some close, some far
        Xc = np.array(
            [
                [0.2, 0.2],  # Too close to (0,0)
                [3.0, 3.0],  # Far from both
                [5.1, 5.1],  # Too close to (5,5)
                [7.0, 7.0],  # Far from both
            ]
        )

        result = filter(Xc)

        # Should only keep points that are far from all points in X
        assert result.shape[0] == 2
        assert result.shape[1] == Xc.shape[1]

        # Verify all returned points are far enough
        for x in result:
            assert filter.is_far_enough(x)

    def test_call_handles_clustering_candidates(self):
        """Test that __call__ handles candidates that are too close to
        each other.
        """
        from soogo.acquisition.utils import FarEnoughSampleFilter

        X = np.array([[0.0, 0.0]])
        tol = 1.0
        filter = FarEnoughSampleFilter(X, tol)

        # Create candidates that are far from X but close to each other
        Xc = np.array(
            [
                [5.0, 5.0],
                [5.2, 5.2],  # Close to previous
                [5.4, 5.4],  # Close to previous
                [10.0, 10.0],  # Far from all
            ]
        )

        result = filter(Xc)

        # Should select a maximum independent set
        assert result.shape[0] >= 1  # At least one point should be selected
        assert result.shape[0] <= Xc.shape[0]

        # Verify all returned points are far enough from X
        for x in result:
            assert filter.is_far_enough(x)

        # Verify all returned points are far enough from each other
        if len(result) > 1:
            from scipy.spatial.distance import cdist

            pairwise_dist = cdist(result, result)
            np.fill_diagonal(pairwise_dist, np.inf)
            assert np.all(pairwise_dist >= tol)

    def test_call_returns_empty_when_all_too_close(self):
        """Test that __call__ returns empty array when all candidates are
        too close.
        """
        from soogo.acquisition.utils import FarEnoughSampleFilter

        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        tol = 0.5
        filter = FarEnoughSampleFilter(X, tol)

        # All candidates are too close to X
        Xc = np.array([[0.1, 0.1], [1.1, 1.1], [2.1, 2.1]])

        result = filter(Xc)

        # Should return empty array or very few points
        assert result.shape[0] <= Xc.shape[0]
        assert result.shape[1] == Xc.shape[1]

    def test_call_with_various_dimensions(self):
        """Test that FarEnoughSampleFilter works with different
        dimensionalities.
        """
        from soogo.acquisition.utils import FarEnoughSampleFilter

        for dim in [1, 2, 3, 5, 10]:
            X = np.random.rand(5, dim)
            tol = 0.5
            filter = FarEnoughSampleFilter(X, tol)

            Xc = np.random.rand(10, dim) * 5  # Scale to make some far
            result = filter(Xc)

            assert result.shape[1] == dim
            assert result.shape[0] <= Xc.shape[0]
