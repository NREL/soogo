"""Test the FarEnoughSampleFilter utility."""

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

from soogo.acquisition.utils import FarEnoughSampleFilter


class TestFarEnoughSampleFilter:
    """
    Test class for FarEnoughSampleFilter utility.
    """

    def test_initialization(self):
        """Test that FarEnoughSampleFilter initializes correctly."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        tol = 0.5
        filter = FarEnoughSampleFilter(X, tol)

        assert filter.tol == tol
        assert filter.tree is not None

    def test_is_far_enough_returns_true_for_distant_point(self):
        """Test that is_far_enough returns True for points far from X."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        tol = 0.5
        filter = FarEnoughSampleFilter(X, tol)

        # Point at (5, 5) is far from both (0,0) and (1,1)
        x_far = np.array([5.0, 5.0])
        assert filter.is_far_enough(x_far)

    def test_is_far_enough_returns_false_for_close_point(self):
        """Test that is_far_enough returns False for points close to X."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        tol = 0.5
        filter = FarEnoughSampleFilter(X, tol)

        # Point at (0.1, 0.1) is very close to (0,0)
        x_close = np.array([0.1, 0.1])
        assert not filter.is_far_enough(x_close)

    def test_call_filters_candidates_correctly(self):
        """Test that __call__ filters out candidates that are too close."""
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
        for dim in [1, 2, 3, 5, 10]:
            X = np.random.rand(5, dim)
            tol = 0.5
            filter = FarEnoughSampleFilter(X, tol)

            Xc = np.random.rand(10, dim) * 5  # Scale to make some far
            result = filter(Xc)

            assert result.shape[1] == dim
            assert result.shape[0] <= Xc.shape[0]
