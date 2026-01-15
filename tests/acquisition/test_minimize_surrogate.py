"""Tests for MinimizeSurrogate acquisition function."""

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

from soogo.acquisition import MinimizeSurrogate
from .utils import MockSurrogateModel


class TestMinimizeSurrogate:
    """Tests for MinimizeSurrogate acquisition function."""

    def test_initialization(self):
        """Test that MinimizeSurrogate can be initialized."""
        acq = MinimizeSurrogate(pool_size=100)
        assert acq is not None
        assert acq.pool_size == 100
        assert hasattr(acq, "rng")

    def test_initialization_with_rtol(self):
        """Test initialization with custom rtol."""
        acq = MinimizeSurrogate(pool_size=100, rtol=1e-2)
        assert acq.rtol == 1e-2

    def test_optimize_returns_array(self):
        """Test that optimize returns a numpy array."""
        X_train = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]])
        Y_train = np.array([0.5, 1.0, 1.5])
        model = MockSurrogateModel(X_train, Y_train)

        acq = MinimizeSurrogate(pool_size=50)
        bounds = [[0, 1], [0, 1]]
        result = acq.optimize(model, bounds, n=1)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 2)

    def test_optimize_multiple_points(self):
        """Test acquiring multiple points."""
        X_train = np.array([[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]])
        Y_train = np.array([0.5, 1.0, 1.5])
        model = MockSurrogateModel(X_train, Y_train)

        acq = MinimizeSurrogate(pool_size=100)
        bounds = [[0, 1], [0, 1]]
        result = acq.optimize(model, bounds, n=3)

        assert len(result) <= 3
        # All points should be within bounds
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_optimize_respects_bounds(self):
        """Test that optimized points respect the bounds."""
        X_train = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
        Y_train = np.array([2.0, 3.0, 4.0])
        model = MockSurrogateModel(X_train, Y_train)

        acq = MinimizeSurrogate(pool_size=50)
        bounds = [[0, 5], [4, 10]]
        result = acq.optimize(model, bounds, n=2)

        assert np.all(result[:, 0] >= 0) and np.all(result[:, 0] <= 5)
        assert np.all(result[:, 1] >= 4) and np.all(result[:, 1] <= 10)

    def test_optimize_with_higher_dimensions(self):
        """Test optimization in higher dimensions."""
        X_train = np.array([[0.2, 0.3, 0.4], [0.5, 0.5, 0.6]])
        Y_train = np.array([0.9, 1.6])
        model = MockSurrogateModel(X_train, Y_train)

        acq = MinimizeSurrogate(pool_size=100)
        bounds = [[0, 1], [0, 1], [0, 1]]
        result = acq.optimize(model, bounds, n=2)

        assert len(result) <= 2
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_optimize_with_integer_variables(self):
        """Test optimization with integer design variables."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y_train = np.array([3.0, 7.0])
        iindex = np.array([1])  # Second dimension is integer
        model = MockSurrogateModel(X_train, Y_train, iindex=iindex)

        acq = MinimizeSurrogate(pool_size=50)
        bounds = [[0, 5], [0, 10]]
        result = acq.optimize(model, bounds, n=2)

        assert len(result) <= 2
        # Integer variables should be integers
        assert np.allclose(result[:, 1], np.round(result[:, 1]))
