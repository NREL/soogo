"""Test OptimizeResult class."""

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

from soogo import OptimizeResult
from tests.acquisition.utils import MockSurrogateModel


class TestOptimizeResultInitialization:
    """Test OptimizeResult initialization and basic properties."""

    def test_default_initialization(self):
        """Test default initialization of OptimizeResult."""
        result = OptimizeResult()
        assert result.x is None
        assert result.fx is None
        assert result.nit == 0
        assert result.nfev == 0
        assert len(result.sample) == 0
        assert len(result.sample) == 0
        assert result.nobj == 1


class TestOptimizeResultInit:
    """Test OptimizeResult.init() method."""

    def test_init_with_empty_surrogate(self):
        """Test init when surrogate has no initial data."""
        result = OptimizeResult()
        bounds = [[0, 1], [0, 1], [0, 1]]
        maxeval = 100

        def fun(x):
            return np.sum(x**2, axis=1)

        # Empty surrogate model
        X_train = np.empty((0, 3))
        Y_train = np.empty((0,))
        model = MockSurrogateModel(X_train, Y_train)

        result.init(
            fun, bounds, mineval=20, maxeval=maxeval, surrogateModel=model
        )

        # Should have created initial samples
        assert result.nobj == 1
        assert result.nfev >= 20
        assert result.sample.shape[0] == maxeval
        assert result.sample.shape[1] == 3
        assert result.fsample.ndim == 1
        assert not np.isnan(result.sample[0 : result.nfev]).any()
        assert np.isnan(result.sample[result.nfev :]).all()
        assert not np.isnan(result.fsample[0 : result.nfev]).any()
        assert np.isnan(result.fsample[result.nfev :]).all()

    def test_init_with_pretrained_surrogate(self):
        """Test init when surrogate already has training data."""
        result = OptimizeResult()
        bounds = [[0, 1], [0, 1]]
        maxeval = 50

        def fun(x):
            return np.sum(x, axis=1)

        # Surrogate with existing data
        X_train = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        Y_train = np.array([0.3, 0.7, 1.1])
        model = MockSurrogateModel(X_train, Y_train)

        result.init(
            fun, bounds, mineval=10, maxeval=maxeval, surrogateModel=model
        )

        # Should not create new samples since surrogate already has data
        assert result.sample.shape == (maxeval, 2)
        assert result.fsample.shape == (maxeval,)
        assert result.nfev == 0
        assert result.nobj == 1

    def test_init_with_multiobjective_function(self):
        """Test init with multi-objective function."""
        result = OptimizeResult()
        bounds = [[0, 1], [0, 1]]
        maxeval = 50

        # Multi-objective function
        def fun_multi(x):
            f1 = np.sum(x, axis=1)
            f2 = np.sum(x**2, axis=1)
            return np.column_stack([f1, f2])

        X_train = np.empty((0, 2))
        Y_train = np.empty((0,))
        model = MockSurrogateModel(X_train, Y_train)

        result.init(
            fun_multi,
            bounds,
            mineval=10,
            maxeval=maxeval,
            surrogateModel=model,
        )

        assert result.nobj == 2
        assert result.fsample.shape[1] == 2

    def test_init_with_integer_variables(self):
        """Test init with integer design variables."""
        result = OptimizeResult()
        bounds = [[0, 10], [0, 10]]
        maxeval = 50

        def fun(x):
            return np.sum(x, axis=1)

        # Surrogate with integer index
        X_train = np.empty((0, 2))
        Y_train = np.empty((0,))
        iindex = np.array([1])  # Second dimension is integer
        model = MockSurrogateModel(X_train, Y_train, iindex=iindex)

        result.init(
            fun, bounds, mineval=10, maxeval=maxeval, surrogateModel=model
        )

        # Check that integer dimensions are actually integers
        integer_values = result.sample[0 : result.nfev, 1]
        assert np.all(integer_values == np.round(integer_values))


class TestOptimizeResultInitBestValues:
    """Test OptimizeResult.init_best_values() method."""

    def test_init_best_values_single_objective(self):
        """Test init_best_values for single objective."""
        result = OptimizeResult()
        result.sample = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result.fsample = np.array([10.0, 5.0, 15.0])
        result.nfev = 3
        result.nobj = 1

        result.init_best_values()

        # Should select point with minimum function value
        assert np.array_equal(result.x, np.array([3.0, 4.0]))
        assert result.fx == 5.0

    def test_init_best_values_with_surrogate(self):
        """Test init_best_values considering surrogate model data."""
        result = OptimizeResult()
        result.sample = np.array([[1.0, 2.0], [3.0, 4.0]])
        result.fsample = np.array([10.0, 8.0])
        result.nfev = 2
        result.nobj = 1

        # Surrogate has even better point
        X_train = np.array([[0.5, 0.5]])
        Y_train = np.array([3.0])
        model = MockSurrogateModel(X_train, Y_train)

        result.init_best_values(model)

        # Should select point from surrogate with best value
        assert np.array_equal(result.x, np.array([0.5, 0.5]))
        assert result.fx == 3.0

    def test_init_best_values_multi_objective(self):
        """Test init_best_values for multi-objective problems."""
        result = OptimizeResult()
        result.sample = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [2.0, 3.0]]
        )
        result.fsample = np.array(
            [[10.0, 5.0], [5.0, 10.0], [15.0, 15.0], [7.0, 7.0]]
        )
        result.nfev = 4
        result.nobj = 2

        result.init_best_values()

        # Should return Pareto front
        assert result.x.shape[0] >= 2  # At least 2 points on Pareto front
        assert result.fx.shape[0] == result.x.shape[0]
        assert result.fx.shape[1] == 2

    def test_init_best_values_with_nan_values(self):
        """Test that NaN values in fsample don't affect best value selection."""
        result = OptimizeResult()
        result.sample = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result.fsample = np.array([10.0, 5.0, np.nan])
        result.nfev = 2  # Only first 2 evaluations are valid
        result.nobj = 1

        result.init_best_values()

        # Should ignore NaN values
        assert np.array_equal(result.x, np.array([3.0, 4.0]))
        assert result.fx == 5.0

    def test_init_best_values_multi_objective_with_surrogate(self):
        """Test multi-objective init_best_values with surrogate data."""
        result = OptimizeResult()
        result.sample = np.array([[1.0, 2.0], [3.0, 4.0]])
        result.fsample = np.array([[10.0, 5.0], [5.0, 10.0]])
        result.nfev = 2
        result.nobj = 2

        # Surrogate has additional Pareto point
        X_train = np.array([[2.0, 3.0]])
        Y_train = np.array([[7.0, 7.0]])
        model = MockSurrogateModel(X_train, Y_train)

        result.init_best_values(model)

        # Should include points from both result and surrogate on Pareto front
        assert result.x.shape[0] >= 2
        assert result.fx.shape[0] == result.x.shape[0]

    def test_init_best_values_single_point(self):
        """Test init_best_values with only one evaluation."""
        result = OptimizeResult()
        result.sample = np.array([[1.0, 2.0]])
        result.fsample = np.array([10.0])
        result.nfev = 1
        result.nobj = 1

        result.init_best_values()

        assert np.array_equal(result.x, np.array([1.0, 2.0]))
        assert result.fx == 10.0

    def test_init_best_values_with_2d_fsample(self):
        """Test init_best_values when fsample is 2D for single objective."""
        result = OptimizeResult()
        result.sample = np.array([[1.0, 2.0], [3.0, 4.0]])
        result.fsample = np.array([[10.0, -100.0], [5.0, 100.0]])  # 2D array
        result.nfev = 2
        result.nobj = 1

        result.init_best_values()

        assert np.array_equal(result.x, np.array([3.0, 4.0]))
        assert np.array_equal(result.fx, np.array([5.0, 100.0]))


class TestOptimizeResultEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_result(self):
        """Test that empty result can be created."""
        result = OptimizeResult()
        assert result.nfev == 0
        assert result.nit == 0

    def test_result_with_zero_dimensions(self):
        """Test that init fails with zero dimensions."""
        result = OptimizeResult()
        bounds = []  # No dimensions
        maxeval = 50

        def fun(x):
            return np.array([0.0])

        X_train = np.empty((0, 0))
        Y_train = np.empty((0,))
        model = MockSurrogateModel(X_train, Y_train)

        with pytest.raises(AssertionError):
            result.init(
                fun, bounds, mineval=10, maxeval=maxeval, surrogateModel=model
            )

    def test_init_best_values_requires_sample(self):
        """Test that init_best_values requires sample to be set."""
        result = OptimizeResult()
        result.sample = None
        result.fsample = None
        result.nfev = 0
        result.nobj = 1

        with pytest.raises((AttributeError, AssertionError)):
            result.init_best_values()

    def test_result_dataclass_fields(self):
        """Test that OptimizeResult has expected dataclass fields."""
        result = OptimizeResult()

        # Check all expected fields exist
        assert hasattr(result, "x")
        assert hasattr(result, "fx")
        assert hasattr(result, "nit")
        assert hasattr(result, "nfev")
        assert hasattr(result, "sample")
        assert hasattr(result, "fsample")
        assert hasattr(result, "nobj")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
