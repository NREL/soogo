"""Tests for integration wrappers."""

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

from soogo.integrations.pymoo import (
    PymooProblem,
    _get_vars,
    _dict_to_array,
    ListDuplicateElimination,
)
from soogo.optimize.result import OptimizeResult


class TestPymooProblem:
    """Tests for PymooProblem wrapper."""

    def test_initialization(self):
        """Test that PymooProblem can be initialized."""

        def objective(x):
            return np.sum(x**2, axis=1)

        bounds = [[0, 1], [0, 1]]
        problem = PymooProblem(objective, bounds)

        assert problem is not None
        assert problem.n_var == 2
        assert problem.n_obj == 1
        assert problem.n_ieq_constr == 0

    def test_initialization_with_integer_variables(self):
        """Test initialization with integer variables."""

        def objective(x):
            return np.sum(x, axis=1)

        bounds = [[0, 1], [0, 10]]
        iindex = [1]  # Second variable is integer
        problem = PymooProblem(objective, bounds, iindex)

        assert problem.n_var == 2
        # Check that variables are properly configured
        assert len(problem.vars) == 2

    def test_initialization_with_constraints(self):
        """Test initialization with constraints."""

        def objective(x):
            return np.sum(x**2, axis=1)

        def constraints(x):
            return np.array([x[:, 0] - 0.5])  # Single constraint

        bounds = [[0, 1], [0, 1]]
        problem = PymooProblem(
            objective, bounds, gfunc=constraints, n_ieq_constr=1
        )

        assert problem.n_var == 2
        assert problem.n_obj == 1
        assert problem.n_ieq_constr == 1

    def test_evaluate_single_objective(self):
        """Test evaluation of single-objective function."""

        def objective(x):
            return np.sum(x**2, axis=1)

        bounds = [[0, 1], [0, 1]]
        problem = PymooProblem(objective, bounds)

        # Evaluate at a point (continuous variables use arrays)
        x_array = np.array([[0.5, 0.5]])
        out = {}
        problem._evaluate(x_array, out)

        assert "F" in out
        assert out["F"][0] == pytest.approx(0.5, abs=0.01)

    def test_evaluate_with_constraints(self):
        """Test evaluation with constraints."""

        def objective(x):
            return np.sum(x**2, axis=1)

        def constraints(x):
            # Return constraint value (should be <= 0)
            return np.array([x[:, 0] - 0.5])

        bounds = [[0, 1], [0, 1]]
        problem = PymooProblem(
            objective, bounds, gfunc=constraints, n_ieq_constr=1
        )

        x_array = np.array([[0.3, 0.5]])
        out = {}
        problem._evaluate(x_array, out)

        assert "F" in out
        assert "G" in out  # Constraints present


class TestGetVars:
    """Tests for _get_vars helper function."""

    def test_all_continuous_variables(self):
        """Test with all continuous variables."""
        bounds = [[0, 1], [0, 2], [0, 3]]
        vars_dict = _get_vars(bounds)

        assert len(vars_dict) == 3
        assert all(k in vars_dict for k in range(3))

    def test_mixed_variables(self):
        """Test with mixed continuous and integer variables."""
        bounds = [[0, 1], [0, 10], [0, 100]]
        iindex = [1, 2]  # Last two are integers
        vars_dict = _get_vars(bounds, iindex)

        assert len(vars_dict) == 3


class TestDictToArray:
    """Tests for _dict_to_array helper function."""

    def test_single_dict(self):
        """Test conversion of single dictionary."""
        x_dict = {0: 0.5, 1: 0.7, 2: 0.3}
        array = _dict_to_array(x_dict)

        assert array.shape == (3,)
        assert np.allclose(array, [0.5, 0.7, 0.3])

    def test_list_of_dicts(self):
        """Test conversion of list of dictionaries."""
        x_list = [{0: 0.1, 1: 0.2}, {0: 0.3, 1: 0.4}, {0: 0.5, 1: 0.6}]
        array = _dict_to_array(x_list)

        assert array.shape == (3, 2)
        assert np.allclose(array[0], [0.1, 0.2])
        assert np.allclose(array[2], [0.5, 0.6])


class TestListDuplicateElimination:
    """Tests for ListDuplicateElimination class."""

    def test_initialization(self):
        """Test that ListDuplicateElimination can be initialized."""
        elim = ListDuplicateElimination()
        assert elim is not None


class TestNomadProblem:
    """Tests for NomadProblem wrapper."""

    def test_initialization(self):
        """Test that NomadProblem can be initialized."""
        try:
            from soogo.integrations.nomad import NomadProblem
        except ImportError:
            pytest.skip("PyNomad not available")

        def objective(x):
            return np.sum(x**2, axis=1)

        result = OptimizeResult()
        result.sample = np.empty((10, 2))
        result.fsample = np.empty(10)
        result.nfev = 0

        problem = NomadProblem(objective, result)

        assert problem is not None
        assert problem.func is objective
        assert problem.out is result

    def test_history_tracking(self):
        """Test that NomadProblem tracks evaluation history."""
        try:
            from soogo.integrations.nomad import NomadProblem
        except ImportError:
            pytest.skip("PyNomad not available")

        def objective(x):
            return np.sum(x**2, axis=1)

        result = OptimizeResult()
        result.sample = np.empty((10, 2))
        result.fsample = np.empty(10)
        result.nfev = 0

        problem = NomadProblem(objective, result)

        # Initially empty
        assert len(problem.get_x_history()) == 0
        assert len(problem.get_f_history()) == 0

    def test_reset(self):
        """Test reset functionality."""
        try:
            from soogo.integrations.nomad import NomadProblem
        except ImportError:
            pytest.skip("PyNomad not available")

        def objective(x):
            return np.sum(x**2, axis=1)

        result = OptimizeResult()
        result.sample = np.empty((10, 2))
        result.fsample = np.empty(10)
        result.nfev = 0

        problem = NomadProblem(objective, result)
        problem.reset()

        assert len(problem.get_x_history()) == 0
        assert len(problem.get_f_history()) == 0
