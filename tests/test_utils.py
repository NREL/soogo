"""Test utility functions."""

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

from soogo.utils import find_pareto_front, gp_expected_improvement


class TestFindParetoFront:
    """Test suite for find_pareto_front function."""

    def test_single_point(self):
        """Single point should form Pareto front."""
        fx = [[1.0, 2.0]]
        result = find_pareto_front(fx)
        assert result == [0]

    def test_two_dominated_points(self):
        """One point dominates the other."""
        fx = np.array([[1.0, 2.0], [2.0, 3.0]])  # First dominates second
        result = find_pareto_front(fx)
        assert result == [0]

    def test_two_non_dominated_points(self):
        """Two points on Pareto front."""
        fx = np.array([[1.0, 3.0], [2.0, 2.0]])  # Neither dominates
        result = find_pareto_front(fx)
        assert result == [0, 1]

    def test_three_points_one_dominated(self):
        """Three points, one dominated."""
        fx = np.array(
            [[1.0, 3.0], [2.0, 2.0], [3.0, 4.0]]
        )  # Third is dominated
        result = find_pareto_front(fx)
        assert result == [0, 1]

    def test_all_dominated(self):
        """One point dominates all others."""
        fx = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        result = find_pareto_front(fx)
        assert result == [0]

    def test_none_dominated(self):
        """All points are non-dominated."""
        fx = np.array(
            [[1.0, 5.0], [2.0, 4.0], [3.0, 3.0], [4.0, 2.0], [5.0, 1.0]]
        )
        result = find_pareto_front(fx)
        assert result == [0, 1, 2, 3, 4]

    def test_three_dimensions(self):
        """Test with 3D objective space."""
        fx = np.array(
            [
                [1.0, 2.0, 3.0],  # On Pareto front
                [2.0, 1.0, 4.0],  # On Pareto front
                [3.0, 3.0, 1.0],  # On Pareto front
                [2.0, 2.0, 2.0],  # Actually NOT dominated - on Pareto front
            ]
        )
        result = find_pareto_front(fx)
        # All points are non-dominated in this case
        assert result == [0, 1, 2, 3]

    def test_five_dimensions(self):
        """Test with higher dimensional space."""
        fx = np.array(
            [
                [1, 1, 1, 1, 5],
                [1, 1, 1, 5, 1],
                [
                    2,
                    2,
                    2,
                    2,
                    2,
                ],  # Actually on Pareto front (ties in first 3 dims)
            ]
        )
        result = find_pareto_front(fx)
        assert result == [0, 1, 2]

    def test_with_istart(self):
        """Test with iStart parameter."""
        # Points 0 and 1 are already known to be in Pareto front
        fx = np.array([[1.0, 5.0], [5.0, 1.0], [2.0, 2.0]])
        result = find_pareto_front(fx, iStart=2)
        # Point 2 is also on Pareto front (not dominated by 0 or 1)
        assert result == [0, 1, 2]

    def test_with_istart_new_dominates(self):
        """Test iStart when new point dominates old ones."""
        fx = np.array(
            [[2.0, 2.0], [3.0, 3.0], [1.0, 1.0]]
        )  # Last dominates first two
        result = find_pareto_front(fx, iStart=2)
        assert result == [2]

    def test_with_equal_values(self):
        """Test with some equal objective values."""
        fx = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 3.0]])
        result = find_pareto_front(fx)
        # First two are identical and dominate third
        assert 2 not in result
        assert 0 in result and 1 in result

    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        fx = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
        result = find_pareto_front(fx)
        assert result == [0, 1]

    def test_complex_pareto_front(self):
        """Test a more complex case."""
        fx = np.array(
            [
                [1.0, 10.0],  # On Pareto front
                [2.0, 9.0],  # On Pareto front
                [3.0, 8.0],  # On Pareto front
                [4.0, 7.0],  # On Pareto front
                [5.0, 6.0],  # On Pareto front
                [6.0, 5.0],  # On Pareto front
                [3.0, 9.0],  # Dominated
                [4.0, 8.0],  # Dominated
            ]
        )
        result = find_pareto_front(fx)
        assert result == [0, 1, 2, 3, 4, 5]


class TestGpExpectedImprovement:
    """Test suite for gp_expected_improvement function."""

    def test_zero_sigma_returns_delta(self):
        """When sigma is very small, EI approaches delta (when positive)."""
        delta = 1.0
        sigma = 1e-10  # Very small sigma
        result = gp_expected_improvement(delta, sigma)
        # When sigma is very small, EI ≈ delta * cdf(delta/sigma)
        # For large delta/sigma, cdf ≈ 1, so EI ≈ delta
        assert np.isclose(result, delta, rtol=1e-3)

    def test_at_minimum(self):
        """Test EI when predicted value equals current best."""
        delta = 0.0
        sigma = 1.0
        result = gp_expected_improvement(delta, sigma)
        expected = 0.39894  # sigma * pdf(0) = 1.0 * 0.39894...
        assert np.isclose(result, expected, rtol=1e-4)

    def test_various_sigma_values(self):
        """Test EI increases with sigma for fixed delta."""
        delta = 0.5
        sigmas = [0.1, 0.5, 1.0, 2.0, 5.0]
        results = [gp_expected_improvement(delta, s) for s in sigmas]

        # EI should increase with sigma
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_various_delta_values(self):
        """Test EI increases with delta for fixed sigma."""
        sigma = 1.0
        deltas = [-2.0, -1.0, 0.0, 1.0, 2.0]
        results = [gp_expected_improvement(d, sigma) for d in deltas]

        # EI should increase with delta
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]

    def test_symmetry_property(self):
        """Test that EI behaves predictably with positive/negative deltas."""
        sigma = 1.0

        ei_positive = gp_expected_improvement(1.0, sigma)
        ei_negative = gp_expected_improvement(-1.0, sigma)

        # Positive delta should give higher EI
        assert ei_positive > ei_negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
