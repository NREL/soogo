"""Test the RBF model."""

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
import sys
import pytest
from soogo.model import (
    CubicRadialBasisFunction,
    LinearRadialBasisFunction,
    MedianLpfFilter,
    RbfModel,
    ThinPlateRadialBasisFunction,
)


class TestRbfModel:
    rbf_model = RbfModel()

    def test_phi(self):
        self.rbf_model.rbf = LinearRadialBasisFunction()
        r_linear = np.array([1.0, 2.0, 3.0])
        result_linear = self.rbf_model.rbf(r_linear)
        expected_linear = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(np.array(result_linear), expected_linear)
        assert self.rbf_model.rbf(4.0) == -4.0

        self.rbf_model.rbf = CubicRadialBasisFunction()
        r_cubic = np.array([1.0, 2.0, 3.0])
        result_cubic = self.rbf_model.rbf(r_cubic)
        expected_cubic = np.array([1.0, 8.0, 27.0])
        np.testing.assert_array_equal(np.array(result_cubic), expected_cubic)
        assert self.rbf_model.rbf(4.0) == 64.0

        self.rbf_model.rbf = ThinPlateRadialBasisFunction()
        r_thinplate = np.array([1.0, 2.0, 3.0])
        result_thinplate = self.rbf_model.rbf(r_thinplate)
        expected_thinplate = np.array([0.0, 2.77258872, 9.8875106])
        np.testing.assert_allclose(
            np.array(result_thinplate), expected_thinplate
        )
        assert self.rbf_model.rbf(4.0) == (
            4 * 4 * np.log(4 + sys.float_info.min)
        )

    def test_dim(self):
        assert self.rbf_model.dim == 0

        self.rbf_model.reserve(0, 3)
        assert self.rbf_model.dim == 3

        self.rbf_model.reserve(1, 4)
        assert self.rbf_model.dim == 4

        # The dimension should not change
        with pytest.raises(Exception):
            self.rbf_model.reserve(1, 2)


def test_median_lpf():
    f = MedianLpfFilter()

    x = [1, 2, 3, 4, 5]
    medianx = np.median(x)
    y = [min(x[i], medianx) for i in range(len(x))]
    assert np.array_equal(f(x), y)

    x = [7, 5, 29, 2, 8]
    medianx = np.median(x)
    y = [min(x[i], medianx) for i in range(len(x))]
    assert np.array_equal(f(x), y)
