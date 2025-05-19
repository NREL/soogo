"""Test the Gaussian Process model and helpers."""

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
from soogo.gp import GaussianProcess


@pytest.mark.parametrize("n", (10, 100))
@pytest.mark.parametrize("copy_X_train", (True, False))
def test_xtrain(n: int, copy_X_train: bool):
    gp = GaussianProcess(copy_X_train=copy_X_train)

    X0 = np.random.rand(n, 3)
    y = np.random.rand(n)
    gp.update(X0, y)
    assert np.isclose(X0, gp.xtrain()).all()

    X1 = np.random.rand(n, 3)
    y = np.random.rand(n)
    gp.update(X1, y)
    assert np.isclose(np.concatenate((X0, X1), axis=0), gp.xtrain()).all()
