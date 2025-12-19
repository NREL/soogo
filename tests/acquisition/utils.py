"""Common utilities and mock classes for acquisition function tests."""

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
from typing import Union, Tuple, Optional

from soogo.model import Surrogate


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
