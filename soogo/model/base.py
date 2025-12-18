"""Surrogate model abstract base class."""

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

from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union, Tuple


class Surrogate(ABC):
    """Abstract base class for surrogate models.

    This class provides a common interface for different types of surrogate
    models such as Gaussian Processes and Radial Basis Function models.
    All surrogate models should inherit from this class and implement the
    abstract methods.

    The interface is designed to support optimization algorithms that use
    surrogate models for approximating expensive objective functions.
    """

    @abstractmethod
    def reserve(self, n: int, dim: int, ntarget: int = 1) -> None:
        """Reserve space for training data.

        :param n: Number of training points to reserve.
        :param dim: Dimension of the input space.
        :param ntarget: Dimension of the target space.
        """
        pass

    @abstractmethod
    def __call__(
        self, x: np.ndarray, i: int = -1, **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Evaluate the surrogate model at given points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param i: Index of the target dimension to evaluate. If -1,
            evaluate all.
        :return: Model predictions at the input points. Some models may return
            additional information such as uncertainty estimates.
        """
        pass

    @abstractmethod
    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update the surrogate model with new training data.

        :param x: m-by-d matrix with m new point coordinates in a
            d-dimensional space.
        :param y: m-by-n matrix with m new function values and n target
            dimensions.
        """
        pass

    @property
    @abstractmethod
    def X(self) -> np.ndarray:
        """Get the training input points.

        :return: m-by-d matrix with m training points in a d-dimensional space.
        """
        pass

    @property
    @abstractmethod
    def Y(self) -> np.ndarray:
        """Get the training output values.

        :return: m-by-n matrix with m training points and n target dimensions.
        """
        pass

    @property
    def ntrain(self) -> int:
        """Get the number of training points (m).

        :return: Number of training points.
        """
        return len(self.X)

    @property
    def dim(self) -> int:
        """Get the dimension (d) of the input space.

        :return: Dimension of the input space.
        """
        if self.X.ndim == 2:
            return self.X.shape[1]
        else:
            assert self.X.size == 0
            return 0

    @property
    def ntarget(self) -> int:
        """Get the dimension (n) of the target space.

        :return: Dimension of the target space.
        """
        return self.Y.shape[1] if self.Y.ndim > 1 else 1

    @property
    @abstractmethod
    def iindex(self) -> tuple[int, ...]:
        """Return the indices of integer variables in the feature space.

        :return: Tuple with indices of integer variables.
        """
        pass

    @abstractmethod
    def min_design_space_size(self, dim: int) -> int:
        """Return the minimum design space size for a given space dimension.

        :param dim: Dimension of the space.
        :return: Minimum number of points needed to build the surrogate model.
        """
        pass

    @abstractmethod
    def check_initial_design(self, sample: np.ndarray) -> bool:
        """Check if the sample is able to generate a valid surrogate.

        :param sample: m-by-d matrix with m training points in a
            d-dimensional space.
        :return: True if the sample is valid for building the surrogate model.
        """
        pass

    @abstractmethod
    def eval_kernel(
        self, x: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Evaluate the kernel function at a pair (x,y).

        The structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters.

        :param x: First entry in the tuple (x,y).
        :param y: Second entry in the tuple (x,y). If None, use x.
        :return: Kernel evaluation result.
        """
        pass

    @abstractmethod
    def reset_data(self) -> None:
        """Reset the surrogate model training data.

        This method is used to clear the training data of the surrogate model,
        allowing it to be reused for a new optimization run.
        """
        pass
