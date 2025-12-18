"""Radial Basis Kernel Functions."""

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

from abc import abstractmethod, ABC
import sys

# Autograd imports
from autograd import numpy as anp
from autograd import grad


class RadialBasisFunction(ABC):
    """Base class for radial basis functions used in RBF models."""

    @abstractmethod
    def __call__(self, r):
        """Evaluate the radial basis function.

        :param r: Radial distance value.
        :return: Function value at distance r.
        """
        pass

    def __init__(self):
        """Initialize the radial basis function and set up autograd
        functions.
        """
        self._grad = grad(self.__call__)
        self._hess = grad(self._grad)

    def grad(self, r):
        """Gradient of the radial function.

        :param r: Radial distance value.
        :return: Gradient value at distance r.
        """
        return self._grad(r)

    def hess(self, r):
        """Hessian of the radial function.

        :param r: Radial distance value.
        :return: Hessian value at distance r.
        """
        return self._hess(r)

    def grad_over_r(self, r):
        """Gradient of the radial function divided by r.

        :param r: Radial distance value.
        :return: Gradient divided by r.
        """
        return self._grad(r) / r

    @staticmethod
    def polynomial_tail_order() -> int:
        """Return the order of the polynomial tail.

        :return: Order of the polynomial tail, or -1 if no polynomial tail
            is present.
        """
        return -1


class LinearRadialBasisFunction(RadialBasisFunction):
    def __call__(self, r):
        return -r

    @staticmethod
    def polynomial_tail_order() -> int:
        return 0


class CubicRadialBasisFunction(RadialBasisFunction):
    def __call__(self, r):
        return r**3

    def grad_over_r(self, r):
        return 3 * r

    @staticmethod
    def polynomial_tail_order() -> int:
        return 1


class ThinPlateRadialBasisFunction(RadialBasisFunction):
    def __init__(self, safe_r=sys.float_info.min):
        """Initialize the thin plate radial basis function.

        :param safe_r: A small value to avoid numerical issues with log(0).
                       Defaults to the smallest positive floating point number.
        """
        super().__init__()
        self.safe_r = safe_r

    def __call__(self, r):
        _r = anp.where(r == 0, 1, r)
        return _r**2 * anp.log(_r)

    def grad_over_r(self, r):
        _r = anp.where(r == 0, self.safe_r, r)
        return 2 * anp.log(_r) + 1

    @staticmethod
    def polynomial_tail_order() -> int:
        return 1
