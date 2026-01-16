"""Acquisition that uses multiple methods."""

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
from typing import Sequence

from .base import Acquisition
from ..model import Surrogate
from .utils import FarEnoughSampleFilter


class MultipleAcquisition(Acquisition):
    """Use multiple acquisition functions.

    This acquisition function runs multiple acquisition strategies, filtering
    candidates to ensure they are far enough apart.

    :param acquisitionFuncArray: Sequence of acquisition functions to apply in
        order.
    :param strategy: Strategy to acquire points. Defaults to fallback behavior.

    .. attribute:: strategy

        Strategy to acquire points. Currently supported strategies are:

        - ``None`` (default): Fallback behavior. Each acquisition function is
          called in sequence with the full number of points to acquire.
        - ``"equal"``: Equal distribution. The number of points to acquire is
          divided equally among the acquisition functions.
    """

    def __init__(
        self,
        acquisitionFuncArray: Sequence[Acquisition],
        strategy=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.acquisitionFuncArray = acquisitionFuncArray
        self.strategy = strategy

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire at most n points using multiple acquisition functions.

        :param surrogateModel: The surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :return: k-by-dim matrix with the selected points, where k <= n.
        """
        dim = len(bounds)
        filter = FarEnoughSampleFilter(np.empty((0, dim)), self.tol(bounds))
        x = np.empty((0, dim))
        for i, acq in enumerate(self.acquisitionFuncArray):
            if self.strategy == "equal":
                n_to_acquire = (n - x.shape[0]) // (
                    len(self.acquisitionFuncArray) - i
                )
                new_x = acq.optimize(
                    surrogateModel, bounds, n=n_to_acquire, **kwargs
                )
            else:
                new_x = acq.optimize(surrogateModel, bounds, n=n, **kwargs)

            x = filter(np.vstack((x, new_x)))

            if x.shape[0] >= n:
                return x[:n, :]

        return x
