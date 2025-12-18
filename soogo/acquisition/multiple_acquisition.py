"""Acquisition that uses multiple methods as needed."""

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
from ..model import RbfModel
from .utils import FarEnoughSampleFilter


class MultipleAcquisition(Acquisition):
    """Apply multiple acquisition functions sequentially.

    This acquisition function runs multiple acquisition strategies in
    sequence, filtering candidates to ensure they are far enough apart.

    :param acquisitionFuncArray: Sequence of acquisition functions to apply in
        order.
    :param kwargs: Additional arguments passed to the base Acquisition class.
    """

    def __init__(
        self,
        acquisitionFuncArray: Sequence[Acquisition],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.acquisitionFuncArray = acquisitionFuncArray

    def optimize(
        self,
        surrogateModel: RbfModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        filter = FarEnoughSampleFilter(surrogateModel.X, self.tol(bounds))
        x = np.empty((0, len(bounds)))
        for i, acq in enumerate(self.acquisitionFuncArray):
            new_x = acq.optimize(surrogateModel, bounds, n, **kwargs)
            x = filter(np.vstack((x, new_x)))
            if x.shape[0] >= n:
                return x[:n, :]

        return x
