"""Acquisition that uses multiple methods as needed."""

import numpy as np
from typing import Sequence

from .base import Acquisition
from ..model import RbfModel
from .utils import FarEnoughSampleFilter


class MultipleAcquisition(Acquisition):
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
