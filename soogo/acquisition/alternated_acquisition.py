"""Alternated acquisition function that cycles through multiple functions."""

import numpy as np
from typing import Sequence

from .base import Acquisition
from ..model import RbfModel, Surrogate
from ..optimize.result import OptimizeResult


class AlternatedAcquisition(Acquisition):
    """
    Alternated acquisition function that cycles through a list of acquisition
    functions.

    The current acquisition function moves to the next in the list when the
    current one's termination criterion is met. To progress through the
    acquisition functions, the `update` method must be called after each
    optimization step. This provides the function with the current optimization
    state, allowing it to determine if the termination condition has been
    satisfied.

    :param acquisitionFuncArray: List of acquisition functions to be used in
        sequence.
    """

    def __init__(
        self,
        acquisitionFuncArray: Sequence[Acquisition],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.acquisitionFuncArray = acquisitionFuncArray
        self.idx = 0

    def optimize(
        self,
        surrogateModel: RbfModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        return self.acquisitionFuncArray[self.idx].optimize(
            surrogateModel, bounds, n, **kwargs
        )

    def update(self, out: OptimizeResult, model: Surrogate) -> None:
        self.acquisitionFuncArray[self.idx].update(out, model)

        # Alternate if the current acquisition function's termination is met
        if self.acquisitionFuncArray[self.idx].has_converged():
            self.idx = (self.idx + 1) % len(self.acquisitionFuncArray)
            self.acquisitionFuncArray[self.idx].termination.reset()
