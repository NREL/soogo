"""Termination criteria for optimization processes."""

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

from abc import ABC, abstractmethod
from collections import deque
from math import isinf
from typing import Optional
import numpy as np

from .optimize_result import OptimizeResult
from .model.base import Surrogate


class Termination(ABC):
    """Base class for termination criteria."""

    @abstractmethod
    def has_terminated(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        """Check if the termination criterion has been met."""
        pass


class NoTermination(Termination):
    """Termination criterion that never terminates."""

    def has_terminated(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        """Always returns False, indicating that termination has not occurred."""
        return False


class SuccessfulImprovementTermination(Termination):
    """Termination criterion that checks for successful improvements."""

    def __init__(self, threshold=0.01) -> None:
        self.threshold = threshold
        self.worst_value = float("inf")
        self.best_value = float("inf")

    def _update(self, fx) -> None:
        """Inform the termination criterion about the current state."""
        maxf = max(fx)
        minf = min(fx)
        self.worst_value = (
            maxf if isinf(self.worst_value) else max(self.worst_value, maxf)
        )
        self.best_value = (
            minf if isinf(self.best_value) else min(self.best_value, minf)
        )

    def has_terminated(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        """Check if the improvement is below the threshold."""
        assert out.nobj == 1, (
            "Expected a single objective function value, but got multiple objectives."
        )
        current_best_value = (
            out.fx[0].item() if isinstance(out.fx, np.ndarray) else out.fx
        )

        if self.worst_value == self.best_value:
            improvement = self.best_value - current_best_value
            ret = improvement <= 0
        else:
            improvement = (self.best_value - current_best_value) / (
                self.worst_value - self.best_value
            )
            ret = improvement < self.threshold

        self._update(np.atleast_2d(out.fsample)[:, 0])
        if model is not None:
            self._update(model.Y)

        return ret


class RobustTermination(Termination):
    """Termination criterion that makes another termination criterion robust."""

    def __init__(self, termination: Termination, period=30) -> None:
        self.termination = termination
        self.history = deque(maxlen=period)

    def has_terminated(self, *args, **kwargs) -> bool:
        """Check if the base termination criterion has been met."""
        self.history.append(self.termination.has_terminated(*args, **kwargs))
        assert isinstance(self.history.maxlen, int), (
            "History maxlen must be an integer."
        )
        if len(self.history) < self.history.maxlen:
            return False
        return all(self.history)
