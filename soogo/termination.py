"""Condition module for optimization termination criteria.

This module defines various conditions that can be used to determine when
an optimization process should terminate. It includes conditions based on
successful improvements, robustness of conditions, and more.
"""

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
from typing import Optional
import numpy as np

from .optimize_result import OptimizeResult
from .model.base import Surrogate


class TerminationCondition(ABC):
    """Base class for termination conditions.

    This class defines the interface for conditions that can be used to
    determine when an optimization process should terminate.

    :param out: The optimization result containing the current state.
    :param model: The surrogate model used in the optimization, if any.
    :return: True if the condition is met, False otherwise.
    """

    @abstractmethod
    def is_met(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        """Check if the condition is met without updating internal state.

        :param out: The optimization result containing the current state.
        :param model: The surrogate model used in the optimization, if any.
        :return: True if the condition is met, False otherwise.
        """
        pass

    def is_not_met(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        """Check if the condition is not met."""
        return not self.is_met(out, model)

    def update(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        """Update the condition based on the optimization result and model.

        :param out: The optimization result containing the current state.
        :param model: The surrogate model used in the optimization, if any.
        :return: True if the condition is met, False otherwise.
        """
        return self.is_met(out, model)

    def reset(self) -> None:
        """Reset the internal state of the condition."""
        return None


class UnsuccessfulImprovement(TerminationCondition):
    """Condition that checks for unsuccessful improvements.

    The condition is met when the relative improvement in the best objective
    function value is less than a specified threshold.

    :param threshold: The relative improvement threshold to determine
        when the condition is met.

    .. attribute:: threshold

        The relative improvement threshold for the condition.

    .. attribute:: value_range

        The range of objective function values known so far, used to
        normalize the improvement check.

    .. attribute:: lowest_value

        The lowest objective function value found so far in the optimization.

    """

    def __init__(self, threshold=0.001) -> None:
        self.threshold = threshold
        self.value_range = 0.0
        self.lowest_value = float("inf")

    def is_met(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        assert out.nobj == 1, (
            "Expected a single objective function value, but got multiple objectives."
        )
        new_best_value = (
            out.fx[0].item() if isinstance(out.fx, np.ndarray) else out.fx
        )

        value_improvement = self.lowest_value - new_best_value
        return value_improvement < self.threshold * self.value_range

    def update(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        ret = self.is_met(out, model)

        maxf = max(
            np.atleast_2d(out.fsample.T)[0, 0 : out.nfev].max(), model.Y.max()
        )
        minf = min(
            np.atleast_2d(out.fsample.T)[0, 0 : out.nfev].min(), model.Y.min()
        )
        self.value_range = max(self.value_range, maxf - minf)
        self.lowest_value = min(self.lowest_value, minf)

        return ret

    def reset(self) -> None:
        self.value_range = 0.0
        self.lowest_value = float("inf")
        return None


class RobustCondition(TerminationCondition):
    """Termination criterion that makes another termination criterion robust."""

    def __init__(self, termination: TerminationCondition, period=30) -> None:
        self.termination = termination
        self.history = deque(maxlen=period)

    def is_met(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        history = self.history.copy()
        history.append(self.termination.is_met(out, model))

        assert isinstance(history.maxlen, int), (
            "History maxlen must be an integer."
        )
        if len(history) < history.maxlen:
            return False
        return all(history)

    def update(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> bool:
        self.history.append(self.termination.update(out, model))

        assert isinstance(self.history.maxlen, int), (
            "History maxlen must be an integer."
        )
        if len(self.history) < self.history.maxlen:
            return False
        return all(self.history)

    def reset(self) -> None:
        self.history.clear()
        self.termination.reset()
        return None
