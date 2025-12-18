"""Condition module for optimization termination criteria.

This module defines various conditions that can be used to determine when
an optimization process should terminate. It includes conditions based on
successful improvements, robustness of conditions, and more.
"""

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

from abc import ABC, abstractmethod
from collections import deque
from typing import Optional
import numpy as np

from .optimize.result import OptimizeResult
from .model import Surrogate


class TerminationCondition(ABC):
    """Base class for termination conditions.

    This class defines the interface for conditions that can be used to
    determine when an optimization process should terminate.
    """

    @abstractmethod
    def is_met(self) -> bool:
        """Check if the termination condition is met.

        :return: True if the condition is met, False otherwise.
        """
        pass

    @abstractmethod
    def update(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> None:
        """Update the condition based on the optimization result and model.

        :param out: The optimization result containing the current state.
        :param model: The surrogate model used in the optimization, if any.
        :return: True if the condition is met, False otherwise.
        """
        pass

    def reset(self, **kwargs) -> None:
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
        self._is_met = False

    def is_met(self) -> bool:
        return self._is_met

    def update(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> None:
        if out.nfev == 0:
            # No function evaluations, cannot update condition
            return

        assert out.nobj == 1, (
            "Expected a single objective function value, but got multiple objectives."
        )

        # Get the new best value from the optimization result
        new_best_value = (
            out.fx.flatten()[0].item()
            if isinstance(out.fx, np.ndarray)
            else out.fx
        )
        assert isinstance(new_best_value, float), (
            "Expected out.fx to be a float, but got a different type."
        )

        # Compute the relative improvement
        value_improvement = self.lowest_value - new_best_value

        # Update the condition state
        self._is_met = value_improvement <= self.threshold * self.value_range

        # Update the knowledge about the optimization problem
        maxf = np.atleast_2d(out.fsample.T)[0, 0 : out.nfev].max()
        minf = np.atleast_2d(out.fsample.T)[0, 0 : out.nfev].min()
        if model is not None and len(model.Y) > 0:
            maxf = max(maxf, model.Y.max())
            minf = min(minf, model.Y.min())
        self.value_range = max(self.value_range, maxf - minf)
        self.lowest_value = min(self.lowest_value, minf)

    def reset(self, keep_data_knowledge: bool = False, **kwargs) -> None:
        self._is_met = False
        if not keep_data_knowledge:
            self.value_range = 0.0
            self.lowest_value = float("inf")


class RobustCondition(TerminationCondition):
    """Termination criterion that makes another termination criterion robust."""

    def __init__(self, termination: TerminationCondition, period=30) -> None:
        self.termination = termination
        self.history = deque(maxlen=period)

    def is_met(self) -> bool:
        assert isinstance(self.history.maxlen, int), (
            "History maxlen must be set."
        )

        if len(self.history) < self.history.maxlen:
            return False
        return all(self.history)

    def update(
        self, out: OptimizeResult, model: Optional[Surrogate] = None
    ) -> None:
        self.termination.update(out, model)
        self.history.append(self.termination.is_met())

    def reset(self, **kwargs) -> None:
        self.history.clear()
        self.termination.reset(**kwargs)


class IterateNTimes(TerminationCondition):
    """
    Termination condition that is met after a specified number of
    iterations.

    :param nTimes: Number of iterations after which the condition is met.
    """

    def __init__(self, nTimes: int = 1) -> None:
        self.nTimes = nTimes
        self.iterationCount = 0

    def is_met(self) -> bool:
        return self.iterationCount >= self.nTimes

    def update(self, *args, **kwargs) -> None:
        self.iterationCount += 1

    def reset(self, **kwargs) -> None:
        self.iterationCount = 0
