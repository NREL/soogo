"""Acquisition functions for surrogate optimization."""

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
from abc import ABC, abstractmethod
from typing import Optional

# Pymoo imports
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating

# Local imports
from ..model import Surrogate
from ..integrations.pymoo import ListDuplicateElimination
from ..termination import TerminationCondition
from ..optimize.result import OptimizeResult


class Acquisition(ABC):
    """Base class for acquisition functions.

    This an abstract class. Subclasses must implement the method
    :meth:`optimize()`.

    Acquisition functions are strategies to propose new sample points to a
    surrogate. The acquisition functions here are modeled as objects with the
    goals of adding states to the learning process. Moreover, this design
    enables the definition of the :meth:`optimize()` method with a similar API
    when we compare different acquisition strategies.

    :param optimizer: Continuous optimizer to be used for the acquisition
        function. Default is Differential Evolution (DE) from pymoo.
    :param mi_optimizer: Mixed-integer optimizer to be used for the acquisition
        function. Default is Genetic Algorithm (MixedVariableGA) from pymoo.
    :param rtol: Minimum distance between a candidate point and the
        previously selected points relative to the domain size. Default is 1e-6.

    .. attribute:: optimizer

        Continuous optimizer to be used for the acquisition function. This is
        used in :meth:`optimize()`.

    .. attribute:: mi_optimizer

        Mixed-integer optimizer to be used for the acquisition function. This is
        used in :meth:`optimize()`.

    .. attribute:: rtol

        Minimum distance between a candidate point and the previously selected
        points.  This figures out as a constraint in the optimization problem
        solved in :meth:`optimize()`.
    """

    def __init__(
        self,
        optimizer=None,
        mi_optimizer=None,
        rtol: float = 1e-6,
        termination: Optional[TerminationCondition] = None,
    ) -> None:
        self.optimizer = DE() if optimizer is None else optimizer
        self.mi_optimizer = (
            MixedVariableGA(
                eliminate_duplicates=ListDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=ListDuplicateElimination()
                ),
            )
            if mi_optimizer is None
            else mi_optimizer
        )
        self.rtol = rtol
        self.termination = termination

    @abstractmethod
    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Propose a maximum of n new sample points to improve the surrogate.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of requested points. Mind that the number of points
            returned may be smaller than n, depending on the implementation.
        :return: m-by-dim matrix with the selected points, where m <= n.
        """
        pass

    def tol(self, bounds) -> float:
        """Compute tolerance used to eliminate points that are too close to
        previously selected ones.

        The tolerance value is based on :attr:`rtol` and the diameter of the
        largest d-dimensional cube that can be inscribed whithin the bounds.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        """
        return (
            self.rtol
            * np.sqrt(len(bounds))
            * np.min([abs(b[1] - b[0]) for b in bounds])
        )

    def has_converged(self) -> bool:
        """Check if the acquisition function has converged.

        This method is used to check if the acquisition function has converged
        based on a termination criterion. The default implementation always
        returns False.
        """
        if self.termination is not None:
            return self.termination.is_met()
        else:
            return False

    def update(self, out: OptimizeResult, model: Surrogate) -> None:
        """Update the acquisition function knowledge about the optimization
        process.

        :param out: Current optimization result containing evaluation history.
        :param model: Updated surrogate model.
        """
        if self.termination is not None:
            self.termination.update(out, model)
