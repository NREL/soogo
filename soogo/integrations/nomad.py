"""Problem definitions for interfacing with PyNomad."""

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

from collections.abc import Callable

import numpy as np

from ..optimize.utils import OptimizeResult, evaluate_and_log_point


class NomadProblem:
    """
    Function wrapper that handles the input and output for optimization with
    NOMAD via the PyNomad library.

    :param func: The objective function to evaluate
    :param out: OptimizeResult object to store results
    """

    def __init__(self, func: Callable, out: OptimizeResult):
        self.func = func
        self.out = out
        self._xHistory = []
        self._fHistory = []

    def __call__(self, x):
        """Wrapper for the objective function to be used with NOMAD.

        :param x: NOMAD point object
        :return: 1 if evaluation successful, 0 otherwise
        """
        point = np.array([x.get_coord(i) for i in range(x.size())])
        self._xHistory.append(point)

        f = evaluate_and_log_point(self.func, point.reshape(1, -1), self.out)[
            0
        ]
        self._fHistory.append(f)

        if np.isfinite(f):
            # Set NOMAD objective function value
            x.setBBO(str(f).encode("UTF-8"))
            return 1
        else:
            return 0

    def get_f_history(self):
        """Get the history of function values."""
        return self._fHistory

    def get_x_history(self):
        """Get the history of input points."""
        return self._xHistory

    def reset(self):
        """Reset the optimization history."""
        self._xHistory = []
        self._fHistory = []
