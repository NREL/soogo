"""GOSAC acquisition function for constrained optimization."""

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

import numpy as np

from pymoo.optimize import minimize as pymoo_minimize

from .base import Acquisition
from ..model import Surrogate
from ..integrations.pymoo import PymooProblem
from .utils import FarEnoughSampleFilter


class GosacSample(Acquisition):
    """GOSAC acquisition function as described in [#]_.

    Minimize the objective function with surrogate constraints. If a feasible
    solution is found and is different from previous sample points, return it as
    the new sample. Otherwise, the new sample is the point that is farthest from
    previously selected sample points.

    This acquisition function is only able to acquire 1 point at a time.

    :param fun: Objective function. Stored in :attr:`fun`.

    .. attribute:: fun

        Objective function.

    References
    ----------
    .. [#] Juliane Mueller and Joshua D. Woodbury. GOSAC: global optimization
        with surrogate approximation of constraints.
        J Glob Optim, 69:117-136, 2017.
        https://doi.org/10.1007/s10898-017-0496-y
    """

    def __init__(self, fun, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fun = fun

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        constr_fun=None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire 1 point.

        :param surrogateModel: Multi-target surrogate model for the constraints.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Unused.
        :param constr_fun: Constraint function to be applied to surrogate model
            predictions. If none is provided, use the surrogate model as
            the constraint function.
        :param kwargs: Additional keyword arguments (unused).
        :return: 1-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        gdim = surrogateModel.ntarget

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        cheapProblem = PymooProblem(
            self.fun,
            bounds,
            iindex,
            gfunc=surrogateModel if constr_fun is None else constr_fun,
            n_ieq_constr=gdim,
        )
        res = pymoo_minimize(
            cheapProblem,
            optimizer,
            seed=surrogateModel.ntrain,
            verbose=False,
        )
        if res.X is not None:
            xnew = np.asarray([[res.X[i] for i in range(dim)]])
            return FarEnoughSampleFilter(surrogateModel.X, self.tol(bounds))(
                xnew
            )
        else:
            return np.empty((0, dim))
