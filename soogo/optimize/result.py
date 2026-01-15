"""Optimization result dataclass for soogo optimizers."""

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

from typing import Optional
import numpy as np

from ..model import Surrogate, create_initial_design
from ..utils import find_pareto_front


class OptimizeResult:
    """Optimization result for the global optimizers provided by this
    package.

    .. attribute:: x

        Best sample point found so far.

    .. attribute:: fx

        Best objective function value.

    .. attribute:: nit

        Number of active learning iterations.

    .. attribute:: nfev

        Number of function evaluations taken.

    .. attribute:: sample

        n-by-dim matrix with all n samples.

    .. attribute:: fsample

        Vector with all n objective values.

    .. attribute:: nobj

        Number of objective function targets.

    """

    def __init__(self) -> None:
        self.x = None
        self.fx = None
        self.nit = 0
        self.nfev = 0
        self.sample = np.array([])
        self.fsample = np.array([])
        self.nobj = 1

    def init(
        self,
        fun,
        bounds,
        mineval: int,
        maxeval: int,
        surrogateModel: Surrogate,
        seed=None,
    ) -> None:
        """Initialize :attr:`nfev` and :attr:`sample` and :attr:`fsample` with
        data about the optimization that is starting.

        This routine calls the objective function :attr:`nfev` times.

        By default, all targets are considered to be used in the objective. If
        that is not the case, set `nobj` after calling this function.

        :param fun: The objective function to be minimized.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param mineval: Minimum number of function evaluations to build the
            surrogate model.
        :param maxeval: Maximum number of function evaluations.
        :param surrogateModel: Surrogate model to be used.

            This is only used to initialize :attr:`nobj` when the surrogate
            model is empty.

        :param seed: Seed, random number generator, or a
            `scipy.stats.qmc.QMCEngine`.
        """
        dim = len(bounds)  # Dimension of the problem
        assert dim > 0

        # Initialize sample array in this object
        self.sample = np.empty((maxeval, dim))
        self.sample[:] = np.nan

        # If the surrogate is empty and no initial sample was given
        if surrogateModel.ntrain == 0:
            # Create initial design
            sample = create_initial_design(
                surrogateModel, bounds, mineval, maxeval, seed=seed
            )
            if sample is None:
                raise RuntimeError("Cannot create valid initial design")

            # Compute f(sample)
            fsample = np.array(fun(sample))

            # Initialize nfev, sample, nobj and fsample
            self.nfev = len(sample)
            self.sample[0 : self.nfev] = sample
            self.nobj = fsample.shape[1] if fsample.ndim > 1 else 1
            self.fsample = np.empty(
                maxeval if self.nobj <= 1 else (maxeval, self.nobj)
            )
            self.fsample[0 : self.nfev] = fsample
            self.fsample[self.nfev :] = np.nan
        else:
            # Initialize nobj and fsample
            self.nobj = surrogateModel.ntarget
            self.fsample = np.empty(
                maxeval if self.nobj <= 1 else (maxeval, self.nobj)
            )
            self.fsample[:] = np.nan

    def init_best_values(
        self, surrogateModel: Optional[Surrogate] = None
    ) -> None:
        """Initialize :attr:`x` and :attr:`fx` based on the best values obtained
        so far.

        :param surrogateModel: Surrogate model.
        """
        # Initialize self.x and self.fx
        assert self.sample is not None
        assert self.fsample is not None
        m = self.nfev

        if surrogateModel is not None and surrogateModel.ntrain > 0:
            combined_x = np.concatenate(
                (self.sample[0:m], surrogateModel.X), axis=0
            )
            if self.fsample.ndim == 1:
                combined_y = np.concatenate(
                    (self.fsample[0:m], surrogateModel.Y), axis=0
                )
            else:
                nrows = surrogateModel.ntrain
                ncols = self.fsample.shape[1]
                combined_y = np.concatenate(
                    (self.fsample[0:m], np.empty((nrows, ncols))), axis=0
                )
                combined_y[m:, 0 : surrogateModel.ntarget] = surrogateModel.Y
        else:
            combined_x = self.sample[0:m]
            combined_y = self.fsample[0:m]

        if self.nobj == 1:
            if combined_y.ndim == 1:
                iBest = np.argmin(combined_y).item()
                self.fx = combined_y[iBest].item()
            else:
                iBest = np.argmin(combined_y[:, 0]).item()
                self.fx = combined_y[iBest].copy()

            self.x = combined_x[iBest].copy()
        else:
            iPareto = find_pareto_front(combined_y[:, 0 : self.nobj])
            self.x = combined_x[iPareto].copy()
            self.fx = combined_y[iPareto].copy()
