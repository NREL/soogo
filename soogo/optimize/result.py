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

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np

from ..sampling import Sampler
from ..model import Surrogate
from ..utils import find_pareto_front


@dataclass
class OptimizeResult:
    """Optimization result for the global optimizers provided by this
    package."""

    x: Optional[np.ndarray] = None  #: Best sample point found so far
    fx: Union[float, np.ndarray, None] = None  #: Best objective function value
    nit: int = 0  #: Number of active learning iterations
    nfev: int = 0  #: Number of function evaluations taken
    sample: Optional[np.ndarray] = None  #: n-by-dim matrix with all n samples
    fsample: Optional[np.ndarray] = None  #: Vector with all n objective values
    nobj: int = 1  #: Number of objective function targets

    def init(
        self,
        fun,
        bounds,
        mineval: int,
        maxeval: int,
        surrogateModel: Surrogate,
        ntarget: int = 1,
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
        :param ntarget: Number of target dimensions. Default is 1.
        """
        dim = len(bounds)  # Dimension of the problem
        assert dim > 0

        # Local variables
        m0 = surrogateModel.ntrain  # Number of initial sample points
        m_for_surrogate = surrogateModel.min_design_space_size(
            dim
        )  # Smallest sample for a valid surrogate
        iindex = surrogateModel.iindex  # Integer design variables

        # Initialize sample array in this object
        self.sample = np.empty((maxeval, dim))
        self.sample[:] = np.nan

        # If the surrogate is empty and no initial sample was given
        m = 0
        if m0 == 0:
            # Create a new sample with SLHD
            m = min(maxeval, max(mineval, 2 * m_for_surrogate))
            self.sample[0:m] = Sampler(m).get_slhd_sample(
                bounds, iindex=iindex
            )
            if m >= 2 * m_for_surrogate:
                count = 0
                while not surrogateModel.check_initial_design(
                    self.sample[0:m]
                ):
                    self.sample[0:m] = Sampler(m).get_slhd_sample(
                        bounds, iindex=iindex
                    )
                    count += 1
                    if count > 100:
                        raise RuntimeError(
                            "Cannot create valid initial design"
                        )

        # Initialize fsample and nobj
        if m0 == 0:
            # Compute f(sample)
            fsample = np.array(fun(self.sample[0:m]))
            self.nfev += m
            self.nobj = fsample.shape[1] if fsample.ndim > 1 else 1

            # Store the function values
            self.fsample = np.empty(
                maxeval if self.nobj <= 1 else (maxeval, self.nobj)
            )
            self.fsample[0:m] = fsample
        else:
            self.nobj = max(ntarget, surrogateModel.ntarget)
            self.fsample = np.empty(
                maxeval if self.nobj <= 1 else (maxeval, self.nobj)
            )
        self.fsample[m:] = np.nan

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
