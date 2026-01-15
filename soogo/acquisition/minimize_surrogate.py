"""Minimize surrogate acquisition function using multi-level single-linkage."""

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
from math import log
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.special import gamma
from scipy.optimize import minimize

from .base import Acquisition
from ..model import Surrogate
from ..sampling import random_sample
from .utils import FarEnoughSampleFilter


class _SurrogateMinimizerWrapper:
    """Helper class to wrap surrogate model for minimization.

    :param surrogateModel: Surrogate model to be minimized.
    :param x: Base point in the space.
    :param i: Index(s) that will be optimized.
    """

    def __init__(self, surrogateModel, x, i):
        self.surrogateModel = surrogateModel
        self.x = x
        self.i = i

    def fun(self, xi):
        """Evaluate surrogate model at modified point x[i] = xi."""
        _x = self.x.copy()
        _x[self.i] = xi
        return self.surrogateModel(_x)

    def jac(self, xi):
        """Evaluate surrogate model Jacobian at modified point x[i] = xi.

        Only the components corresponding to indices i are returned.
        """
        _x = self.x.copy()
        _x[self.i] = xi
        return self.surrogateModel.jac(_x)[self.i]


class MinimizeSurrogate(Acquisition):
    """Obtain sample points that are local minima of the surrogate model.

    This implementation is based on the one of MISO-MS used in the paper [#]_.
    The original method, Multi-level Single-Linkage, was described in [#]_.
    In each iteration, the algorithm generates a pool of candidates and select
    the best candidates (lowest predicted value) that are far enough from each
    other. The number of candidates chosen as well as the distance threshold
    vary with each iteration. The hypothesis is that the successful candidates
    each belong to a different region in the space, which may contain a local
    minimum, and those regions cover the whole search space. In the sequence,
    the algorithm runs multiple local minimization procedures using the
    successful candidates as local guesses. The results of the minimization are
    collected for the new sample.

    :param pool_size: Number of uniform candidates generated each iteration.
    :param rtol: Minimum distance between a candidate and already selected
        points, relative to the domain size. Default is ``1e-3``.
    :param seed: Seed or random number generator.

    .. attribute:: pool_size

        Number of uniform candidates generated each iteration.

    .. attribute:: rng

        Random engine used to draw candidates.

    References
    ----------
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    .. [#] Rinnooy Kan, A.H.G., Timmer, G.T. Stochastic global optimization
        methods part II: Multi level methods. Mathematical Programming 39, 57–78
        (1987). https://doi.org/10.1007/BF02592071
    """

    def __init__(
        self, pool_size: int, rtol: float = 1e-3, seed=None, **kwargs
    ) -> None:
        super().__init__(rtol=rtol, **kwargs)
        self.pool_size = pool_size
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points based on MISO-MS from Müller (2016).

        The critical distance is the same used in the seminal work from
        Rinnooy Kan and Timmer (1987).

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Max number of points to be acquired.
        :param kwargs: Additional keyword arguments (unused).
        :return: n-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        volumeBounds = np.prod([b[1] - b[0] for b in bounds])

        # Get index and bounds of the continuous variables
        cindex = [i for i in range(dim) if i not in surrogateModel.iindex]
        cbounds = [bounds[i] for i in cindex]

        # Local parameters
        remevals = 1000 * dim  # maximum number of RBF evaluations
        maxiter = 10  # maximum number of iterations to find local minima.
        sigma = 4.0  # default value for computing crit distance
        critdist = (
            (gamma(1 + (dim / 2)) * volumeBounds * sigma) ** (1 / dim)
        ) / np.sqrt(np.pi)  # critical distance when 2 points are equal
        has_jac = hasattr(surrogateModel, "jac") and callable(
            getattr(surrogateModel, "jac")
        )

        # Local space to store information
        candidates = np.empty((self.pool_size * maxiter, dim))
        distCandidates = np.empty(
            (self.pool_size * maxiter, self.pool_size * maxiter)
        )
        fcand = np.empty(self.pool_size * maxiter)
        startpID = np.full((self.pool_size * maxiter,), False)
        selected = np.empty((n, dim))

        # Create a KDTree with the training data points
        filter = FarEnoughSampleFilter(surrogateModel.X, self.tol(bounds))

        iter = 0
        k = 0
        while iter < maxiter and k < n and remevals > 0:
            iStart = iter * self.pool_size
            iEnd = (iter + 1) * self.pool_size

            # if computational budget is exhausted, then return
            if remevals <= iEnd - iStart:
                break

            # Critical distance for the i-th iteration
            critdistiter = critdist * (log(iEnd) / iEnd) ** (1 / dim)

            # Consider only the best points to start local minimization
            counterLocalStart = iEnd // maxiter

            # Choose candidate points uniformly in the search space
            candidates[iStart:iEnd, :] = random_sample(
                self.pool_size,
                bounds,
                iindex=surrogateModel.iindex,
                seed=self.rng,
            )

            # Compute the distance between the candidate points
            distCandidates[iStart:iEnd, iStart:iEnd] = cdist(
                candidates[iStart:iEnd, :], candidates[iStart:iEnd, :]
            )
            distCandidates[0:iStart, iStart:iEnd] = cdist(
                candidates[0:iStart, :], candidates[iStart:iEnd, :]
            )
            distCandidates[iStart:iEnd, 0:iStart] = distCandidates[
                0:iStart, iStart:iEnd
            ].T

            # Evaluate the surrogate model on the candidate points and sort them
            fcand[iStart:iEnd] = surrogateModel(candidates[iStart:iEnd, :])
            ids = np.argsort(fcand[0:iEnd])
            remevals -= iEnd - iStart

            # Select the best points that are not too close to each other
            chosenIds = np.zeros((counterLocalStart,), dtype=int)
            nSelected = 0
            for i in range(counterLocalStart):
                if not startpID[ids[i]]:
                    select = True
                    for j in range(i):
                        if distCandidates[ids[i], ids[j]] <= critdistiter:
                            select = False
                            break
                    if select:
                        chosenIds[nSelected] = ids[i]
                        nSelected += 1
                        startpID[ids[i]] = True

            # Evolve the best points to find the local minima
            for i in range(nSelected):
                xi = candidates[chosenIds[i], :]

                wrapper = _SurrogateMinimizerWrapper(
                    surrogateModel, xi, cindex
                )
                res = minimize(
                    wrapper.fun,
                    xi[cindex],
                    method="L-BFGS-B",
                    jac=wrapper.jac if has_jac else None,
                    bounds=cbounds,
                    options={
                        "maxfun": remevals,
                        "maxiter": max(2, round(remevals / 20)),
                        "disp": False,
                    },
                )
                remevals -= res.nfev
                xi[cindex] = res.x

                if len(filter(xi.reshape(1, -1))) > 0:
                    selected[k, :] = xi
                    k += 1
                    if k == n:
                        break
                    else:
                        filter.tree = KDTree(np.vstack([filter.tree.data, xi]))

                if remevals <= 0:
                    break

            e_nlocmin = (
                k * (counterLocalStart - 1) / (counterLocalStart - k - 2)
            )
            e_domain = (
                (counterLocalStart - k - 1)
                * (counterLocalStart + k)
                / (counterLocalStart * (counterLocalStart - 1))
            )
            if (e_nlocmin - k < 0.5) and (e_domain >= 0.995):
                break

            iter += 1

        return selected[0:k, :]
