"""Expected improvement acquisition function for Gaussian Process."""

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
from scipy.optimize import differential_evolution
from scipy.linalg import cholesky, solve_triangular

from .base import Acquisition
from ..model import GaussianProcess
from ..sampling import Sampler, Mitchel91Sampler


class MaximizeEI(Acquisition):
    """Acquisition by maximization of the expected improvement of a Gaussian
    Process.

    It starts by running a
    global optimization algorithm to find a point `xs` that maximizes the EI. If
    this point is found and the sample size is 1, return this point. Else,
    creates a pool of candidates using :attr:`sampler` and `xs`. From this pool,
    select the set of points with that maximize the expected improvement. If
    :attr:`avoid_clusters` is `True` avoid points that are too close to already
    chosen ones inspired in the strategy from [#]_. Mind that the latter
    strategy can slow down considerably the acquisition process, although is
    advisable for a sample of good quality.

    :param sampler: Sampler to generate candidate points. Stored in
        :attr:`sampler`.
    :param avoid_clusters: When `True`, use a strategy that avoids points too
        close to already chosen ones. Stored in :attr:`avoid_clusters`.

    .. attribute:: sampler

        Sampler to generate candidate points.

    .. attribute:: avoid_clusters

        When `True`, use a strategy that avoids points too close to already
        chosen ones.

    References
    ----------
    .. [#] Che Y, Müller J, Cheng C. Dispersion-enhanced sequential batch
        sampling for adaptive contour estimation. Qual Reliab Eng Int. 2024;
        40: 131–144. https://doi.org/10.1002/qre.3245
    """

    def __init__(
        self, sampler=None, avoid_clusters: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.sampler = Sampler(0) if sampler is None else sampler
        self.avoid_clusters = avoid_clusters

    def optimize(
        self,
        surrogateModel: GaussianProcess,
        bounds,
        n: int = 1,
        *,
        ybest=None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Run a global optimization procedure to try to find a point that has the
        highest expected improvement for the Gaussian Process.
        Moreover, if `ybest` isn't provided, run a global optimization procedure
        to find the minimum value of the surrogate model. Use the minimum point
        as a candidate for this acquisition.

        This implementation only works for continuous design variables.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param ybest: Best point so far. If not provided, find the minimum value
            for the surrogate. Use it as a possible candidate.
        """
        # TODO: Extend this method to work with mixed-integer problems
        assert len(surrogateModel.iindex) == 0

        if n == 0:
            return np.empty((0, len(bounds)))

        xbest = None
        if ybest is None:
            # Compute an estimate for ybest using the surrogate.
            res = differential_evolution(
                lambda x: surrogateModel(np.asarray([x])), bounds
            )
            ybest = res.fun
            if res.success:
                xbest = res.x

        # Use the point that maximizes the EI
        res = differential_evolution(
            lambda x: -surrogateModel.expected_improvement(
                np.asarray([x]), ybest
            ),
            bounds,
        )
        xs = res.x if res.success else None

        # Returns xs if n == 1
        if res.success and n == 1:
            return np.asarray([xs])

        # Generate the complete pool of candidates
        if isinstance(self.sampler, Mitchel91Sampler):
            current_sample = surrogateModel.X
            if xs is not None:
                current_sample = np.concatenate((current_sample, [xs]), axis=0)
            if xbest is not None:
                current_sample = np.concatenate(
                    (current_sample, [xbest]), axis=0
                )
            x = self.sampler.get_mitchel91_sample(
                bounds, current_sample=current_sample
            )
        else:
            x = self.sampler.get_sample(bounds)

        if xs is not None:
            x = np.concatenate(([xs], x), axis=0)
        if xbest is not None:
            x = np.concatenate((x, [xbest]), axis=0)
        nCand = len(x)

        # Create EI and kernel matrices
        eiCand = surrogateModel.expected_improvement(x, ybest)

        # If there is no need to avoid clustering return the maximum of EI
        if not self.avoid_clusters or n == 1:
            return x[np.flip(np.argsort(eiCand)[-n:]), :]
        # Otherwise see what follows...

        # Rescale EI to [0,1] and create the kernel matrix with all candidates
        if eiCand.max() > eiCand.min():
            eiCand = (eiCand - eiCand.min()) / (eiCand.max() - eiCand.min())
        else:
            eiCand = np.ones_like(eiCand)
        Kss = surrogateModel.eval_kernel(x)

        # Score to be maximized and vector with the indexes of the candidates
        # chosen.
        score = np.zeros(nCand)
        iBest = np.empty(n, dtype=int)

        # First iteration
        j = 0
        for i in range(nCand):
            Ksi = Kss[:, i]
            Kii = Kss[i, i]
            score[i] = ((np.dot(Ksi, Ksi) / Kii) / nCand) * eiCand[i]
        iBest[j] = np.argmax(score)
        eiCand[iBest[j]] = 0.0  # Remove this candidate expectancy

        # Remaining iterations
        for j in range(1, n):
            currentBatch = iBest[0:j]

            Ksb = Kss[:, currentBatch]
            Kbb = Ksb[currentBatch, :]

            # Cholesky factorization using points in the current batch
            Lfactor = cholesky(Kbb, lower=True)

            # Solve linear systems for KbbInvKbs
            LInvKbs = solve_triangular(Lfactor, Ksb.T, lower=True)
            KbbInvKbs = solve_triangular(
                Lfactor, LInvKbs, lower=True, trans="T"
            )

            # Compute the b part of the score
            scoreb = np.sum(np.multiply(Ksb, KbbInvKbs.T))

            # Reserve memory to avoid excessive dynamic allocations
            aux0 = np.empty(nCand)
            aux1 = np.empty((j, nCand))

            # If the remaining candidates are not expected to improve the
            # solution, choose sample based on the distance criterion only.
            if np.max(eiCand) == 0.0:
                eiCand[:] = 1.0

            # Compute the final score
            for i in range(nCand):
                if i in currentBatch:
                    score[i] = 0
                else:
                    # Compute the square of the diagonal term of the
                    # updated Cholesky factorization
                    li = LInvKbs[:, i]
                    d2 = Kss[i, i] - np.dot(li, li)

                    # Solve the linear system Kii*aux = Ksi.T
                    Ksi = Kss[:, i]
                    aux0[:] = (Ksi.T - LInvKbs.T @ li) / d2
                    aux1[:] = LInvKbs - np.outer(li, aux0)
                    aux1[:] = solve_triangular(
                        Lfactor, aux1, lower=True, trans="T", overwrite_b=True
                    )

                    # Local score computation
                    scorei = np.sum(np.multiply(Ksb, aux1.T)) + np.dot(
                        Ksi, aux0
                    )

                    # Final score
                    score[i] = ((scorei - scoreb) / nCand) * eiCand[i]
                    # assert(score[i] >= 0)

            iBest[j] = np.argmax(score)
            eiCand[iBest[j]] = 0.0  # Remove this candidate expectancy

        return x[iBest, :]
