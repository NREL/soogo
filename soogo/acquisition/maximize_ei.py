"""Expected improvement acquisition function for Gaussian Process."""

# Copyright (c) 2024-2026 Alliance for Energy Innovation, LLC

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
from scipy.linalg import cholesky, solve_triangular
from typing import Optional
import logging
import functools

from pymoo.optimize import minimize as pymoo_minimize

from .base import Acquisition
from ..model import GaussianProcess
from ..sampling import random_sample
from ..integrations.pymoo import PymooProblem
from .utils import FarEnoughSampleFilter

logger = logging.getLogger(__name__)


def _negative_ei(surrogate, best_value, x):
    """Negative expected improvement (for minimization).

    :param surrogate: Surrogate model.
    :param best_value: Best objective value so far.
    :param x: Point to evaluate.
    :return: Negative expected improvement at x.
    """
    return -surrogate.expected_improvement(x, best_value)


class MaximizeEI(Acquisition):
    """Acquisition by maximizing the expected improvement (EI) of a
    :class:`.GaussianProcess`.

    First, runs a global optimizer to find a point ``xs`` that maximizes EI.
    If successful and ``n == 1``, returns ``xs``. Otherwise, builds a candidate
    pool using :attr:`sampler` (optionally including ``xs`` and the surrogate
    minimizer) and selects points that maximize EI. When
    :attr:`avoid_clusters` is ``True``, it penalizes candidates too close to the
    already selected ones, inspired by [#]_.

    :param sampler: Space-filling candidate generator. Defaults to
        :class:`soogo.sampling.SpaceFillingSampler`.
    :param int pool_size: Number of candidates generated per call.
    :param avoid_clusters: Whether to avoid clustering within the selected
        batch.
    :param seed: Seed for random number generator.

    .. attribute:: sampler

        Space-filling candidate generator.

    .. attribute:: pool_size

        Number of candidates generated per :meth:`optimize()` call.

    .. attribute:: avoid_clusters

        When ``True``, discourages points close to those already selected.

    .. attribute:: rng

        Random number generator.

    References
    ----------
    .. [#] Che Y, Müller J, Cheng C. Dispersion-enhanced sequential batch
        sampling for adaptive contour estimation. Qual Reliab Eng Int. 2024;
        40: 131–144. https://doi.org/10.1002/qre.3245
    """

    def __init__(
        self,
        pool_size: int = Acquisition.DEFAULT_N_MAX_EVALS_OPTIMIZER,
        avoid_clusters: bool = True,
        n_max_evals_optimizer: Optional[int] = None,
        seed=None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_max_evals_optimizer=n_max_evals_optimizer or pool_size, **kwargs
        )
        self.pool_size = pool_size
        self.avoid_clusters = avoid_clusters
        self.rng = np.random.default_rng(seed)

    def optimize(
        self,
        surrogateModel: GaussianProcess,
        bounds,
        n: int = 1,
        ybest=None,
        exclusion_set: Optional[np.ndarray] = None,
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
        :param ybest: Best objective value so far. If ``None``, approximately
            minimize the surrogate and use its value.
        :param exclusion_set: Known points, if any, in addition to the ones
            used to train the surrogate.
        """
        # TODO: Extend this method to work with mixed-integer problems
        if len(surrogateModel.iindex) > 0:
            raise ValueError(
                "MaximizeEI acquisition only works for continuous problems."
            )

        dim = len(bounds)

        # Report unused kwargs
        super().report_unused_optimize_kwargs(kwargs)

        if n == 0:
            return np.empty((0, dim))

        xbest = None
        if ybest is None:
            # Compute an estimate for ybest using the surrogate.
            problem = PymooProblem(surrogateModel, bounds)
            res = pymoo_minimize(
                problem,
                self.optimizer,
                seed=self.rng.integers(np.iinfo(np.int32).max).item(),
                verbose=False,
            )
            if res.X is not None and res.F is not None:
                xbest = res.X
                ybest = res.F[0]
            else:
                logger.warning(
                    "Surrogate model minimization failed; "
                    "using best training point."
                )
                idx = surrogateModel.Y.argmin()
                xbest = surrogateModel.X[idx]
                ybest = surrogateModel.Y[idx]

        # Use the point that maximizes the EI
        problem = PymooProblem(
            functools.partial(_negative_ei, surrogateModel, ybest),
            bounds,
        )
        res = pymoo_minimize(
            problem,
            self.optimizer,
            seed=self.rng.integers(np.iinfo(np.int32).max).item(),
            verbose=False,
        )
        if res.X is not None:
            xs = res.X

        # Returns xs if n == 1
        # print(f"MaximizeEI selected point with EI = {-res.F[0]}")
        # print(f"At location: x = {res.X}")
        # print(f"Success: {res.success}")
        if n == 1:
            return np.asarray([xs])

        # Generate the complete pool of candidates
        x = random_sample(self.pool_size, bounds, seed=self.rng)
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

            iBest[j] = np.argmax(score)
            eiCand[iBest[j]] = 0.0  # Remove this candidate expectancy

        exclusion_set = (
            np.vstack((exclusion_set, surrogateModel.X))
            if exclusion_set is not None
            else surrogateModel.X
        )
        return FarEnoughSampleFilter(exclusion_set, self.tol(bounds))(
            x[iBest, :]
        )
