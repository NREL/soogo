"""Sampling strategies for the optimization algorithms."""

# Copyright (c) 2025 Alliance for Energy Innovation, LLC
# Copyright (C) 2014 Cornell University

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
from typing import Optional

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.stats import truncnorm
from scipy.stats.qmc import LatinHypercube, QMCEngine


class SymmetricLatinHypercube(LatinHypercube):
    """Symmetric Latin Hypercube Sampling (SLHD).

    Subclass of `scipy.stats.qmc.LatinHypercube` that implements the
    Symmetric Latin Hypercube Sampling (SLHD) algorithm.

    .. attribute:: lhs_method

        Method used to generate the Latin Hypercube samples. In this class,
        it is set to :meth:`_random_symmetric_lhs()`.
    """

    def __init__(self, *args, **kwargs):
        if "strength" in kwargs:
            raise ValueError("Strength parameter is not supported.")

        super().__init__(*args, **kwargs)

        self.lhs_method = self._random_symmetric_lhs

    def _random_symmetric_lhs(self, n: int = 1) -> np.ndarray:
        """Symmetric LHS algorithm."""
        k = n // 2

        # Compute perturbations
        if not self.scramble:
            samples: np.ndarray | float = 0.5
        else:
            samples = np.full((n, self.d), 0.5)
            samples[:k, :] = self.rng.uniform(size=(k, self.d))
            samples[n - k :, :] = 1.0 - samples[k - 1 :: -1, :]

        # Compute permutations
        perms = np.tile(np.arange(1, n + 1), (self.d, 1))  # type: ignore[arg-type]
        for i in range(1, self.d):
            perms[i, :k] = self.rng.permutation(np.arange(1, k + 1))
            for j in range(k):
                if self.rng.random() < 0.5:
                    perms[i, n - 1 - j] = n + 1 - perms[i, j]
                else:
                    perms[i, n - 1 - j] = perms[i, j]
                    perms[i, j] = n + 1 - perms[i, j]
        perms = perms.T

        samples = (perms - samples) / n

        return np.asarray(samples)


def random_sample(
    n,
    bounds,
    iindex: tuple[int, ...] = (),
    seed=None,
    **kwargs,
) -> np.ndarray:
    """Generate random samples in the given bounds.

    :param n: Number of sample points to generate.
    :param sequence bounds: List with the limits [x_min,x_max] of each
        direction x in the space.
    :param tuple iindex: Indices of the input space that are integer.
    :param seed: Seed, random number generator, or a
        `scipy.stats.qmc.QMCEngine`.

    :return: n-by-dim matrix with the sampled points.
    """
    if isinstance(seed, QMCEngine):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # Extract bounds
    l_bounds = np.array([b[0] for b in bounds])
    u_bounds = np.array([b[1] for b in bounds])
    u_bounds[list(iindex)] += 1  # For integer variables

    # Sample
    dim = len(bounds)
    if isinstance(rng, QMCEngine):
        samples01 = rng.random(n)
    else:
        samples01 = rng.random((n, dim))
    sample = l_bounds + samples01 * (u_bounds - l_bounds)

    # Floor integer variables
    sample[:, list(iindex)] = np.floor(sample[:, list(iindex)])

    return sample


def truncnorm_sample(
    n,
    bounds,
    mu,
    sigma_ref=1.0,
    iindex: tuple[int, ...] = (),
    seed=None,
    **kwargs,
) -> np.ndarray:
    """Truncated normal sample generator.

    :param n: Number of sample points to generate.
    :param sequence bounds: List with the limits [x_min,x_max] of each
        direction x in the space.
    :param mu: Point around which the sample will be generated.
    :param sigma_ref: Standard deviation of the truncated normal distribution,
        relative to a unitary interval. Default is 1.0.
    :param iindex: Indices of the input space that are integer.
    :param seed: Seed to initialize the random number generator forwarded to
        `scipy.stats.truncnorm`.
    :return: n-by-dim matrix with the sampled points.
    """
    dim = len(bounds)
    l_bounds = np.array([b[0] for b in bounds])
    u_bounds = np.array([b[1] for b in bounds])
    u_bounds[list(iindex)] += 1  # For integer variables
    sigma = sigma_ref * (u_bounds - l_bounds)

    # Distribution parameters
    loc = mu
    scale = sigma
    a = (l_bounds - loc) / scale
    b = (u_bounds - loc) / scale

    # Sample
    sample = np.asarray(
        truncnorm.rvs(a, b, loc, scale, size=(n, dim), random_state=seed)
    )

    # Floor integer variables
    sample[:, list(iindex)] = np.floor(sample[:, list(iindex)])

    return sample


def dds_sample(
    n,
    bounds,
    probability: float,
    mu,
    sigma_ref=1.0,
    iindex: tuple[int, ...] = (),
    seed=None,
    **kwargs,
) -> np.ndarray:
    """Generate a sample based on the Dynamically Dimensioned Search (DDS)
    algorithm described in [#]_.

    This algorithm generated a sample by perturbing a subset of the
    coordinates of `mu`. The set of coordinates perturbed varies for each
    sample point and is determined probabilistically. When a perturbation
    occurs, it is guided by a normal distribution with mean zero and
    standard deviation :attr:`sigma`.

    :param n: Number of sample points to generate.
    :param sequence bounds: List with the limits [x_min,x_max] of each
        direction x in the space.
    :param probability: Perturbation probability.
    :param mu: Point around which the sample will be generated.
    :param sigma_ref: Standard deviation of the truncated normal distribution,
        relative to a unitary interval. Default is 1.0.
    :param iindex: Indices of the input space that are integer.
    :param seed: Seed to initialize the random number generator.

    :return: n-by-dim matrix with the sampled points.

    References
    ----------
    .. [#] Tolson, B. A., and C. A. Shoemaker (2007), Dynamically
        dimensioned search algorithm for computationally efficient watershed
        model calibration, Water Resour. Res., 43, W01413,
        https://doi.org/10.1029/2005WR004723.
    """
    if len(iindex) == len(bounds):
        raise ValueError(
            "DDS sampling requires at least one continuous variable."
        )
    if not (0 <= probability <= 1):
        raise ValueError("Probability must be in the interval [0, 1].")
    if probability == 1:
        return truncnorm_sample(
            n,
            bounds,
            mu,
            sigma_ref=sigma_ref,
            iindex=iindex,
            seed=seed,
        )

    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Extract bounds
    dim = len(bounds)
    l_bounds = np.array([b[0] for b in bounds])
    u_bounds = np.array([b[1] for b in bounds])
    u_bounds[list(iindex)] += 1  # For integer variables
    sigma = sigma_ref * (u_bounds - l_bounds)

    # Distribution parameters
    loc = mu
    scale = sigma
    a = (l_bounds - loc) / scale
    b = (u_bounds - loc) / scale

    # Generate perturbation matrix
    ar = rng.random((n, dim)) < probability

    # Ensure at least one dimension is perturbed in each sample
    _idx = ar.sum(axis=1) == 0
    ar[_idx, rng.integers(dim, size=np.sum(_idx))] = True

    # Generate perturbations
    nperturb = max([np.sum(ar[:, i]) for i in range(dim)])
    _sample = np.asarray(
        truncnorm.rvs(
            a,
            b,
            loc,
            scale,
            size=(nperturb, dim),
            random_state=rng.integers(np.iinfo(np.int32).max).item(),
        )
    )
    _sample[:, list(iindex)] = np.floor(_sample[:, list(iindex)])

    # Assemble final sample
    sample = np.tile(mu, (n, 1))
    for i in range(dim):
        mask = ar[:, i]
        count = int(np.sum(mask))
        if count > 0:
            sample[mask, i] = _sample[:count, i]

    # Perturb all coordinates of points equal to mu
    _idx = np.all(sample == mu, axis=1)
    sample[_idx, :] = truncnorm_sample(
        np.sum(_idx),
        bounds,
        mu,
        sigma_ref=sigma_ref,
        iindex=iindex,
        seed=rng.integers(np.iinfo(np.int32).max).item(),
    )

    return sample


def dds_uniform_sample(
    n,
    bounds,
    probability: float,
    mu,
    sigma_ref=1.0,
    iindex: tuple[int, ...] = (),
    seed=None,
    **kwargs,
) -> np.ndarray:
    """Generate a sample based on the Dynamically Dimensioned Search (DDS)
    algorithm and uniform perturbations.

    `n // 2` points are generated using the DDS algorithm with normal
    perturbations, and the other half using uniform perturbations within the
    variable bounds.

    :param n: Number of sample points to generate.
    :param sequence bounds: List with the limits [x_min,x_max] of each
        direction x in the space.
    :param probability: Perturbation probability.
    :param mu: Point around which the sample will be generated.
    :param sigma_ref: Standard deviation of the truncated normal distribution,
        relative to a unitary interval. Default is 1.0.
    :param iindex: Indices of the input space that are integer.
    :param seed: Seed to initialize the random number generator.

    :return: n-by-dim matrix with the sampled points.
    """
    rng = np.random.default_rng(seed)

    x0 = dds_sample(
        n // 2,
        bounds,
        probability,
        mu,
        sigma_ref=sigma_ref,
        iindex=iindex,
        seed=rng.integers(np.iinfo(np.int32).max).item(),
    )
    x1 = random_sample(
        n - n // 2,
        bounds,
        iindex=iindex,
        seed=rng.integers(np.iinfo(np.int32).max).item(),
    )

    return np.vstack((x0, x1))


class SpaceFillingSampler:
    """Sampler that generates samples that fill gaps in the search space.

    :param dim: Dimension of the input space.
    :param max_cand: Maximum number of candidates to be generated in each
        iteration.
    :param scale: Scale factor to determine the number of candidates in each
        iteration.
    :param seed: Seed to initialize the random number generator, or random
        number generator.

    .. attribute:: dim

        Dimension of the input space.

    .. attribute:: max_cand

        Maximum number of candidates to be generated in each iteration.

    .. attribute:: scale

        Scale factor to determine the number of candidates in each iteration.

    .. attribute:: rng

        Random number generator used in sampling.
    """

    def __init__(
        self,
        max_cand: int = 10000,
        scale: float = 10.0,
        seed=None,
    ):
        self.max_cand = max_cand
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        n,
        bounds,
        current_sample: Optional[np.ndarray] = None,
        iindex: tuple[int, ...] = (),
        **kwargs,
    ) -> np.ndarray:
        """Generate a sample that aims to fill gaps in the search space.

        This algorithm generates a sample that fills gaps in the search space.
        In each iteration, it generates a pool of candidates, and selects the
        point that is farthest from current sample points to integrate the new
        sample. Algorithm proposed by [#]_.

        When no current sample points are provided, the algorithm falls back to
        generating random samples using the Latin Hypercube Sampling method.

        :param n: Number of sample points to generate.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param current_sample: Sample points already drawn.
        :param iindex: Indices of the input space that are integer.

        :return: n-by-dim matrix with the selected points.

        References
        ----------
        .. [#] Mitchell, D. P. (1991). Spectrally optimal sampling for
            distribution ray tracing. Computer Graphics, 25, 157–164.
        """
        dim = len(bounds)

        if current_sample is None or len(current_sample) == 0:
            return random_sample(
                n,
                bounds,
                iindex=iindex,
                seed=LatinHypercube(d=dim, seed=self.rng),
                **kwargs,
            )

        ncurrent = len(current_sample)
        sample = np.empty((n, dim))
        tree = KDTree(current_sample)

        # Choose candidates that are far from current sample and each other
        for i in range(n):
            npool = int(min(self.scale * (i + ncurrent), self.max_cand))

            # Pool of candidates in iteration i
            candPool = random_sample(
                npool, bounds, iindex=iindex, seed=self.rng, **kwargs
            )

            # Compute distance to current sample
            minDist, _ = tree.query(candPool)

            # Now, consider distance to candidates selected up to iteration i-1
            if i > 0:
                minDist = np.minimum(
                    minDist, np.min(cdist(candPool, sample[0:i]), axis=1)
                )

            # Choose the farthest point
            sample[i] = candPool[np.argmax(minDist)]

        return sample
