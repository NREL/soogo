"""Sampling strategies for the optimization algorithms."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC
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

__authors__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__deprecated__ = False

import numpy as np
from enum import Enum

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.stats import truncnorm


class SamplingStrategy(Enum):
    """Sampling strategy tags to be used by :meth:`get_sample()` methods that
    override :meth:`Sampler.get_sample()`.
    """

    NORMAL = 1  #: Normal distribution
    DDS = 2  #: DDS sampling. Used in the DYCORS algorithm
    UNIFORM = 3  #: Uniform distribution
    DDS_UNIFORM = 4  #: Sample half via DDS, then half via uniform distribution
    SLHD = 5  #: Symmetric Latin Hypercube Design
    MITCHEL91 = 6  #: Cover empty regions in the search space


def _slhd_permutation_matrix(m: int, d: int):
    # Generate permutation matrix P
    P = np.zeros((m, d), dtype=int)
    P[:, 0] = np.arange(m)
    if m % 2 == 0:
        k = m // 2
    else:
        k = (m - 1) // 2
        P[k, :] = k * np.ones((1, d))
    for j in range(1, d):
        P[0:k, j] = np.random.permutation(np.arange(k))

        for i in range(k):
            # Use numpy functions for better performance
            if np.random.rand() < 0.5:
                P[m - 1 - i, j] = m - 1 - P[i, j]
            else:
                P[m - 1 - i, j] = P[i, j]
                P[i, j] = m - 1 - P[i, j]

    return P


class Sampler:
    """Abstract base class for samplers.

    The main goal of a sampler is to draw samples from a d-dimensional box.
    For that, one should use :meth:`get_sample()` or the specific
    :meth:`get_[strategy]_sample()`. The former uses the information in
    :attr:`strategy` to decide which specific sampler to use.

    :param n: Number of sample points. Stored in :attr:`n`.
    :param strategy: Sampling strategy. Stored in :attr:`strategy`.

    .. attribute:: n

        Number of sample points returned by :meth:`get_[strategy]_sample()`.

    .. attribute:: strategy

        Sampling strategy that will be used by :meth:`get_sample()` and
        :meth:`get_[strategy]_sample()`.

    """

    def __init__(
        self,
        n: int,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
    ) -> None:
        self.strategy = strategy
        self.n = n

    def get_uniform_sample(
        self, bounds, *, iindex: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Generate a sample from a uniform distribution inside the bounds.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.

        :return: Matrix with a sample point per line.
        """
        dim = len(bounds)

        # Generate n sample points
        xnew = np.empty((self.n, dim))
        for i in range(dim):
            b = bounds[i]
            if i in iindex:
                xnew[:, i] = np.random.randint(b[0], b[1] + 1, self.n)
            else:
                xnew[:, i] = b[0] + np.random.rand(self.n) * (b[1] - b[0])

        return xnew

    def get_slhd_sample(
        self, bounds, *, iindex: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Creates a Symmetric Latin Hypercube Design.

        Note that, for integer variables, it may not be possible to generate
        a SLHD. In this case, the algorithm will do its best to try not to
        repeat values in the integer variables.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.

        :return: Matrix with a sample point per line.
        """
        d = len(bounds)
        m = self.n

        # Create the initial design
        X = np.empty((m, d))
        for j in range(d):
            if j not in iindex:
                delta = (bounds[j][1] - bounds[j][0]) / m
                b0 = bounds[j][0] + 0.5 * delta
                X[:, j] = [b0 + i * delta for i in range(m)]
            else:
                delta = (bounds[j][1] - bounds[j][0] + 1) / m
                b0 = bounds[j][0] + 0.5 * (delta - 1)
                X[:, j] = [round(b0 + i * delta) for i in range(m)]

        if m > 1:
            # Permute the initial design
            P = _slhd_permutation_matrix(m, d)
            for j in range(d):
                X[:, j] = X[P[:, j], j]

        return X

    def get_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), **kwargs
    ) -> np.ndarray:
        """Generate a sample based on :attr:`Sampler.strategy`.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.

        :return: Matrix with a sample point per line.
        """
        if self.strategy == SamplingStrategy.UNIFORM:
            return self.get_uniform_sample(bounds, iindex=iindex)
        elif self.strategy == SamplingStrategy.SLHD:
            return self.get_slhd_sample(bounds, iindex=iindex)
        else:
            raise ValueError("Invalid sampling strategy")


class NormalSampler(Sampler):
    """Sampler that generates sample points from a truncated normal
    distribution.

    :param sigma: Standard deviation of the truncated normal distribution,
        relative to a unitary interval. Stored in :attr:`sigma`.
    :param sigma_min: Minimum limit for the standard deviation, relative to
        a unitary interval. Stored in :attr:`sigma_min`.
    :param sigma_max: Maximum limit for the standard deviation, relative to
        a unitary interval. Stored in :attr:`sigma_max`.

    .. attribute:: sigma

        Standard deviation of the truncated normal distribution, relative to a
        unitary interval. Used by :meth:`get_normal_sample()` and
        :meth:`get_dds_sample()`.

    .. attribute:: sigma_min

        Minimum standard deviation of the truncated normal distribution,
        relative to a unitary interval.

    .. attribute:: sigma_max

        Maximum standard deviation of the truncated normal distribution,
        relative to a unitary interval.

    """

    def __init__(
        self,
        n: int,
        sigma: float,
        *,
        sigma_min: float = 0,
        sigma_max: float = 0.25,
        strategy: SamplingStrategy = SamplingStrategy.NORMAL,
    ) -> None:
        super().__init__(n, strategy=strategy)
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        assert 0 <= self.sigma_min <= self.sigma <= self.sigma_max

    def get_normal_sample(
        self,
        bounds,
        mu,
        *,
        iindex=(),
        coord=(),
    ) -> np.ndarray:
        """Generate a sample from a truncated normal distribution around a given
        point mu.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param mu: Point around which the sample will be generated.
        :param iindex: Indices of the input space that are integer.
        :param sequence coord:
            Coordinates of the input space that will vary. If (), all
            coordinates will vary.

        :return: Matrix with a sample point per line.
        """
        dim = len(bounds)
        sigma = np.array([self.sigma * (b[1] - b[0]) for b in bounds])

        # Create xnew
        xnew = np.tile(mu, (self.n, 1))
        assert xnew.shape == (self.n, dim)

        # By default all coordinates are perturbed
        if len(coord) == 0:
            coord = tuple(range(dim))

        # generate n sample points
        for i in coord:
            loc = mu[i]
            scale = sigma[i]
            if i in iindex:
                a = (bounds[i][0] - 0.5 - loc) / scale
                b = (bounds[i][1] + 0.5 - loc) / scale
                xnew[:, i] = np.round(
                    truncnorm.rvs(a, b, loc=loc, scale=scale, size=self.n)
                )
                xnew[:, i] = np.maximum(
                    bounds[i][0], np.minimum(xnew[:, i], bounds[i][1])
                )
            else:
                a = (bounds[i][0] - loc) / scale
                b = (bounds[i][1] - loc) / scale
                xnew[:, i] = truncnorm.rvs(
                    a, b, loc=loc, scale=scale, size=self.n
                )

        return xnew

    def get_dds_sample(
        self,
        bounds,
        mu,
        probability: float,
        *,
        iindex: tuple[int, ...] = (),
        coord=(),
    ) -> np.ndarray:
        """Generate a sample based on the Dynamically Dimensioned Search (DDS)
        algorithm described in [#]_.

        This algorithm generated a sample by perturbing a subset of the
        coordinates of `mu`. The set of coordinates perturbed varies for each
        sample point and is determined probabilistically. When a perturbation
        occurs, it is guided by a normal distribution with mean zero and
        standard deviation :attr:`sigma`.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param mu: Point around which the sample will be generated.
        :param probability: Perturbation probability.
        :param iindex: Indices of the input space that are integer.
        :param coord:
            Coordinates of the input space that will vary.  If (), all
            coordinates will vary.

        :return: Matrix with a sample point per line.

        References
        ----------
        .. [#] Tolson, B. A., and C. A. Shoemaker (2007), Dynamically
            dimensioned search algorithm for computationally efficient watershed
            model calibration, Water Resour. Res., 43, W01413,
            https://doi.org/10.1029/2005WR004723.
        """
        # Check if probability is valid
        if not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1")

        # Redirect if probability is 1
        if probability == 1:
            return self.get_normal_sample(
                bounds, mu, coord=coord, iindex=iindex
            )

        dim = len(bounds)
        sigma = np.array([self.sigma * (b[1] - b[0]) for b in bounds])

        # Create xnew
        xnew = np.tile(mu, (self.n, 1))
        assert xnew.shape == (self.n, dim)

        # By default all coordinates are perturbed
        if len(coord) == 0:
            coord = tuple(range(dim))

        # Generate perturbation matrix
        cdim = len(coord)
        ar = np.zeros((self.n, dim), dtype=bool)
        ar[:, coord] = np.random.rand(self.n, cdim) < probability
        for i in range(self.n):
            if not (any(ar[i, coord])):
                ar[i, np.random.randint(cdim)] = True

        # generate n sample points
        for i in coord:
            loc = mu[i]
            scale = sigma[i]
            perturbIdx = np.argwhere(ar[:, i]).flatten()
            nperturb = len(perturbIdx)
            if i in iindex:
                a = (bounds[i][0] - 0.5 - loc) / scale
                b = (bounds[i][1] + 0.5 - loc) / scale
                xnew[perturbIdx, i] = truncnorm.rvs(
                    a, b, loc=loc, scale=scale, size=nperturb
                )
                for j in perturbIdx:
                    xj = xnew[j, i]
                    if xj <= bounds[i][0]:
                        xnew[j, i] = bounds[i][0]
                    elif xj > bounds[i][0] and xj < loc:
                        xnew[j, i] = min(round(xj), loc - 1)
                    elif xj == loc:
                        sgn = +1 if np.random.randint(2) == 1 else -1
                        xnew[j, i] = loc + sgn
                    elif xj < bounds[i][1] and xj > loc:
                        xnew[j, i] = max(round(xj), loc + 1)
                    else:
                        xnew[j, i] = bounds[i][1]
            else:
                a = (bounds[i][0] - loc) / scale
                b = (bounds[i][1] - loc) / scale
                xnew[perturbIdx, i] = truncnorm.rvs(
                    a, b, loc=loc, scale=scale, size=nperturb
                )
        return xnew

    def get_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), **kwargs
    ) -> np.ndarray:
        """Generate a sample.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.
        :param mu: Point around which the sample will be generated. Used by:

            * :attr:`SamplingStrategy.NORMAL`
            * :attr:`SamplingStrategy.DDS`
            * :attr:`SamplingStrategy.DDS_UNIFORM`

        :param coord:
            Coordinates of the input space that will vary.  If (), all
            coordinates will vary. Used by:

            * :attr:`SamplingStrategy.NORMAL`
            * :attr:`SamplingStrategy.DDS`
            * :attr:`SamplingStrategy.DDS_UNIFORM`

        :param probability: Perturbation probability. Used by:

            * :attr:`SamplingStrategy.DDS`
            * :attr:`SamplingStrategy.DDS_UNIFORM`

        :return: Matrix with a sample point per line.
        """
        if self.strategy == SamplingStrategy.NORMAL:
            mu = kwargs["mu"]
            coord = kwargs["coord"] if "coord" in kwargs else ()
            return self.get_normal_sample(
                bounds, mu, iindex=iindex, coord=coord
            )
        elif self.strategy == SamplingStrategy.DDS:
            mu = kwargs["mu"]
            probability = (
                kwargs["probability"] if "probability" in kwargs else 1
            )
            coord = kwargs["coord"] if "coord" in kwargs else ()
            return self.get_dds_sample(
                bounds, mu, probability, iindex=iindex, coord=coord
            )
        elif self.strategy == SamplingStrategy.DDS_UNIFORM:
            nTotal = self.n
            mu = kwargs["mu"]
            probability = (
                kwargs["probability"] if "probability" in kwargs else 1
            )
            coord = kwargs["coord"] if "coord" in kwargs else ()

            self.n = self.n // 2
            sample0 = self.get_dds_sample(
                bounds,
                mu,
                probability,
                iindex=iindex,
                coord=coord,
            )

            self.n = nTotal - self.n
            sample1 = self.get_uniform_sample(bounds, iindex=iindex)

            self.n = nTotal
            return np.concatenate((sample0, sample1), axis=0)
        else:
            return super().get_sample(bounds, iindex=iindex)


class Mitchel91Sampler(Sampler):
    """Sampler based from [#]_ that fills gaps in the search space.

    :param maxCand: The maximum number of random candidates from which each
        sample points is selected. If None it receives the value `10*n` instead.
        Stored in :attr:`maxCand`.
    :param scale: A scaling factor proportional to the number of candidates used
        to select each sample point. Stored in :attr:`scale`.

    .. attribute:: maxCand

        The maximum number of random candidates from which each
        sample points is selected. Used by :meth:`get_mitchel91_sample()`.

    .. attribute:: scale

        Scaling factor that controls the number of candidates in the pool used
        to select a sample point. The pool has size
        `scale * [# current points]`. Used by :meth:`get_mitchel91_sample()`.

    References
    ----------
    .. [#] Mitchell, D. P. (1991). Spectrally optimal sampling for distribution
        ray tracing. Computer Graphics, 25, 157–164.
    """

    def __init__(
        self,
        n: int,
        strategy: SamplingStrategy = SamplingStrategy.MITCHEL91,
        *,
        maxCand: int = 10000,
        scale: float = 10,
    ) -> None:
        super().__init__(n, strategy)
        self.maxCand = maxCand
        self.scale = scale

    def get_mitchel91_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), current_sample=()
    ) -> np.ndarray:
        """Generate a sample that aims to fill gaps in the search space.

        This algorithm generates a sample that fills gaps in the search space.
        In each iteration, it generates a pool of candidates, and selects the
        point that is farthest from current sample points to integrate the new
        sample. This algorithm was proposed by Mitchel (1991).

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.
        :param current_sample: Sample points already drawn.

        :return: Matrix with a sample point per line.
        """
        dim = len(bounds)
        ncurrent = len(current_sample)
        cand = np.empty((self.n, dim))

        if ncurrent == 0:
            # Select the first sample randomly in the domain
            cand[0, :] = self.get_uniform_sample(bounds, iindex=iindex)[0]
            i0 = 1
        else:
            i0 = 0

        if ncurrent > 0:
            tree = KDTree(current_sample)

        # Choose candidates that are far from current sample and each other
        for i in range(i0, self.n):
            npool = int(min(self.scale * (i + ncurrent), self.maxCand))

            # Pool of candidates in iteration i
            candPool = Sampler(npool).get_uniform_sample(bounds, iindex=iindex)

            # Compute distance to current sample
            minDist = tree.query(candPool)[0] if ncurrent > 0 else 0

            # Now, consider distance to candidates selected up to iteration i-1
            if i > 0:
                minDist = np.minimum(
                    minDist, np.min(cdist(candPool, cand[0:i, :]), axis=1)
                )

            # Choose the farthest point
            cand[i, :] = candPool[np.argmax(minDist), :]

        return cand

    def get_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), **kwargs
    ) -> np.ndarray:
        """Generates a sample.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.
        :param current_sample:
            Sample points already drawn. Used by
            :attr:`SamplingStrategy.MITCHEL91`.

        :return: Matrix with a sample point per line.
        """
        if self.strategy == SamplingStrategy.MITCHEL91:
            current_sample = (
                kwargs["current_sample"] if "current_sample" in kwargs else []
            )
            return self.get_mitchel91_sample(
                bounds, iindex=iindex, current_sample=current_sample
            )
        else:
            return super().get_sample(bounds, iindex=iindex)
