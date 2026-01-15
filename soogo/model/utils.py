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

from scipy.stats.qmc import QMCEngine
import numpy as np

from ..sampling import random_sample
from .base import Surrogate


def create_initial_design(
    surrogate: Surrogate,
    bounds,
    mineval: int,
    maxeval: int,
    n_trials=100,
    seed=None,
):
    """Create an initial design for a surrogate model.

    :param surrogate: Surrogate model.
    :param bounds: Bounds of the design space.
    :param mineval: Minimum number of evaluations.
    :param maxeval: Maximum number of evaluations.
    :param n_trials: Number of trials to create a valid initial design.
    :param seed: Seed, random number generator, or a
        `scipy.stats.qmc.QMCEngine`.
    :return: Initial design sample or None if a valid design cannot be created.
    """
    dim = len(bounds)  # Dimension of the problem
    m_for_surrogate = surrogate.min_design_space_size(
        dim
    )  # Smallest sample for a valid surrogate
    iindex = surrogate.iindex  # Integer design variables

    # Initialize random number generator
    if isinstance(seed, QMCEngine):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # Create a new sample with SLHD
    m = min(maxeval, max(mineval, 2 * m_for_surrogate))
    sample = random_sample(m, bounds, iindex=iindex, seed=rng)
    if m >= m_for_surrogate:
        count = 0
        while not surrogate.check_initial_design(sample):
            sample = random_sample(m, bounds, iindex=iindex, seed=rng)
            count += 1
            if count > n_trials:
                return None

    return sample
