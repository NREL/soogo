"""Utilities for Soogo optimize module."""

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
from scipy.spatial.distance import cdist

import numpy as np

from .result import OptimizeResult


def evaluate_and_log_point(
    fun: Callable, x: np.ndarray, out: OptimizeResult
) -> np.ndarray:
    """Evaluate an array of points and log the results in out.

    :param fun: The function to evaluate.
    :param x: 2D array of points to evaluate.
    :param out: The output object to log the results.

    :return: The function value(s) or NaN.
    """
    assert out.sample is not None and out.fsample is not None, (
        "Output object not initialized."
    )

    # Evaluate function
    n = len(x)
    try:
        y = np.asarray(fun(x))
    except Exception:
        shape_y = (n,) if out.fsample.ndim == 1 else (n, out.fsample.shape[1])
        y = np.full(shape_y, np.nan)

    # Log results
    out.sample[out.nfev : out.nfev + n] = x
    out.fsample[out.nfev : out.nfev + n] = y
    out.nfev += n

    return y


def uncertainty_score(candidates, points, fvals, k=3):
    """
    Calculate the uncertainty (distance and fitness value criterion)
    score as defined in [#]_.

    :param candidates: The candidate points to find the scores for.
    :param points: The set of already evaluated points.
    :param fvals: The set of corresponding function values.
    :param k: The number of nearest neighbors to consider in
        the uncertainty calculation. Default is 3.

    :return: The uncertainty score for each candidate point.

    References
    ----------
    .. [#] Li, F., Shen, W., Cai, X., Gao, L., & Gary Wang, G. 2020; A
        fast surrogate-assisted particle swarm optimization algorithm for
        computationally expensive problems. Applied Soft Computing, 92,
        106303. https://doi.org/10.1016/j.asoc.2020.106303
    """
    candidates = np.asarray(candidates)
    points = np.asarray(points)
    fvals = np.asarray(fvals)

    # Compute all distances
    dists = cdist(candidates, points)

    # For each candidate, get indices of k nearest points
    nearestIndices = np.argsort(dists, axis=1)[:, :k]

    # Extract distances and function values for k nearest points
    nCandidates = candidates.shape[0]
    distances = np.zeros((nCandidates, k))
    functionValues = np.zeros((nCandidates, k))

    for i in range(nCandidates):
        indices = nearestIndices[i]
        distances[i] = dists[i, indices]
        functionValues[i] = fvals[indices]

    # Calculate the mean dist and std of k nearest
    distMean = np.mean(distances, axis=1)
    sigma = np.std(functionValues, axis=1)

    # Normalize
    distMean /= np.sum(distMean)
    sigma /= np.sum(sigma)

    # Calculate scaled dist to nearest neighbor
    nearestScaled = 5 * distances[:, 0] / np.sum(distances[:, 0])

    # Calculate Sigmoid function value
    sigmoid = 1 / (1 + np.exp(-nearestScaled)) - 0.5

    # Calculate the final scores
    scores = sigmoid * (distMean + sigma)

    return scores
