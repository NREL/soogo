"""Utilities for Soogo optimize module."""

from collections.abc import Callable
from scipy.spatial.distance import cdist

import numpy as np

from .result import OptimizeResult


def evaluate_and_log_point(fun: Callable, x: np.ndarray, out: OptimizeResult):
    """
    Evaluate a point or array of points and log the results. If the function
    errors or the result is invalid (NaN or Inf), the output is logged as NaN.
    If the function value is less than the current best, the current best (
    out.x, out.fx) is updated. Provided points are evaluated as a batch. This
    function only supports single-objective functions.

    :param fun: The function to evaluate.
    :param x: The point(s) to evaluate. Can be a 1D array (single point) or
              2D array (multiple points).
    :param out: The output object to log the results.

    :return: The function value(s) or NaN. Returns a scalar for single point,
             array for multiple points.
    """
    x = np.atleast_2d(x)

    try:
        results = fun(x)
        results = np.atleast_1d(results)
    except Exception:
        results = np.full(x.shape[0], np.nan)

    # Process each result individually
    for i, y in enumerate(results):
        if hasattr(y, "__len__") and len(y) > 0:
            y = y[0]
        if np.isnan(y) or np.isinf(y):
            y = np.nan
        results[i] = y

        out.sample[out.nfev, :] = x[i]
        out.fsample[out.nfev] = y
        out.nfev += 1

        if not np.isnan(y) and (out.fx is None or y < out.fx):
            out.x = x[i]
            out.fx = y

    return results[0] if len(results) == 1 else results


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
    .. [#] Li, F., Shen, W., Cai, X., Gao, L., & Gary Wang, G. 2020; A fast
        surrogate-assisted particle swarm optimization algorithm for computationally
        expensive problems. Applied Soft Computing, 92, 106303.
        https://doi.org/10.1016/j.asoc.2020.106303
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
