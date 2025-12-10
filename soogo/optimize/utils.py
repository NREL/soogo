"""Utilities for Soogo optimize module."""

import numpy as np
from collections.abc import Callable
from ..utils import OptimizeResult


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
