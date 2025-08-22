"""Utility functions for Soogo."""

# Copyright (c) 2025 Alliance for Sustainable Energy, LLC

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
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = ["Weslley S. Pereira"]
__deprecated__ = False

from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
    from .optimize_result import OptimizeResult


def find_pareto_front(fx, iStart: int = 0) -> list:
    """Find the Pareto front given a set of points in the target space.

    :param fx: List with n points in the m-dimensional target space.
    :param iStart: Points from 0 to iStart - 1 are already known to be in the
        Pareto front.
    :return: Indices of the points in the Pareto front.
    """
    pareto = [True] * len(fx)
    for i in range(iStart, len(fx)):
        for j in range(i):
            if pareto[j]:
                if all(fx[i] <= fx[j]) and any(fx[i] < fx[j]):
                    # x[i] dominates x[j]
                    pareto[j] = False
                elif all(fx[j] <= fx[i]) and any(fx[j] < fx[i]):
                    # x[j] dominates x[i]
                    # No need to continue checking, otherwise the previous
                    # iteration was not a balid Pareto front
                    pareto[i] = False
                    break
    return [i for i in range(len(fx)) if pareto[i]]


def gp_expected_improvement(delta, sigma):
    """Expected Improvement function for a distribution from [#]_.

    :param delta: Difference :math:`f^*_n - \\mu_n(x)`, where :math:`f^*_n` is
        the current best function value and :math:`\\mu_n(x)` is the expected
        value for :math:`f(x)`.
    :param sigma: The standard deviation :math:`\\sigma_n(x)`.

    References
    ----------
    .. [#] Donald R. Jones, Matthias Schonlau, and William J. Welch. Efficient
        global optimization of expensive black-box functions. Journal of Global
        Optimization, 13(4):455–492, 1998.
    """
    from scipy.stats import norm

    return delta * norm.cdf(delta / sigma) + sigma * norm.pdf(delta / sigma)


def evaluate_and_log_point(fun: callable, x: np.ndarray, out: "OptimizeResult"):
    """
    Evaluate a point or array of points and log the results. If the function
    errors or the result is invalid (NaN or Inf), the output is logged as NaN.
    If the function value is less than the current best, the current best (
    out.x & out.fx) is updated. Each point is evaluated individually. This
    function only supports single-objective functions.

    :param fun: The function to evaluate.
    :param x: The point(s) to evaluate. Can be a 1D array (single point) or
              2D array (multiple points).
    :param out: The output object to log the results.

    :return: The function value(s) or NaN. Returns a scalar for single point,
             array for multiple points.
    """
    x = np.atleast_2d(x)
    results = []

    for point in x:
        try:
            y = fun(point)
            if hasattr(y, '__len__'):
                y = y[0]
            if np.isnan(y) or np.isinf(y):
                y = np.nan
        except Exception:
            y = np.nan

        out.sample[out.nfev, :] = point
        out.fsample[out.nfev] = y
        out.nfev += 1
        results.append(y)

        if not np.isnan(y) and (out.fx is None or y < out.fx):
            out.x = point
            out.fx = y

    return results[0] if len(results) == 1 else np.array(results)
