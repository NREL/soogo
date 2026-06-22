"""Utility functions for acquisition methods."""

# Copyright (c) 2024-2026 Alliance for Energy Innovation, LLC
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

__authors__ = ["Weslley S. Pereira", "Byron Selvage"]

import logging

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set

logger = logging.getLogger(__name__)


def weighted_score(
    sx,
    dx,
    weight: float,
    sx_min: float = 0.0,
    sx_max: float = 1.0,
    dx_max: float = 1.0,
):
    r"""Computes the weighted score from the predicted value of the surrogate
    model at a point and minimum distance from the point to the set of
    previously selected evaluation points.

    The score is

    .. math::

        w \frac{s(x)-s_{min}}{s_{max}-s_{min}} +
        (1-w) \frac{d_{max}-d(x,X)}{d_{max}},

    where :math:`w` is a weight; :math:`s(x)` is the value for the
    surrogate model on :math:`x`; :math:`d(x,X)` is the minimum distance
    between :math:`x` and the previously selected evaluation points;
    :math:`s_{min}` and :math:`s_{max}` are the minimum and maximum values of
    the surrogate model, respectively; and :math:`d_{max}` is the maximum
    distance between a candidate point and the set :math:`X` of previously
    selected evaluation points.


    In case :math:`s_{max} = s_{min}`, the score is computed as

    .. math::

        \frac{d_{max}-d(x,X)}{d_{max}}.

    :param sx: Function value(s) :math:`s(x)`.
    :param dx: Distance(s) between candidate(s) and the set X.
    :param weight: Weight :math:`w`.
    :param sx_min: Minimum value of the surrogate model.
    :param sx_max: Maximum value of the surrogate model.
    :param dx_max: Maximum distance between a candidate point and the set X.
    """
    if sx_max == sx_min:
        return (dx_max - dx) / dx_max
    else:
        return (
            weight * ((sx - sx_min) / (sx_max - sx_min))
            + (1 - weight) * (dx_max - dx) / dx_max
        )


def argmin_weighted_score(
    scaledvalue: np.ndarray,
    dist: np.ndarray,
    weight: float,
    tol: float,
) -> int:
    """Gets the index of the candidate point that minimizes the weighted score.

    The score is :math:`w f_s(x) + (1-w) (-d_s(x))`, where

    - :math:`w` is a weight.
    - :math:`f_s(x)` is the estimated value for the objective function on x,
        scaled to [0,1].
    - :math:`d_s(x)` is the minimum distance between x and the previously
        selected evaluation points, scaled to [-1,0].

    Returns -1 if there is no feasible point.

    :param scaledvalue: Function values :math:`f_s(x)` scaled to [0, 1].
    :param dist: Minimum distance between a candidate point and previously
        evaluated sampled points.
    :param weight: Weight :math:`w`.
    :param tol: Tolerance value for excluding candidates that are too close to
        current sample points.
    """
    # Scale distance values to [0,1]
    maxdist = np.max(dist)
    mindist = np.min(dist)
    if maxdist == mindist:
        scaleddist = np.ones(dist.size)
    else:
        scaleddist = (dist - mindist) / (maxdist - mindist)

    # Compute weighted score for all candidates
    score = weighted_score(scaledvalue, scaleddist, weight)

    # Assign bad values to points that are too close to already
    # evaluated/chosen points
    score[dist < tol] = np.inf

    # Return index with the best (smallest) score
    iBest = np.argmin(score)
    if score[iBest] == np.inf:
        logger.warning(
            "All candidates are too close to evaluated points; choose a better tolerance."
        )
        return -1

    return int(iBest)


def select_weighted_candidates(
    x: np.ndarray,
    distx: np.ndarray,
    fx: np.ndarray,
    n: int,
    tol: float,
    weightpattern: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively select n points from a pool of candidates
    using the weighted score criterion.

    The score on the iteration `i > 1` uses the distances to candidates
    selected in the iterations `0` to `i-1`.

    :param x: Matrix with candidate points.
    :param distx: Matrix with the distances between the candidate points and
        the m number of rows of x.
    :param fx: Vector with the estimated values for the objective function
        on the candidate points.
    :param n: Number of points to be selected for the next costly
        evaluation.
    :param tol: Tolerance value for excluding candidates that are too close to
        current sample points.
    :param weightpattern: List of weights to use cyclically for each selection.
    :return: Tuple containing (1) n-by-dim matrix with the selected
        points, and (2) n-by-(n+m) matrix with the distances between the
        n selected points and the (n+m) sampled points (m is the number
        of points that have been sampled so far).
    """
    # Compute neighbor distances
    dist = np.min(distx, axis=1)

    m = distx.shape[1]
    dim = x.shape[1]

    xselected = np.zeros((n, dim))
    distselected = np.zeros((n, m + n))

    # Scale function values to [0,1]
    if fx.ndim == 1:
        minval = np.min(fx)
        maxval = np.max(fx)
        if minval == maxval:
            scaledvalue = np.ones(fx.size)
        else:
            scaledvalue = (fx - minval) / (maxval - minval)
    elif fx.ndim == 2:
        minval = np.min(fx, axis=0)
        maxval = np.max(fx, axis=0)
        scaledvalue = np.average(
            np.where(
                maxval - minval > 0, (fx - minval) / (maxval - minval), 1
            ),
            axis=1,
        )

    selindex = argmin_weighted_score(scaledvalue, dist, weightpattern[0], tol)
    if selindex >= 0:
        xselected[0, :] = x[selindex, :]
        distselected[0, 0:m] = distx[selindex, :]
    else:
        return np.empty((0, dim)), np.empty((0, m))

    for ii in range(1, n):
        # compute distance of all candidate points to the previously selected
        # candidate point
        newDist = cdist(xselected[ii - 1, :].reshape(1, -1), x)[0]
        dist = np.minimum(dist, newDist)

        selindex = argmin_weighted_score(
            scaledvalue,
            dist,
            weightpattern[ii % len(weightpattern)],
            tol,
        )
        if selindex >= 0:
            xselected[ii, :] = x[selindex, :]
        else:
            return xselected[0:ii], distselected[0:ii, 0 : m + ii]

        distselected[ii, 0:m] = distx[selindex, :]
        for j in range(ii - 1):
            distselected[ii, m + j] = np.linalg.norm(
                xselected[ii, :] - xselected[j, :]
            )
            distselected[j, m + ii] = distselected[ii, m + j]
        distselected[ii, m + ii - 1] = newDist[selindex]
        distselected[ii - 1, m + ii] = distselected[ii, m + ii - 1]

    return xselected, distselected


class FarEnoughSampleFilter:
    """Filter candidate points that are too close to existing points.

    This utility class filters out candidates that are within a minimum
    distance threshold from already sampled points.

    :param X: Matrix of existing sample points (n x d).
    :param tol: Minimum distance threshold.
    :param approach: Method to select points when multiple candidates are close
        to each other. Options are:
        "independent_set" (default) which uses a maximum independent set
        algorithm, and "greedy" which uses a simple greedy approach.

    .. attribute:: tree

        KDTree built from the existing sample points for efficient distance
        queries.

    .. attribute:: tol

        Minimum distance threshold. Points closer than this are filtered out.

    .. attribute:: approach

        Method to select points when multiple candidates are close to each
        other.
    """

    def __init__(self, X, tol, approach="independent_set"):
        self.tree = KDTree(X)
        self.tol = tol
        self.approach = approach

    def is_far_enough(self, x):
        """Check if a point is far enough from existing samples.

        :param x: Point to check (d-dimensional vector).
        :return: True if the point is far enough, False otherwise.
        """
        dist, _ = self.tree.query(x.reshape(1, -1))
        return dist[0] >= self.tol

    def max_independent_set(self, X):
        dist = cdist(X, X)
        np.fill_diagonal(dist, np.inf)
        g = nx.Graph()
        g.add_nodes_from(range(len(X)))
        g.add_edges_from(
            [
                (i, j)
                for i in range(len(X))
                for j in range(i + 1, len(X))
                if dist[i, j] < self.tol
            ]
        )
        return maximum_independent_set(g)

    def greedy_independent_set(self, X):
        idx = []
        for i in range(len(X)):
            if all(np.linalg.norm(X[i] - X[j]) >= self.tol for j in idx):
                idx.append(i)
        return idx

    def indices(self, Xc):
        """Filter candidates based on minimum distance criterion.

        :param Xc: Matrix of candidate points (m x d).
        :return: Filtered indices for points that are far
            enough from existing samples.
        """
        # Discard points that are too close to X
        mask0 = np.array([self.is_far_enough(x) for x in Xc], dtype=bool)
        Xc0 = Xc[mask0]

        # Find the maximum independent set among the remaining points
        if self.approach == "independent_set":
            try:
                idx = self.max_independent_set(Xc0)
            except Exception as e:
                logger.warning(
                    "Error in maximum independent set computation: %s. "
                    "Falling back to a greedy approach.",
                    e,
                )
                idx = self.greedy_independent_set(Xc0)
        else:
            idx = self.greedy_independent_set(Xc0)

        # Recover original indices
        original_indices = np.where(mask0)[0]
        return original_indices[list(idx)]

    def __call__(self, Xc):
        """Filter candidates based on minimum distance criterion.

        :param Xc: Matrix of candidate points (m x d).
        :return: Filtered matrix containing only points that are far
            enough from existing samples.
        """
        return Xc[self.indices(Xc)]
