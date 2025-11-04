"""Maximize distance acquisition function."""

import numpy as np
from typing import Optional
from scipy.spatial import KDTree

from pymoo.optimize import minimize as pymoo_minimize

from soogo.acquisition.base import Acquisition
from soogo.model.base import Surrogate
from soogo.problem import PymooProblem


class MaximizeDistance(Acquisition):
    """
    Maximizing distance acquisition function as described in [#]_.

    This acquisition function is used to find new sample points that maximize
    the minimum distance to previously sampled points.

    References
    ----------
    .. [#] Juliane Müller and Marcus Day. Surrogate Optimization of
        Computationally Expensive Black-Box Problems with Hidden Constraints.
        INFORMS Journal on Computing, 31(4):689-702, 2019.
        https://doi.org/10.1287/ijoc.2018.0864
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        points: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Acquire n points that maximize the minimum distance to previously
        sampled points.

        :param surrogateModel: The surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param points: Points to consider for distance maximization. If None,
            use all previously sampled points in the surrogate model.
        """
        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        # Calculate tolerance using the tol function
        atol = self.tol(bounds)

        selectedPoints = []
        if points is None:
            currentPoints = surrogateModel.X.copy()
        else:
            currentPoints = points.copy()

        tree = KDTree(currentPoints)
        for i in range(n):
            problem = PymooProblem(
                lambda x: -tree.query(x)[0],
                bounds,
                iindex,
            )

            res = pymoo_minimize(
                problem,
                optimizer,
                seed=surrogateModel.ntrain + i + 1,
                verbose=False,
            )
            if res.X is not None:
                new_point = np.array([res.X[j] for j in range(len(bounds))])

                # Check if the new point is far enough from existing points
                distance_to_existing = tree.query(new_point.reshape(1, -1))[0]
                if distance_to_existing >= atol:
                    selectedPoints.append(new_point)
                    currentPoints = np.vstack([currentPoints, new_point])

                    # Update the KDTree with the new point
                    tree = KDTree(currentPoints)

        return (
            np.array(selectedPoints)
            if selectedPoints
            else np.empty((0, len(bounds)))
        )
