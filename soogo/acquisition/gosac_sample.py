"""GOSAC acquisition function for constrained optimization."""

import numpy as np
from scipy.spatial import KDTree

from pymoo.optimize import minimize as pymoo_minimize

from soogo.acquisition.base import Acquisition
from soogo.acquisition.maximize_distance import MaximizeDistance
from soogo.model.base import Surrogate
from soogo.problem import PymooProblem


class GosacSample(Acquisition):
    """GOSAC acquisition function as described in [#]_.

    Minimize the objective function with surrogate constraints. If a feasible
    solution is found and is different from previous sample points, return it as
    the new sample. Otherwise, the new sample is the point that is farthest from
    previously selected sample points.

    This acquisition function is only able to acquire 1 point at a time.

    :param fun: Objective function. Stored in :attr:`fun`.

    .. attribute:: fun

        Objective function.

    References
    ----------
    .. [#] Juliane Mueller and Joshua D. Woodbury. GOSAC: global optimization
        with surrogate approximation of constraints.
        J Glob Optim, 69:117-136, 2017.
        https://doi.org/10.1007/s10898-017-0496-y
    """

    def __init__(self, fun, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fun = fun

    def optimize(
        self,
        surrogateModel: Surrogate,
        bounds,
        n: int = 1,
        constraintTransform: callable = None,
        **kwargs,
    ) -> np.ndarray:
        """Acquire 1 point.

        :param surrogateModel: Multi-target surrogate model for the constraints.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Unused.
        :param constraintTransform: Function to transform the constraints.
            If not provided, use the identity function. The optimizer, pymoo,
            expects that negative and zero values are feasible, and positive
            values are infeasible.
        :return: 1-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        gdim = surrogateModel.ntarget

        iindex = surrogateModel.iindex
        optimizer = self.optimizer if len(iindex) == 0 else self.mi_optimizer

        assert n == 1

        if constraintTransform is None:

            def constraintTransform(x):
                return x

        def transformedConstraint(x):
            surrogateOutput = surrogateModel(x)
            return constraintTransform(surrogateOutput)

        # Create KDTree with previously evaluated points
        tree = KDTree(surrogateModel.X)
        atol = self.tol(bounds)

        cheapProblem = PymooProblem(
            self.fun,
            bounds,
            iindex,
            gfunc=transformedConstraint,
            n_ieq_constr=gdim,
        )
        res = pymoo_minimize(
            cheapProblem,
            optimizer,
            seed=surrogateModel.ntrain,
            verbose=False,
        )

        # If either no feasible solution was found or the solution found is too
        # close to already sampled points, we then
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of sample points.
        isGoodCandidate = True
        if res.X is None:
            isGoodCandidate = False
        else:
            xnew = np.asarray([[res.X[i] for i in range(dim)]])
            if tree.query(xnew)[0] < atol:
                isGoodCandidate = False

        if not isGoodCandidate:
            maximizeDistance = MaximizeDistance(rtol=self.rtol)

            xnew = maximizeDistance.optimize(surrogateModel, bounds, n=1)

            assert len(xnew) == 1

        return xnew
