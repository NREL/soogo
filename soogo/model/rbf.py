"""Radial Basis Function model."""

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

from typing import Optional, Union, Tuple
import warnings
import numpy as np
from math import comb

# Autograd imports
from autograd import grad, hessian

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.linalg import solve, ldl, solve_triangular

# Local imports
from .base import Surrogate
from .rbf_kernel import RadialBasisFunction, CubicRadialBasisFunction


class RbfFilter:
    """Base filter class for the RBF target training set. Trivial identity
    filter."""

    def __call__(self, x) -> np.ndarray:
        return x


class MedianLpfFilter(RbfFilter):
    """Filter values by replacing large function values by the median of all.

    This strategy was proposed by [#]_ based on results from [#]_. Use this
    strategy to reduce oscillations of the interpolator, especially if the range
    target function is large. This filter may reduce the quality of the
    approximation by the surrogate.

    References
    ----------

    .. [#] Gutmann, HM. A Radial Basis Function Method for Global Optimization.
        Journal of Global Optimization 19, 201–227 (2001).
        https://doi.org/10.1023/A:1011255519438

    .. [#] Björkman, M., Holmström, K. Global Optimization of Costly Nonconvex
        Functions Using Radial Basis Functions. Optimization and Engineering 1,
        373–397 (2000). https://doi.org/10.1023/A:1011584207202
    """

    def __call__(self, x) -> np.ndarray:
        return np.where(x > np.median(x), np.median(x), x)


class RbfModel(Surrogate):
    r"""Radial Basis Function model.

    .. math::

        f(x)    = \sum_{i=1}^{m} \beta_i \phi(\|x - x_i\|)
                + \sum_{i=1}^{n} \beta_{m+i} p_i(x),

    where:

    - :math:`m` is the number of sampled points.
    - :math:`x_i` are the sampled points.
    - :math:`\beta_i` are the coefficients of the RBF model.
    - :math:`\phi` is the kernel function.
    - :math:`p_i` are the basis functions of the polynomial tail.
    - :math:`n` is the dimension of the polynomial tail.

    This implementation focuses on quick successive updates of the model, which
    is essential for the good performance of active learning processes.

    :param kernel: Kernel function :math:`\phi` used in the RBF model.
    :param iindex: Indices of integer variables in the feature space.
    :param filter: Filter to be used in the target (image) space.

    .. attribute:: kernel

        Kernel function :math:`\phi` used in the RBF model.

    .. attribute:: filter

        Filter to be used in the target (image) space.

    """

    def __init__(
        self,
        kernel: RadialBasisFunction = CubicRadialBasisFunction(),
        iindex: tuple[int, ...] = (),
        filter: Optional[RbfFilter] = None,
    ):
        self.rbf = kernel
        self._iindex = iindex
        self.filter = RbfFilter() if filter is None else filter

        self._m = 0
        self._x = np.array([])
        self._fx = np.array([])
        self._coef = np.array([])
        self._PHI = np.array([])

        self.shift = 0.0
        self.scale = 1.0

    def reserve(self, maxeval: int, dim: int, ntarget: int = 1) -> None:
        """Reserve space for the RBF model.

        This routine avoids successive dynamic memory allocations with
        successive calls of :meth:`update()`. If the input `maxeval` is smaller
        than the current number of sample points, nothing is done.

        :param maxeval: Maximum number of function evaluations.
        :param dim: Dimension of the domain space.
        :param ntarget: Dimension of the target space.
        """
        if maxeval < self._m:
            return
        if maxeval == self._m and self.dim == dim and self.ntarget == ntarget:
            return

        if self._x.size == 0:
            self._x = np.empty((maxeval, dim))
        else:
            additional_rows = max(0, maxeval - self._x.shape[0])
            self._x = np.concatenate(
                (self._x, np.empty((additional_rows, dim))), axis=0
            )

        if self._fx.size == 0:
            self._fx = (
                np.empty(maxeval)
                if ntarget == 1
                else np.empty((maxeval, ntarget))
            )
        else:
            additional_values = max(0, maxeval - self._fx.shape[0])
            self._fx = np.concatenate(
                (
                    self._fx,
                    np.empty(additional_values)
                    if ntarget == 1
                    else np.empty((additional_values, ntarget)),
                ),
                axis=0,
            )

        if self._coef.size == 0:
            self._coef = (
                np.empty(maxeval + self.polynomial_tail_size())
                if ntarget == 1
                else np.empty((maxeval + self.polynomial_tail_size(), ntarget))
            )
        else:
            additional_values = max(
                0, maxeval + self.polynomial_tail_size() - self._coef.shape[0]
            )
            self._coef = np.concatenate(
                (
                    self._coef,
                    np.empty(additional_values)
                    if ntarget == 1
                    else np.empty((additional_values, ntarget)),
                ),
                axis=0,
            )

        if self._PHI.size == 0:
            self._PHI = np.empty((maxeval, maxeval))
        else:
            additional_rows = max(0, maxeval - self._PHI.shape[0])
            additional_cols = max(0, maxeval - self._PHI.shape[1])
            new_rows = max(maxeval, self._PHI.shape[0])
            self._PHI = np.concatenate(
                (
                    np.concatenate(
                        (
                            self._PHI,
                            np.empty((additional_rows, self._PHI.shape[1])),
                        ),
                        axis=0,
                    ),
                    np.empty((new_rows, additional_cols)),
                ),
                axis=1,
            )

    def eval_kernel(self, x, y=None):
        """Evaluate the RBF kernel between points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param y: n-by-d matrix with n point coordinates in a d-dimensional
            space. If None, use x.
        :return: Kernel matrix (m x n).
        """
        if y is None:
            y = x
        return self.rbf(cdist(x, y))

    def polynomial_tail(self, x: np.ndarray) -> np.ndarray:
        """Computes the polynomial tail matrix for a given set of points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        """
        m = len(x)
        dim = x.shape[1]
        order = self.rbf.polynomial_tail_order()
        xs = (x - self.shift) / self.scale

        P = np.empty((m, 0))
        if order >= 0:
            P = np.ones((m, 1))
            if order >= 1:
                P = np.concatenate((P, xs), axis=1)
                if order >= 2:
                    for i in range(dim):
                        xi = xs[:, i : i + 1]
                        P = np.concatenate((P, xi * xs[:, i:dim]), axis=1)
                    if order >= 3:
                        raise ValueError("Invalid polynomial tail")

        return P

    def _polynomial_tail_basis_single_x(self, x: np.ndarray, i: int):
        """Computes the polynomial tail ith basis function matrix at a given x.

        :param x: Point in a d-dimensional space.
        :param i: Index of the basis function.
        """
        dim = len(x)
        tail_size = self.polynomial_tail_size()
        assert i < tail_size, "Index out of bounds for polynomial tail size."

        xs = (x - self.shift) / self.scale

        if i <= 0:
            return 1.0

        i -= 1
        if i < dim:
            return xs[i]

        i -= dim
        for j in range(dim):
            if i < dim - j:
                return xs[j] * xs[j + i]
            i -= dim - j

        raise ValueError("Invalid polynomial tail")

    def __call__(
        self, x: np.ndarray, i: int = -1, return_dist: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Evaluates the model at one or multiple points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param i: Index of the target dimension to evaluate. If -1,
            evaluate all.
        :param return_dist: If `True`, returns the distance matrix between the
            input points and the training points.
        :return:

            * Value for the RBF model on each of the input points.

            * Matrix D where D[i, j] is the distance between the i-th
                input point and the j-th training point.
        """
        dim = self.dim
        X = x.reshape(-1, dim)

        # Coefficients for the RBF model
        coef0 = self._coef[0 : self._m]
        coef1 = self._coef[self._m : self._m + self.polynomial_tail_size()]
        if i >= 0:
            assert i < self.ntarget, (
                "Index out of bounds for target dimension."
            )
            if self.ntarget > 1:
                coef0 = coef0[:, i]
                coef1 = coef1[:, i]

        # compute pairwise distances between candidates and sampled points
        D = cdist(X, self.X)

        y = np.matmul(self.rbf(D), coef0)

        Px = self.polynomial_tail(X)
        if Px.size > 0:
            y += np.dot(Px, coef1)

        if return_dist:
            return y, D

        return y

    def jac(self, x: np.ndarray) -> np.ndarray:
        r"""Evaluates the derivative of the model at one point.

        .. math::

            \nabla f(x) = \sum_{i=1}^{m} \beta_i \frac{\phi'(r_i)}{r_i} x
                        + \sum_{i=1}^{n} \beta_{m+i} \nabla p_i(x).

        where :math:`r_i = \|x - x_i\|`.

        :param x: Point in a d-dimensional space.
        """
        dim = self.dim
        pdim = self.polynomial_tail_size()

        # compute pairwise distances between candidates and sampled points
        d = cdist(x.reshape(-1, dim), self.X).flatten()

        A = np.array([self.rbf.grad_over_r(d[i]) * x for i in range(d.size)])
        B = np.array(
            [
                grad(self._polynomial_tail_basis_single_x, argnum=0)(x, i=i)
                for i in range(pdim)
            ]
        )

        y = np.matmul(A.T, self._coef[0 : self._m])
        if B.size > 0:
            y += np.matmul(B.T, self._coef[self._m : self._m + pdim])

        return y.flatten()

    def hessp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        r"""Evaluates the Hessian of the model at x in the direction of v.

        .. math::

            H(f)(x) v   = \sum_{i=1}^{m} \beta_i \left(
                            \phi''(r_i)\frac{(x^Tv)x}{r_i^2} +
                            \frac{\phi'(r_i)}{r_i}
                            \left(v - \frac{(x^Tv)x}{r_i^2}\right)
                        \right)
                        + \sum_{i=1}^{n} \beta_{m+i} H(p_i)(x) v.

        where :math:`r_i = \|x - x_i\|`.

        :param x: Point in a d-dimensional space.
        :param v: Direction in which the Hessian is evaluated.
        """
        dim = self.dim
        pdim = self.polynomial_tail_size()

        # compute pairwise distances between candidates and sampled points
        d = cdist(x.reshape(-1, dim), self.X).flatten()

        xxTp = np.dot(v, x) * x
        A = np.array(
            [
                self.rbf.hess(d[i]) * (xxTp / (d[i] * d[i]))
                + self.rbf.grad_over_r(d[i]) * (v - (xxTp / (d[i] * d[i])))
                for i in range(d.size)
            ]
        )
        B = np.array(
            [
                hessian(self._polynomial_tail_basis_single_x, argnum=0)(x, i=i)
                @ v
                for i in range(pdim)
            ]
        )

        y = np.matmul(A.T, self._coef[0 : self._m])
        if B.size > 0:
            y += np.matmul(B.T, self._coef[self._m : self._m + pdim])

        return y.flatten()

    def update(self, xNew: np.ndarray, fx) -> None:
        """Updates the model with new pairs of data (x,y).

        :param xNew: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param fx: Function values on the sampled points.
        """
        oldm = self._m
        newm = xNew.shape[0]
        dim = xNew.shape[1]
        m = oldm + newm

        if oldm > 0:
            assert dim == self.dim
        if newm == 0:
            return

        # Reserve space for the new data
        self.reserve(
            m,
            dim,
            np.asarray(fx).shape[-1]
            if (oldm == 0 and np.asarray(fx).ndim > 1)
            else self.ntarget,
        )

        # Update x and fx
        self._x[oldm:m] = xNew
        self._fx[oldm:m] = fx

        # Compute distances between new points and sampled points
        distNew = cdist(self._x[oldm:m], self._x[0:m])

        # Update matrices _PHI and _P
        self._PHI[oldm:m, 0:m] = self.rbf(distNew)
        self._PHI[0:oldm, oldm:m] = self._PHI[oldm:m, 0:oldm].T

        # Update m
        self._m = m

        # Get full matrix for the fitting
        A = self._get_RBFmatrix()

        # TODO: See if there is a solver specific for saddle-point systems
        zero_shape = list(self.Y.shape)
        zero_shape[0] = self.polynomial_tail_size()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._coef = solve(
                    A,
                    np.concatenate(
                        (self.filter(self.Y), np.zeros(zero_shape))
                    ),
                    assume_a="sym",
                )
            except np.linalg.LinAlgError as e:
                from numpy.linalg import cond

                condA = cond(A)
                print(f"Condition number of A: {condA}")
                print(f"A.X: {self.X}")

                raise np.linalg.LinAlgError(
                    "Failed to solve the RBF model system. "
                    "This may be due to a singular matrix or insufficient data."
                ) from e

    @property
    def X(self) -> np.ndarray:
        """Get the training data points.

        :return: m-by-d matrix with m training points in a d-dimensional space.
        """
        return self._x[0 : self._m]

    @property
    def Y(self) -> np.ndarray:
        """Get f(x) for the sampled points."""
        return self._fx[0 : self._m]

    def _compute_shift_and_scale(self) -> None:
        if len(self.X) > 0:
            min_x = np.min(self.X, axis=0)
            max_x = np.max(self.X, axis=0)
            self.shift = (max_x + min_x) / 2
            self.scale = (max_x - min_x) / 2
            self.scale[self.scale == 0.0] = 1.0

    def _get_RBFmatrix(self) -> np.ndarray:
        r"""Get the complete matrix used to compute the RBF weights.

        This is a blocked matrix :math:`[[\Phi, P],[P^T, 0]]`, where
        :math:`\Phi` is the kernel matrix, and
        :math:`P` is the polynomial tail basis matrix.

        :return: (m+pdim)-by-(m+pdim) matrix used to compute the RBF weights.
        """
        pdim = self.polynomial_tail_size()

        self._compute_shift_and_scale()
        _P = self.polynomial_tail(self.X)

        return np.block(
            [
                [self._PHI[0 : self._m, 0 : self._m], _P],
                [_P.T, np.zeros((pdim, pdim))],
            ]
        )

    def min_design_space_size(self, dim: int) -> int:
        """Return the minimum design space size for a given space dimension."""
        order = self.rbf.polynomial_tail_order()
        if order == -1:
            return 1
        else:
            return comb(dim + order, order)

    def polynomial_tail_size(self) -> int:
        """Get the dimension of the polynomial tail."""
        order = self.rbf.polynomial_tail_order()
        if order == -1:
            return 0
        else:
            return comb(self.dim + order, order)

    def check_initial_design(self, sample: np.ndarray) -> bool:
        """Check if the sample is able to generate a valid surrogate.

        :param sample: m-by-d matrix with m training points in a d-dimensional
            space.
        """
        if self.polynomial_tail_size() == 0:
            return True
        if len(sample) < 1:
            return False
        P = self.polynomial_tail(sample)
        return np.linalg.matrix_rank(P) == P.shape[1]

    @property
    def iindex(self) -> tuple[int, ...]:
        """Return iindex, the sequence of integer variable indexes."""
        return self._iindex

    def prepare_mu_measure(self) -> None:
        """Prepare the model for mu measure computation.

        This routine computes the LDLt factorization of the matrix A, which is
        used to compute the mu measure. The factorization is computed only once
        and can be reused for multiple calls to :meth:`mu_measure`.
        """
        self._LDLt = ldl(self._get_RBFmatrix())

    def mu_measure(self, x: np.ndarray) -> np.ndarray:
        """Compute the value of abs(mu) for an RBF model.

        The mu measure was first defined in [#]_ with suggestions of usage for
        global optimization with RBF functions. In [#]_, the authors detail the
        strategy to make the evaluations computationally viable.

        The current
        implementation, uses a different strategy than that from Björkman and
        Holmström (2000), where a single LDLt factorization is used instead of
        the QR and Cholesky factorizations. The new algorithm's performs 10
        times less operations than the former. Like the former, the new
        algorithm is also able to use high-intensity linear algebra operations
        when the routine is called with multiple points :math:`x` are evaluated
        at once.

        .. note::
            Before calling this method, the model must be prepared with
            :meth:`prepare_mu_measure`.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :return: The value of abs(mu) for every point in `x`.

        References
        ----------
        .. [#] Gutmann, HM. A Radial Basis Function Method for Global
            Optimization. Journal of Global Optimization 19, 201–227 (2001).
            https://doi.org/10.1023/A:1011255519438
        .. [#] Björkman, M., Holmström, K. Global Optimization of Costly
            Nonconvex Functions Using Radial Basis Functions. Optimization and
            Engineering 1, 373–397 (2000). https://doi.org/10.1023/A:1011584207202
        """
        # compute rbf value of the new point x
        xdist = cdist(self.X, x)
        newCols = np.concatenate(
            (
                np.asarray(self.rbf(xdist)),
                self.polynomial_tail(x).T,
            ),
            axis=0,
        )

        # Get the L factor, the block-diagonal matrix D, and the permutation
        # vector p
        ptL, D, p = self._LDLt
        L = ptL[p]

        # 0. Permute the new terms
        newCols = newCols[p]

        # 1. Solve P [a;b] = L (D l) for (D l)
        Dl = solve_triangular(
            L,
            newCols,
            lower=True,
            unit_diagonal=True,
            # check_finite=False,
            overwrite_b=True,
        )

        # 2. Compute l := inv(D) (Dl)
        ell = Dl.copy()
        i = 0
        while i < len(ell) - 1:
            if D[i + 1, i] == 0:
                # Invert block of size 1x1
                ell[i] /= D[i, i]
                i += 1
            else:
                # Invert block of size 2x2
                det = D[i, i] * D[i + 1, i + 1] - D[i, i + 1] ** 2
                ell[i], ell[i + 1] = (
                    (ell[i] * D[i + 1, i + 1] - ell[i + 1] * D[i, i + 1])
                    / det,
                    (ell[i + 1] * D[i, i] - ell[i] * D[i, i + 1]) / det,
                )
                i += 2
        if i == len(ell) - 1:
            # Invert last block of size 1x1
            ell[i] /= D[i, i]

        # 3. d = \phi(0) - l^T D l and \mu = 1/d
        d = self.rbf(np.array(0.0)).item() - (ell * Dl).sum(axis=0)
        mu = np.where(d != 0, 1 / d, np.inf)

        # Return huge value if the matrix is ill-conditioned
        mu = np.where(mu <= 0, np.inf, mu)

        return mu

    def reset_data(self) -> None:
        self._m = 0
