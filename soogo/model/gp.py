"""Gaussian process module."""

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

import warnings
import numpy as np
import scipy.optimize as scipy_opt

# Scikit-learn imports
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF as GPkernelRBF

# Local imports
from .base import Surrogate
from ..utils import gp_expected_improvement


class GaussianProcess(Surrogate):
    """Gaussian Process model.

    This model uses default attributes and parameters from
    GaussianProcessRegressor with the following exceptions:

    * :attr:`kernel`: Default is `sklearn.gaussian_process.kernels.RBF()`.
    * :attr:`optimizer`: Default is :meth:`_optimizer()`.
    * :attr:`normalize_y`: Default is `True`.
    * :attr:`n_restarts_optimizer`: Default is 10.

    Check other attributes and parameters for GaussianProcessRegressor at
    https://scikit-learn.org/dev/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html.

    :param scaler: Scaler for the input data. For details, see
        https://scikit-learn.org/stable/modules/preprocessing.html.

    .. attribute:: scaler

        Scaler used to preprocess input data.

    .. attribute:: model

        The underlying GaussianProcessRegressor model instance.
        This is initialized with the provided parameters and can be accessed
        for further customization or inspection.

    """

    def __init__(self, scaler=None, **kwargs) -> None:
        super().__init__()

        # Scaler for x
        self.scaler = scaler

        # Redefine some of the defaults in GaussianProcessRegressor:
        if "kernel" not in kwargs:
            kwargs["kernel"] = GPkernelRBF()
        if "optimizer" not in kwargs:
            kwargs["optimizer"] = self._optimizer
        if "normalize_y" not in kwargs:
            kwargs["normalize_y"] = True
        if "n_restarts_optimizer" not in kwargs:
            kwargs["n_restarts_optimizer"] = 10

        self.model = GaussianProcessRegressor(**kwargs)

    # TODO: Make this method more useful
    def reserve(self, n: int, dim: int, ntarget: int = 1) -> None:
        return

    def __call__(
        self,
        x: np.ndarray,
        i: int = -1,
        return_std: bool = False,
        return_cov: bool = False,
    ):
        """Evaluates the model at one or multiple points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param i: Index of the target dimension to evaluate. If -1,
            evaluate all.
        :param return_std: If `True`, returns the standard deviation of the
            predictions.
        :param return_cov: If `True`, returns the covariance of the predictions.
        :return:

            * m-by-n matrix with m predictions.

            * If `return_std` is `True`, the second output is a m-by-n matrix
                with the standard deviations.

            * If `return_cov` is `True`, the third output is a m-by-m
                matrix with the covariances if n=1, otherwise it is a
                m-by-m-by-n matrix.
        """
        res = self.model.predict(
            x if self.scaler is None else self.scaler.transform(x),
            return_std=return_std,
            return_cov=return_cov,
        )
        assert i < self.ntarget
        if i == -1 or self.ntarget == 1:
            return res
        else:
            assert i >= 0
            if return_std or return_cov:
                if return_std:
                    if return_cov:
                        return res[0][:, i], res[1][:, i], res[2][:, :, i]
                    else:
                        return res[0][:, i], res[1][:, i]
                else:
                    return res[0][:, i], res[1][:, :, i]
            else:
                return res[:, i]

    @property
    def X(self) -> np.ndarray:
        """Get the training data points.

        :return: m-by-d matrix with m training points in a d-dimensional space.
        """
        if not hasattr(self.model, "X_train_"):  # Unfitted
            return np.empty((0, 0))
        if self.scaler is None:
            return self.model.X_train_
        else:
            return self.scaler.inverse_transform(self.model.X_train_)

    def eval_kernel(self, x, y=None):
        xs = x if self.scaler is None else self.scaler.transform(x)
        if y is None:
            return self.model.kernel_(xs, xs)
        else:
            ys = y if self.scaler is None else self.scaler.transform(y)
            return self.model.kernel_(xs, ys)

    def min_design_space_size(self, dim: int) -> int:
        """Return the minimum design space size for a given space dimension."""
        return 1 if dim > 0 else 0

    def check_initial_design(self, sample: np.ndarray) -> bool:
        """Check if the sample is able to generate a valid surrogate.

        :param sample: m-by-d matrix with m training points in a d-dimensional
            space.
        """
        if sample.ndim != 2 or len(sample) < 1:
            return False
        return True

    def update(self, Xnew, ynew) -> None:
        """Updates the model with new pairs of data (x,y).

        When the default optimizer method, :meth:`_optimizer()`, is used as
        :attr:`optimizer`, this routine reports different warnings compared to
        `sklearn.gaussian_process.GaussianProcessRegressor.fit()`. The latter
        reports any convergence failure in L-BFGS-B. This implementation reports
        the last convergence failure in the multiple L-BFGS-B runs only if there
        all the runs end up failing. The number of optimization runs is
        :attr:`n_restarts_optimizer` + 1.

        :param Xnew: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param ynew: Function values on the sampled points.
        """
        if self.ntrain > 0:
            X = np.concatenate((self.X, Xnew), axis=0)
            y = np.concatenate((self.Y, ynew), axis=0)
        else:
            X = Xnew
            y = ynew

        if self.scaler is None:
            self.model.fit(X, y)
        else:
            self.scaler = self.scaler.fit(X)
            self.model.fit(self.scaler.transform(X), y)

        if hasattr(self, "_optimizer_success"):
            # Check for overall failure
            if not self._optimizer_success:
                warnings.warn(
                    (
                        "L-BFGS-B failed to converge (status={}):\n{}.\n\n"
                        "Increase the number of iterations (maxiter) "
                        "or scale the data as shown in:\n"
                        "    https://scikit-learn.org/stable/modules/"
                        "preprocessing.html"
                    ).format(self._optimizer_status, self._optimizer_message),
                    ConvergenceWarning,
                    stacklevel=2,
                )
            del self._optimizer_success

    @property
    def iindex(self) -> tuple[int, ...]:
        """Return iindex, the sequence of integer variable indexes."""
        return ()

    @property
    def Y(self) -> np.ndarray:
        """Get f(x) for the sampled points."""
        if not hasattr(self.model, "y_train_"):  # Unfitted
            return np.empty((0,))
        return (
            self.model._y_train_mean
            + self.model.y_train_ * self.model._y_train_std
        )

    def _optimizer(self, obj_func, initial_theta, bounds):
        """Optimizer used in the GP fitting.

        This function also sets the attributes: :attr:`_optimizer_success`,
        :attr:`self._optimizer_status` and :attr:`self._optimizer_message` to
        be used by :meth:`update()`.

        :param obj_func: The objective function to be minimized, which
            takes the hyperparameters theta as a parameter and an
            optional flag eval_gradient, which determines if the
            gradient is returned additionally to the function value.
        :param initial_theta: The initial value for theta, which can be
            used by local optimizers.
        :param bounds: The bounds on the values of theta.
        :return: Returned are the best found hyperparameters theta and
            the corresponding value of the target function.
        """
        res = scipy_opt.minimize(
            obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds
        )

        if res.success:
            self._optimizer_success = True
        else:
            self._optimizer_status = res.status
            self._optimizer_message = res.message

        return res.x, res.fun

    def expected_improvement(self, x, ybest):
        """Compute expected improvement at given points.

        :param x: Points at which to evaluate expected improvement.
        :param ybest: Best observed function value so far.
        :return: Expected improvement values.
        """
        mu, sigma = self(x, return_std=True)
        return gp_expected_improvement(ybest - mu, sigma)

    def reset_data(self) -> None:
        if hasattr(self.model, "X_train_"):
            del self.model.X_train_
        if hasattr(self.model, "y_train_"):
            del self.model.y_train_
