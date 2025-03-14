"""Gaussian process module."""

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

import copy
import warnings
import numpy as np
from sklearn import preprocessing
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
import scipy.optimize as scipy_opt
from sklearn.gaussian_process.kernels import RBF as GPkernelRBF


class GaussianProcess(GaussianProcessRegressor):
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
    :param maxiterLBFGS: Maximum number of iterations for the L-BFGS-B optimizer.
        Stored in :attr:`maxiterLBFGS`.

    .. attribute:: maxiterLBFGS

        Maximum number of iterations for the L-BFGS-B optimizer. Used in the
        training of the gaussian process.

    .. attribute:: scaler

        Scaler used to preprocess input data.

    """

    def __init__(
        self,
        kernel=None,
        *,
        scaler=None,
        maxiterLBFGS: int = 15000,
        **kwargs,
    ) -> None:
        super().__init__(kernel, **kwargs)
        self.X_train_ = np.array([])
        self.y_train_ = np.array([])
        self._y_train_mean = np.array([])
        self._y_train_std = np.array([])

        # Not in GaussianProcessRegressor:
        self.scaler = scaler
        self.maxiterLBFGS = maxiterLBFGS

        # Redefine some of the defaults:
        if kernel is None:
            self.kernel = GPkernelRBF()
        if "optimizer" not in kwargs:
            self.optimizer = self._optimizer
        if "normalize_y" not in kwargs:
            self.normalize_y = True
        if "n_restarts_optimizer" not in kwargs:
            self.n_restarts_optimizer = 10

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the model at one or multiple points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :return:

            * Mean value predicted by the GP model on each of the input points.
            * Std value predicted by the GP model on each of the input points.
        """
        return self.predict(
            x if self.scaler is None else self.scaler.transform(x),
            return_std=True,
            return_cov=False,
        )

    def xtrain(self) -> np.ndarray:
        """Get the training data points.

        :return: m-by-d matrix with m training points in a d-dimensional space.
        """
        if len(self.X_train_) == 0 or self.scaler is None:
            return self.X_train_
        else:
            return self.scaler.inverse_transform(self.X_train_)

    def eval_kernel(self, x, y=None):
        """Evaluate the kernel function at a pair (x,y).

        The structure of the kernel is the same as the one passed as parameter
        but with optimized hyperparameters.

        :param x: First entry in the tuple (x,y).
        :param y: Second entry in the tuple (x,y). If None, use x.
        """
        if self.scaler is None:
            return self.kernel_(x, x) if y is None else self.kernel_(x, y)
        else:
            xs = self.scaler.transform(x)
            if y is None:
                return self.kernel_(xs, xs)
            else:
                return self.kernel_(xs, self.scaler.transform(y))

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
        try:
            copy.deepcopy(self).fit(
                preprocessing.MinMaxScaler().fit_transform(sample),
                np.ones(len(sample)),
            )
            return True
        except np.linalg.LinAlgError:
            return False

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
        if self.ntrain() > 0:
            X = np.concatenate((self.xtrain(), Xnew), axis=0)
            y = np.concatenate((self.ytrain(), ynew), axis=0)
        else:
            X = Xnew
            y = ynew

        if self.optimizer == self._optimizer:
            # Prepare flag for verifying overall optimizer success
            self._optimizer_success = False
            self._optimizer_status = 0
            self._optimizer_message = ""

        if self.scaler is None:
            self.fit(X, y)
        else:
            self.scaler = self.scaler.fit(X)
            self.fit(self.scaler.transform(X), y)

        if self.optimizer == self._optimizer:
            # Check for overall failure
            if not self._optimizer_success:
                warnings.warn(
                    (
                        "L-BFGS-B failed to converge (status={}):\n{}.\n\n"
                        "Increase the number of iterations (maxiterLBFGS > {}) "
                        "or scale the data as shown in:\n"
                        "    https://scikit-learn.org/stable/modules/"
                        "preprocessing.html"
                    ).format(
                        self._optimizer_status,
                        self._optimizer_message,
                        self.maxiterLBFGS,
                    ),
                    ConvergenceWarning,
                    stacklevel=2,
                )

    def ntrain(self) -> int:
        """Get the number of sampled points."""
        return len(self.xtrain())

    def get_iindex(self) -> tuple[int, ...]:
        """Return iindex, the sequence of integer variable indexes."""
        return ()

    def ytrain(self) -> np.ndarray:
        """Get f(x) for the sampled points."""
        return self._y_train_mean + self.y_train_ * self._y_train_std

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
            obj_func,
            initial_theta,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": self.maxiterLBFGS},
        )

        if res.success:
            self._optimizer_success = True
        else:
            self._optimizer_status = res.status
            self._optimizer_message = res.message

        return res.x, res.fun
