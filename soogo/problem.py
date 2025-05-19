"""Problem definitions for interfacing with pymoo."""

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

import numpy as np
from typing import Union
from scipy.spatial.distance import cdist

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer
from pymoo.core.duplicate import DefaultDuplicateElimination


def _get_vars(bounds, iindex=()) -> dict:
    """Get the type of variables for a problem.

    :param bounds: Bounds for the variables.
    :param iindex: Indices of the input space that are integer.
    :return: Dictionary with the variable types in the format expected by pymoo.
    """
    dim = len(bounds)
    vars = {
        i: Integer(bounds=bounds[i]) if i in iindex else Real(bounds=bounds[i])
        for i in range(dim)
    }
    return vars


def _dict_to_array(xdict: Union[dict, list[dict]]) -> np.ndarray:
    """Convert a dictionary indexed by a range(n_var) to an array of values.

    Also accepts a list of dictionaries, in which case it returns a 2D array.

    :param xdict: Dictionary with the variables or list of dictionaries.
    :return: Array with the values of the variables.
    """
    if isinstance(xdict, dict):
        return np.array([xdict[i] for i in sorted(xdict)])
    else:
        # xdict is a list of dictionaries
        return np.array([[xi[i] for i in sorted(xi)] for xi in xdict])


class ListDuplicateElimination(DefaultDuplicateElimination):
    """Specialization of DefaultDuplicateElimination for better performance
    in the problems we have.

    The particularity in this software is that the labels for the variables
    always go from 0 to n-1, so we can rely on that information to using
    ElementwiseDuplicateElimination.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def calc_dist(self, pop, other=None):
        """Compute the distances between the members of two populations.

        If `other` is None, compute the distance between `pop` with itself.

        This implementation uses the built-in Python :func:`sorted()`
        function for sorting the keys of the dictionary. This is faster than
        using the runtime :attr:`func()`.

        :param pop: First population of size m.
        :param other: Second population of size n.
        :return: m-by-n matrix with the distances.
        """
        X = np.array([[xi.X[i] for i in sorted(xi.X)] for xi in pop])
        if other is None:
            D = cdist(X, X)
            D[np.triu_indices(len(X))] = np.inf
        else:
            _X = np.array([[xi.X[i] for i in sorted(xi.X)] for xi in other])
            D = cdist(X, _X)
        return D


class ProblemWithConstraint(Problem):
    """Mixed-integer problem with constraints for pymoo.

    :param objfunc: Objective function. Stored in :attr:`objfunc`.
    :param gfunc: Constraint function. Stored in :attr:`gfunc`.
    :param bounds: List with the limits [x_min,x_max] of each direction x in
        the search space.
    :param iindex: Indices of the input space that are integer.
    :param n_ieq_constr: Number of inequality constraints.

    .. attribute:: objfunc

        Objective function.

    .. attribute:: gfunc

        Constraint function.

    """

    def __init__(self, objfunc, gfunc, bounds, iindex, n_ieq_constr: int = 1):
        vars = _get_vars(bounds, iindex)
        self.objfunc = objfunc
        self.gfunc = gfunc
        super().__init__(vars=vars, n_obj=1, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, X, out):
        x = _dict_to_array(X)
        out["F"] = self.objfunc(x)
        out["G"] = self.gfunc(x)


class ProblemNoConstraint(Problem):
    """Mixed-integer problem with no constraints for pymoo.

    :param objfunc: Objective function. Stored in :attr:`objfunc`.
    :param bounds: List with the limits [x_min,x_max] of each direction x in
        the search space.
    :param iindex: Indices of the input space that are integer.

    .. attribute:: objfunc

        Objective function.

    """

    def __init__(self, objfunc, bounds, iindex):
        vars = _get_vars(bounds, iindex)
        self.objfunc = objfunc
        super().__init__(vars=vars, n_obj=1)

    def _evaluate(self, X, out):
        x = _dict_to_array(X)
        out["F"] = self.objfunc(x)


class MultiobjTVProblem(Problem):
    """Mixed-integer multi-objective problem whose objective functions is the
    entry-wise absolute difference between the surrogate models and the target
    values.

    :param surrogateModels: List of surrogate models. Stored in
        :attr:`surrogateModels`.
    :param tau: List of target values. Stored in :attr:`tau`.
    :param bounds: List with the limits [x_min,x_max] of each direction x in
        the search space.

    .. attribute:: surrogateModels

        List of surrogate models.

    .. attribute:: tau

        List of target values.

    """

    def __init__(self, surrogateModels, tau, bounds):
        vars = _get_vars(bounds, surrogateModels[0].iindex)
        self.surrogateModels = surrogateModels
        self.tau = tau
        super().__init__(vars=vars, n_obj=len(surrogateModels))

    def _evaluate(self, X, out):
        assert self.elementwise is False
        x = _dict_to_array(X)
        out["F"] = np.empty((x.shape[0], self.n_obj))
        for i in range(self.n_obj):
            out["F"][:, i] = np.absolute(
                self.surrogateModels[i](x)[0] - self.tau[i]
            )


class MultiobjSurrogateProblem(Problem):
    """Mixed-integer multi-objective problem whose objective functions is the
    evaluation function of the surrogate models.

    :param surrogateModels: List of surrogate models. Stored in
        :attr:`surrogateModels`.
    :param bounds: List with the limits [x_min,x_max] of each direction x in
        the search space.

    .. attribute:: surrogateModels

        List of surrogate models.

    """

    def __init__(self, surrogateModels, bounds):
        vars = _get_vars(bounds, surrogateModels[0].iindex)
        self.surrogateModels = surrogateModels
        super().__init__(vars=vars, n_obj=len(surrogateModels))

    def _evaluate(self, X, out):
        assert self.elementwise is False
        x = _dict_to_array(X)
        out["F"] = np.empty((x.shape[0], self.n_obj))
        for i in range(self.n_obj):
            out["F"][:, i] = self.surrogateModels[i](x)[0]
