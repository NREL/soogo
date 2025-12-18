"""Test the optimization algorithms converge."""

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

__authors__ = ["Weslley S. Pereira", "Byron Selvage"]

import pygoblet
import numpy as np
import pytest

from soogo import (
    surrogate_optimization,
    multistart_msrs,
    dycors,
    cptv,
    cptvl,
    gosac,
    bayesian_optimization,
    fsapso,
)


class ShiftedProblem:
    """
    Shifts the input and bounds of a pyGOBLET problem by a random vector.
    """

    def __init__(self, base_class, dim):
        self.base = base_class(dim)
        lb, ub = np.array(self.base.bounds()).T
        rng = np.random.default_rng(12345)
        self.shift = rng.uniform(lb, ub)
        self._orig_bounds = (lb, ub)

    def evaluate(self, x):
        return self.base.evaluate(np.asarray(x) - self.shift)

    def min(self):
        return self.base.min()

    def bounds(self):
        lb, ub = self._orig_bounds
        return np.stack([lb + self.shift, ub + self.shift], axis=1)

    def constraint1(self, x):
        return self.base.constraint1(np.asarray(x) - self.shift)


def make_soogo_objective(prob_instance):
    """
    Wraps a pyGOBLET problem instance's evaluate method to support batch input
    for use with soogo algorithms.
    """
    return lambda X: np.array(
        [prob_instance.evaluate(x) for x in np.atleast_2d(X)]
    )


def make_soogo_constraint(prob_instance):
    """
    Wraps a pyGOBLET problem instance's constraint method to support batch input
    for use with soogo algorithms.
    """
    return lambda X: np.array(
        [-prob_instance.constraint1(x) for x in np.atleast_2d(X)]
    )


unconstrained_algorithms = [multistart_msrs, dycors, cptv, cptvl]
unconstrained_problems = [
    pygoblet.standard.Trid,  # Bowl-shaped, range ~(-7, 350)
    lambda dim: ShiftedProblem(
        pygoblet.standard.Zakharov, dim
    ),  # Plate-shaped, shifted, range ~(0, 58500)
    lambda dim: ShiftedProblem(
        pygoblet.standard.Griewank, dim
    ),  # Dispersed local minima, shifted, range ~(0, 250)
]


@pytest.mark.parametrize("alg", unconstrained_algorithms)
def test_unconstrained_algorithms(
    alg,
    dim=3,
    n_runs=5,
    maxevals=250,
    tol=1,
    min_success_rate=0.6,
    problems=unconstrained_problems,
):
    """
    Test unconstrained single-objective algorithms from soogo on a set of
    standard optimization problems from the pyGOBLET library. Ensures that
    an algorithm succeeds at solving each problem at least at a specified
    success rate.

    :param alg: The optimization algorithm from soogo to test.
    :param dim: Dimensionality of the test problems (default is 3).
    :param n_runs: Number of independent runs for each algorithm-problem pair
        (default is 5).
    :param maxevals: Maximum number of function evaluations allowed per run
        (default is 250).
    :param tol: Acceptable tolerance from known minimum to consider a run
        successful (default is 1).
    :param min_success_rate: Minimum required success rate (fraction of runs
        that must be successful) for the algorithm to pass the test
        (default is 0.6).
    :param problems: List of pyGOBLET problem classes to test (default includes
        Trid, Zakharov, and Griewank).
    """
    for problem in problems:
        prob_instance = problem(dim)
        min_value = prob_instance.min()
        run_vals = []
        soogo_objective = make_soogo_objective(prob_instance)
        for run in range(n_runs):
            np.random.seed(run + 42)
            out = alg(soogo_objective, prob_instance.bounds(), maxevals)
            if out.fx is None:
                raise ValueError("Algorithm did not return a function value.")
            run_vals.append(out.fx)
            print(
                f"Testing {alg.__name__} on {type(prob_instance).__name__}, run {run + 1}: fx = {out.fx}, best known = {min_value}"
            )
        run_vals = np.array(run_vals)
        n_success = np.sum(np.abs(run_vals - min_value) < tol)
        success_rate = n_success / n_runs
        assert success_rate >= min_success_rate, (
            f"{alg.__name__} failed on {type(prob_instance).__name__}: success rate {success_rate:.2f} < {min_success_rate}"
        )


slow_algorithms = [surrogate_optimization, bayesian_optimization, fsapso]


@pytest.mark.parametrize("alg", slow_algorithms)
def test_unconstrained_quick(alg):
    """
    A test for unconstrained single-objective algorithms from soogo with
    lower maxevals and looser tolerance than the default test, meant to test
    algorithms that take too long to run with the default
    test_unconstrained_algorithms settings.
    """
    test_unconstrained_algorithms(
        alg, dim=2, n_runs=2, maxevals=125, tol=1, min_success_rate=0.5
    )


constrained_algorithms = [gosac]


@pytest.mark.parametrize("alg", constrained_algorithms)
def test_constrained_algorithms(
    alg,
    dim=2,
    n_runs=2,
    maxevals=100,
    tol=1,
    min_success_rate=0.5,
    problems=[
        pygoblet.standard.RosenbrockConstrained,
    ],
):
    """
    A test for constrained single-objective algorithms from soogo.
    Ensures that an algorithm succeeds at solving each problem at least at a
    specified success rate.

    :param alg: The optimization algorithm from soogo to test.
    :param dim: Dimensionality of the test problems (default is 2).
    :param n_runs: Number of independent runs for each algorithm-problem pair
        (default is 2).
    :param maxevals: Maximum number of function evaluations allowed per run
        (default is 100).
    :param tol: Acceptable tolerance from known minimum to consider a run
        successful (default is 1).
    :param min_success_rate: Minimum required success rate (fraction of runs
        that must be successful) for the algorithm to pass the test
        (default is 0.5).
    :param problems: List of pyGOBLET problem classes to test (default is
        RosenbrockConstrained).
    """
    for problem in problems:
        prob_instance = problem(dim)
        min_value = prob_instance.min()
        run_vals = []
        soogo_objective = make_soogo_objective(prob_instance)
        soogo_constraint = make_soogo_constraint(prob_instance)
        for run in range(n_runs):
            np.random.seed(run + 142)
            out = alg(
                soogo_objective,
                soogo_constraint,
                prob_instance.bounds(),
                maxevals,
            )
            assert -prob_instance.constraint1(out.x) <= 0, (
                "Returned solution does not satisfy constraint"
            )
            run_vals.append(out.fx[0])
            print(
                f"Testing {alg.__name__} on {problem.__name__}, run {run + 1}: fx = {out.fx}, best known = {min_value}"
            )
        run_vals = np.array(run_vals)
        n_success = np.sum(np.abs(run_vals - min_value) < tol)
        success_rate = n_success / n_runs
        assert success_rate >= min_success_rate, (
            f"{alg.__name__} failed on {problem.__name__}: success rate {success_rate:.2f} < {min_success_rate}"
        )
