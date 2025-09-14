"""Use test problems from the pyGOLD library to check the algorithms provided by soogo."""
import pygold
import numpy as np
import pytest
import time

from soogo import (
    surrogate_optimization,
    multistart_msrs,
    dycors,
    cptv,
    cptvl,
    socemo,
    gosac,
    bayesian_optimization,
)

def make_soogo_objective(prob_instance):
    """
    Wraps a pyGOLD problem instance's evaluate method to support batch input
    for use with soogo algorithms.
    """
    def soogo_objective(X):
        X = np.atleast_2d(X)
        return np.array([prob_instance.evaluate(x) for x in X])
    return soogo_objective

def make_soogo_constraint(prob_instance):
    """
    Wraps a pyGOLD problem instance's constraint method to support batch input
    for use with soogo algorithms.
    """
    def soogo_constraint(X):
        X = np.atleast_2d(X)
        return np.array([-prob_instance.constraint1(x) for x in X])
    return soogo_constraint

unconstrained_algorithms = [multistart_msrs, dycors, cptv, cptvl]
@pytest.mark.parametrize("alg", unconstrained_algorithms)
def test_unconstrained_algorithms(
    alg,
    dim=3,
    n_runs=5,
    maxevals=250,
    tol=1,
    min_success_rate=0.6,
    problems = [
        pygold.standard.Trid,       # Bowl-shaped
        pygold.standard.Zakharov,   # Plate-shaped
        pygold.standard.Griewank,   # Dispersed local minima
    ]
):
    """
    Test unconstrained single-objective algorithms from soogo on a set of
    standard optimization problems from the pyGOLD library. Ensures that
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
    :param problems: List of pyGOLD problem classes to test (default includes
        Trid, Zakharov, and Griewank).
    """
    start_time = time.time()
    for problem in problems:
        prob_time = time.time()
        prob_instance = problem(dim)
        min_value = prob_instance.min()
        run_vals = []
        soogo_objective = make_soogo_objective(prob_instance)
        for run in range(n_runs):
            run_time = time.time()
            np.random.seed(run + 42)
            out = alg(soogo_objective, prob_instance.bounds(), maxevals)
            run_vals.append(out.fx)
            print(f"Testing {alg.__name__} on {problem.__name__}, run {run+1}: fx = {out.fx}, best known = {min_value}, time = {time.time()-run_time:.2f}s")
        run_vals = np.array(run_vals)
        n_success = np.sum(np.abs(run_vals - min_value) < tol)
        success_rate = n_success / n_runs
        print(f"Total time for {alg.__name__} on {problem.__name__}: {time.time()-prob_time:.2f}s")
        assert success_rate >= min_success_rate, (
            f"{alg.__name__} failed on {problem.__name__}: success rate {success_rate:.2f} < {min_success_rate}"
        )
    print(f"Total time for {alg.__name__}: {time.time()-start_time:.2f}s")

slow_algorithms = [surrogate_optimization, bayesian_optimization]
@pytest.mark.parametrize("alg", slow_algorithms)
def test_unconstrained_quick(alg):
    """
    A test for unconstrained single-objective algorithms from soogo with
    lower maxevals and looser tolerance than the default test, meant to test
    algorithms that take too long to run with the default
    test_unconstrained_algorithms settings.

    This is a wrapper to test_unconstrained_algorithms with dimension
    reduced to 2, number of runs reduced to 2, maxevals reduced to 125,
    and min_success_rate reduced to 0.5.

    :param alg: The optimization algorithm from soogo to test.
    """
    test_unconstrained_algorithms(alg, dim=2, n_runs=2, maxevals=125, tol=1, min_success_rate=0.5)

constrained_algorithms = [gosac]
@pytest.mark.parametrize("alg", constrained_algorithms)
def test_constrained_algorithms(
    alg,
    dim=2,
    n_runs=2,
    maxevals=100,
    tol=1,
    min_success_rate=0.5,
    problems = [
        pygold.standard.RosenbrockConstrained,
    ]
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
    :param problems: List of pyGOLD problem classes to test (default is
        RosenbrockConstrained).
    """
    start_time = time.time()
    for problem in problems:
        prob_time = time.time()
        prob_instance = problem(dim)
        min_value = prob_instance.min()
        run_vals = []
        soogo_objective = make_soogo_objective(prob_instance)
        soogo_constraint = make_soogo_constraint(prob_instance)
        for run in range(n_runs):
            run_time = time.time()
            np.random.seed(run + 142)
            out = alg(soogo_objective, soogo_constraint, prob_instance.bounds(), maxevals)
            assert -prob_instance.constraint1(out.x) <= 0, "Returned solution does not satisfy constraint"
            run_vals.append(out.fx[0])
            print(f"Testing {alg.__name__} on {problem.__name__}, run {run+1}: fx = {out.fx}, best known = {min_value}, time = {time.time()-run_time:.2f}s")
        run_vals = np.array(run_vals)
        n_success = np.sum(np.abs(run_vals - min_value) < tol)
        success_rate = n_success / n_runs
        print(f"Total time for {alg.__name__} on {problem.__name__}: {time.time()-prob_time:.2f}s")
        assert success_rate >= min_success_rate, (
            f"{alg.__name__} failed on {problem.__name__}: success rate {success_rate:.2f} < {min_success_rate}"
        )
    print(f"Total time for {alg.__name__}: {time.time()-start_time:.2f}s")

## Add test for multiobjective algorithms (socemo())
