"""Fast surrogate-assisted particle swarm optimization (FSAPSO)."""

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

from typing import Callable, Optional

import numpy as np
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.core.individual import Individual
from pymoo.core.population import Population

from ..acquisition import (
    MaximizeDistance,
    MinimizeSurrogate,
    MultipleAcquisition,
    Acquisition,
    FarEnoughSampleFilter,
)
from ..model import RbfModel, Surrogate, MedianLpfFilter
from .utils import OptimizeResult, evaluate_and_log_point, uncertainty_score
from ..integrations.pymoo import PymooProblem
from ..sampling import SpaceFillingSampler


def fsapso(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    acquisitionFunc: Optional[Acquisition] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    disp: bool = False,
    seed=None,
) -> OptimizeResult:
    """
    Minimize a scalar function of one or more variables using the fast
    surrogate-assisted particle swarm optimization (FSAPSO) algorithm
    presented in [#]_.

    :param fun: The objective function to be minimized.
    :param bounds: List with the limits [x_min,x_max] of each direction x in the
        search space.
    :param maxeval: Maximum number of function evaluations.
    :param surrogateModel: Surrogate model to be used. If None is provided, a
        :class:`.RbfModel` model with cubic kernel is used. On exit, if
        provided, the surrogate model will contain the points used during the
        optimization.
    :param callback: If provided, the callback function will be called after
        each iteration with the current optimization result. The default is
        None.
    :param disp: If True, print information about the optimization process.
        The default is False.
    :param seed: Seed or random number generator.

    :return: The optimization result.


    References
    ----------
    .. [#] Li, F., Shen, W., Cai, X., Gao, L., & Gary Wang, G. 2020; A
        fast surrogate-assisted particle swarm optimization algorithm for
        computationally expensive problems. Applied Soft Computing, 92,
        106303. https://doi.org/10.1016/j.asoc.2020.106303
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # FAPSO parameters
    vMax = 0.1 * np.array([b[1] - b[0] for b in bounds])  # max velocity
    nSwarm = 20
    tol = np.min([np.sqrt(0.001**2 * dim), 5e-5 * dim * 10 * vMax.min()])

    # Random number generator
    rng = np.random.default_rng(seed)

    # Initialize optional variables
    return_surrogate = True
    if surrogateModel is None:
        return_surrogate = False
        surrogateModel = RbfModel(filter=MedianLpfFilter())
    if acquisitionFunc is None:
        acquisitionFunc = MultipleAcquisition(
            (
                MinimizeSurrogate(
                    1000,
                    rtol=1e-3,
                    seed=rng.integers(np.iinfo(np.int32).max).item(),
                ),
                MaximizeDistance(
                    rtol=1e-3,
                    seed=rng.integers(np.iinfo(np.int32).max).item(),
                ),
            )
        )

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(surrogateModel.ntrain + maxeval, dim)

    # Initialize output
    out = OptimizeResult()
    out.init(fun, bounds, 1, maxeval, surrogateModel, seed=seed)
    out.init_best_values(surrogateModel)
    assert isinstance(out.x, np.ndarray)

    # Select initial swarm
    _x = np.vstack((surrogateModel.X, out.sample[0 : out.nfev]))
    if surrogateModel.ntrain + out.nfev >= nSwarm:
        # Take 20 best points as initial swarm
        _fx = np.hstack((surrogateModel.Y, out.fsample[0 : out.nfev]))
        bestIndices = np.argsort(_fx)[:nSwarm]
        swarmInitX = _x[bestIndices]

        if disp:
            print(f"Selected {nSwarm} best training points for initial swarm")

    else:
        # If not enough training data, use random sampling
        if disp:
            print(
                "Not enough training data for initial swarm. Using random sampling to increase population."
            )

        swarmInitX = SpaceFillingSampler(
            seed=rng.integers(np.iinfo(np.int32).max).item()
        ).generate(
            nSwarm - surrogateModel.ntrain,
            bounds,
            current_sample=_x,
            iindex=surrogateModel.iindex,
        )
        swarmInitX = np.vstack((_x, swarmInitX))

    # PSO problem
    surrogateProblem = PymooProblem(
        objfunc=lambda x: surrogateModel(x).reshape(-1, 1), bounds=bounds
    )

    # Initialize PSO algorithm
    pso = PSO(
        pop_size=nSwarm,
        c1=1.491,
        c2=1.491,
        max_velocity_rate=vMax,
        adaptive=False,
        seed=rng.integers(np.iinfo(np.int32).max).item(),
    )
    pso.setup(surrogateProblem)

    # Set initial swarm positions
    initialPop = Population()
    for x in swarmInitX:
        ind = Individual(X=x)
        initialPop = Population.merge(initialPop, Population([ind]))

    # Evaluate initial swarm with surrogate
    pso.evaluator.eval(surrogateProblem, initialPop)

    # Set initial swarm population
    pso.pop = initialPop

    if disp:
        print("Starting main FSAPSO loop...")

    # Main FSAPSO loop
    xselected = np.array(out.sample[0 : out.nfev, :], copy=True)
    ySelected = np.array(out.fsample[0 : out.nfev], copy=True)
    while out.nfev < maxeval:  # and pso.has_next():
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % out.nfev)
            print("Best value: %f" % out.fx)

        # Reset improvement flag
        improvedThisIter = False

        # Update surrogate model
        surrogateModel.update(xselected, ySelected)

        # Get minimum of surrogate
        xselected = acquisitionFunc.optimize(
            surrogateModel, bounds, n=1, xbest=out.x, ybest=out.fx
        )

        # Compute f(xselected)
        ySelected = evaluate_and_log_point(fun, xselected, out)

        # determine best one of newly sampled points
        if ySelected[0] < out.fx:
            out.x[:] = xselected[0]
            out.fx = ySelected[0]

            # If Improved, update PSO's global best
            improvedThisIter = True
            pso.opt = Population.create(
                Individual(X=out.x, F=np.array([out.fx]))
            )

        if out.nfev < maxeval:
            # Update surrogate model
            surrogateModel.update(xselected, ySelected)

            # Create a filter to ensure points are far enough from samples
            filter = FarEnoughSampleFilter(surrogateModel.X, tol)

            # Update w value
            pso.w = 0.792 - (0.792 - 0.2) * out.nfev / maxeval

            # Update PSO velocities and positions
            swarm = pso.ask()

            # Evaluate particles with cheap surrogate
            pso.evaluator.eval(surrogateProblem, swarm)

            # Take swarm best
            fSurr = swarm.get("F")
            bestParticleIdx = np.argmin(fSurr)
            xselected = filter(swarm.get("X")[bestParticleIdx].reshape(1, -1))

            # If particle is far enough
            if len(xselected) > 0:
                # Evaluate with true function
                ySelected = evaluate_and_log_point(fun, xselected, out)

                # Update the particle's value in the swarm for PSO
                fUpdated = fSurr.copy()
                fUpdated[bestParticleIdx] = ySelected[0]
                swarm.set("F", fUpdated)

                # determine best one of newly sampled points
                if ySelected[0] < out.fx:
                    out.x[:] = xselected[0]
                    out.fx = ySelected[0]

                    # If Improved, update PSO's global best
                    improvedThisIter = True
                    pso.opt = Population.create(
                        Individual(X=out.x, F=np.array([out.fx]))
                    )
            else:
                ySelected = np.empty((0,))

            # If no improvement, evaluate particle with greatest uncertainty
            if not improvedThisIter and out.nfev < maxeval:
                # Update surrogate model
                surrogateModel.update(xselected, ySelected)

                scores = uncertainty_score(
                    swarm.get("X"), surrogateModel.X, surrogateModel.Y
                )
                ibest = np.argmax(scores)
                xselected = filter(swarm.get("X")[ibest].reshape(1, -1))

                if len(xselected) > 0:
                    # Evaluate with true function
                    ySelected = evaluate_and_log_point(fun, xselected, out)

                    # Update particle's fitness
                    fFinal = swarm.get("F")
                    fFinal[ibest] = ySelected[0]
                    swarm.set("F", fFinal)

                    # determine best one of newly sampled points
                    if ySelected[0] < out.fx:
                        out.x[:] = xselected[0]
                        out.fx = ySelected[0]

                        # If Improved, update PSO's global best
                        pso.opt = Population.create(
                            Individual(X=out.x, F=np.array([out.fx]))
                        )
                else:
                    ySelected = np.empty((0,))

        # Tell PSO the results
        pso.tell(infills=swarm)

        # Update out.nit
        out.nit = out.nit + 1

        # Call callback
        if callback is not None:
            callback(out)

        # Terminate if acquisition function has converged
        acquisitionFunc.update(out, surrogateModel)
        if acquisitionFunc.has_converged():
            break

    # Update output
    out.sample.resize(out.nfev, dim)
    out.fsample.resize(out.nfev)

    # Update surrogate model if it lives outside the function scope
    if return_surrogate:
        surrogateModel.update(xselected, ySelected)

    return out
