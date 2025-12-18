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
from scipy.spatial.distance import cdist

from ..acquisition import (
    MaximizeDistance,
    MinimizeSurrogate,
    MultipleAcquisition,
)
from ..model import RbfModel, Surrogate
from .utils import OptimizeResult, evaluate_and_log_point, uncertainty_score
from ..integrations.pymoo import PymooProblem
from ..sampling import Sampler


def fsapso(
    fun,
    bounds,
    maxeval: int,
    *,
    surrogateModel: Optional[Surrogate] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    disp: bool = False,
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

    :return: The optimization result.


    References
    ----------
    .. [#] Li, F., Shen, W., Cai, X., Gao, L., & Gary Wang, G. 2020; A
        fast surrogate-assisted particle swarm optimization algorithm for
        computationally expensive problems. Applied Soft Computing, 92,
        106303. https://doi.org/10.1016/j.asoc.2020.106303
    """
    # Initialize parameters
    bounds = np.array(bounds)
    dim = len(bounds)

    lb = bounds[:, 0]
    ub = bounds[:, 1]
    vMax = 0.1 * (ub - lb)
    nSwarm = 20
    nInitialPts = min(max(dim, 20), maxeval)
    tol = np.min([np.sqrt(0.001**2 * dim), 5e-5 * dim * np.min(ub - lb)])

    # Initialize acquisition function(s)
    surrogateMinimizer = MultipleAcquisition(
        (
            MinimizeSurrogate(1000, rtol=1e-3),
            MaximizeDistance(rtol=1e-3),
        )
    )

    # Initialize surrogate
    if surrogateModel is None:
        surrogateModel = RbfModel()

    # Reserve space in the surrogate model
    surrogateModel.reserve(maxeval + surrogateModel.ntrain, dim)

    # Initialize output
    out = OptimizeResult(
        x=np.empty(dim),
        fx=np.inf,
        nit=0,
        nfev=surrogateModel.ntrain,
        sample=np.zeros((maxeval + surrogateModel.ntrain, dim)),
        fsample=np.zeros(maxeval + surrogateModel.ntrain),
    )

    if disp:
        print("Starting FSAPSO optimization...")

    # Initialize surrogate model
    if surrogateModel.ntrain == 0:
        nInitial = 0
        sampler = Sampler(nInitialPts)
        xInit = sampler.get_slhd_sample(bounds.tolist())

        if disp:
            print(f"Evaluating {len(xInit)} initial points for surrogate...")

        # Evaluate initial points
        for x0 in xInit:
            y0 = evaluate_and_log_point(fun, x0.reshape(1, -1), out)[0]

            if y0 < out.fx:
                out.x[:] = x0
                out.fx = y0

            if disp:
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

        # Build surrogate model
        surrogateModel.update(
            out.sample[0 : out.nfev], out.fsample[0 : out.nfev]
        )

        if disp:
            print(f"Built surrogate model with {surrogateModel.ntrain} points")

    else:
        # Initialize best point in output
        out.x = surrogateModel.X[np.argmin(surrogateModel.Y)]
        out.fx = np.min(surrogateModel.Y)

        out.sample[0 : surrogateModel.ntrain] = surrogateModel.X
        out.fsample[0 : surrogateModel.ntrain] = surrogateModel.Y
        nInitial = surrogateModel.ntrain

        if disp:
            print(
                f"Using pre-trained surrogate with {surrogateModel.ntrain} points"
            )

    # Select initial swarm
    if surrogateModel.ntrain >= nSwarm:
        # Take 20 best points as initial swarm
        bestIndices = np.argsort(surrogateModel.Y)[:nSwarm]
        swarmInitX = surrogateModel.X[bestIndices]

        if disp:
            print(f"Selected {nSwarm} best training points for initial swarm")

    else:
        # If not enough training data, use random sampling
        if disp:
            print(
                "Not enough training data for initial swarm. Using random sampling to increase population."
            )

        swarmSampler = Sampler(nSwarm - surrogateModel.ntrain)
        swarmInitX = swarmSampler.get_slhd_sample(bounds.tolist())
        swarmInitX = np.vstack((swarmInitX, surrogateModel.X))

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
    prevGlobalBest = out.fx

    while out.nfev < maxeval + nInitial:  # and pso.has_next():
        improvedThisIter = False

        # Get minimum of surrogate
        xMin = surrogateMinimizer.optimize(surrogateModel, bounds, n=1)[0]

        # Check xMin is at least tol away from existing points
        if np.min(cdist(xMin.reshape(1, -1), out.sample[: out.nfev])) > tol:
            # Evaluate minimum with true objective
            fMin = evaluate_and_log_point(fun, xMin.reshape(1, -1), out)[0]

            if fMin < out.fx:
                out.x[:] = xMin
                out.fx = fMin

            if disp:
                print("fEvals: %d" % out.nfev)
                print("Best value: %f" % out.fx)

            # Update surrogate model with new point
            surrogateModel.update(xMin.reshape(1, -1), fMin)

            # If Improved, update PSO's global best
            if fMin < prevGlobalBest:
                improvedThisIter = True
                prevGlobalBest = fMin

                # Update PSO's global best
                pso.opt = Population.create(
                    Individual(X=xMin, F=np.array([fMin]))
                )

        # Update w value
        pso.w = 0.792 - (0.792 - 0.2) * out.nfev / maxeval

        # Update PSO velocities and positions
        swarm = pso.ask()

        # Evaluate particles with cheap surrogate
        pso.evaluator.eval(surrogateProblem, swarm)

        if out.nfev < maxeval:
            # Take swarm best
            fSurr = swarm.get("F")
            bestParticleIdx = np.argmin(fSurr)
            xBestParticle = swarm.get("X")[bestParticleIdx]

            # Evaluate best particle
            if (
                np.min(
                    cdist(xBestParticle.reshape(1, -1), out.sample[: out.nfev])
                )
                > tol
            ):
                fBestParticle = evaluate_and_log_point(
                    fun, xBestParticle.reshape(1, -1), out
                )[0]

                if fBestParticle < out.fx:
                    out.x[:] = xBestParticle
                    out.fx = fBestParticle

                if disp:
                    print("fEvals: %d" % out.nfev)
                    print("Best value: %f" % out.fx)

                # Update surrogate with true evaluation
                surrogateModel.update(
                    xBestParticle.reshape(1, -1), fBestParticle
                )

                # Update the particle's value in the swarm for PSO
                fUpdated = fSurr.copy()
                fUpdated[bestParticleIdx] = fBestParticle
                swarm.set("F", fUpdated)

                # Check if this improved global best
                if fBestParticle < prevGlobalBest:
                    improvedThisIter = True
                    prevGlobalBest = fBestParticle

                    # Update PSO's global best
                    pso.opt = Population.create(
                        Individual(
                            X=xBestParticle, F=np.array([fBestParticle])
                        )
                    )

        # If no improvement, evaluate particle with greatest uncertainty
        if not improvedThisIter and out.nfev < maxeval:
            scores = uncertainty_score(
                swarm.get("X"), surrogateModel.X, surrogateModel.Y
            )
            xMostUncertain = swarm.get("X")[np.argmax(scores)]

            if (
                np.min(
                    cdist(
                        xMostUncertain.reshape(1, -1), out.sample[: out.nfev]
                    )
                )
                > tol
            ):
                fMostUncertain = evaluate_and_log_point(
                    fun, xMostUncertain.reshape(1, -1), out
                )[0]

                if fMostUncertain < out.fx:
                    out.x[:] = xMostUncertain
                    out.fx = fMostUncertain

                if disp:
                    print("fEvals: %d" % out.nfev)
                    print("Best value: %f" % out.fx)

                # Update surrogate
                surrogateModel.update(
                    xMostUncertain.reshape(1, -1), fMostUncertain
                )

                # Update particle's fitness
                fFinal = swarm.get("F")
                fFinal[np.argmax(scores)] = fMostUncertain
                swarm.set("F", fFinal)

                # Check if this improved global best
                if fMostUncertain < prevGlobalBest:
                    prevGlobalBest = fMostUncertain

                    # Update PSO's global best
                    pso.opt = Population.create(
                        Individual(
                            X=xMostUncertain, F=np.array([fMostUncertain])
                        )
                    )

        # Tell PSO the results
        pso.tell(infills=swarm)

        # Call callback
        if callback is not None:
            callback(out)

    # Remove empty if PSO terminates before maxevals
    out.sample = out.sample[: out.nfev]
    out.fsample = out.fsample[: out.nfev]

    return out
