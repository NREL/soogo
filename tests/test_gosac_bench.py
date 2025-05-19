import numpy as np
import pytest

# from soogo.optimize import gosac
# from soogo.rbf import RbfModel
import tests.gosac_benchmark as gosacbmk


# @pytest.mark.parametrize("problem", gosacbmk.gosac_p)
# def test_gosac(problem: gosacbmk.Problem) -> None:
#     np.random.seed(3)

#     dim = len(problem.bounds)
#     gdim = problem.gfun(
#         np.array([[problem.bounds[i][0] for i in range(dim)]])
#     ).shape[1]

#     maxeval = 50 * dim
#     s = [RbfModel(iindex=problem.iindex) for _ in range(gdim)]

#     res = gosac(
#         problem.objf,
#         problem.gfun,
#         problem.bounds,
#         maxeval,
#         surrogateModels=s,
#         disp=False,
#     )
#     assert isinstance(res.fx, np.ndarray)

#     # Print the results for debugging
#     print(res.x)
#     print(res.fx)

#     # A feasible solution was found
#     assert res.x.size > 0
#     assert res.fx.size > 0
#     assert np.all(res.fx[1:] <= 0)

#     # Check if the solution respect the integer constraints
#     for i in problem.iindex:
#         assert res.x[i] == np.round(res.x[i])

#     # Check if the solution is within the bounds
#     for i in range(dim):
#         assert problem.bounds[i][0] <= res.x[i] <= problem.bounds[i][1]

#     # Check if the solution is close to the known minimum
#     if problem.xmin is not None and problem.fmin is not None:
#         if problem.fmin != 0:
#             assert (res.fx[0] - problem.fmin) / np.abs(problem.fmin) <= 1e-2
#         else:
#             assert (res.fx[0] - problem.fmin) <= 1e-6


@pytest.mark.parametrize("problem", gosacbmk.gosac_p)
def test_benchmark(problem: gosacbmk.Problem) -> None:
    if problem.xmin is not None and problem.fmin is not None:
        print(problem.xmin)
        print(problem.fmin)
        print(problem.objf(np.asarray([problem.xmin]))[0])
        print(problem.gfun(np.asarray([problem.xmin]))[0])

        if problem.fmin != 0:
            assert (
                abs(problem.objf(np.asarray([problem.xmin]))[0] - problem.fmin)
                / abs(problem.fmin)
                <= 1e-2
            )
        else:
            assert problem.objf(np.asarray([problem.xmin]))[0] == problem.fmin

        assert np.all(problem.gfun(np.asarray([problem.xmin]))[0] <= 1e-2)


# if __name__ == "__main__":
#     np.random.seed(3)
#     test_gosac(gosacbmk.gosac_p[1])
