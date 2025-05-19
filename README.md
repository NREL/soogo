[![Python39](https://img.shields.io/badge/python-3.9-green.svg)](https://github.com/NREL/soogo/actions/workflows/python-package.yml)
[![Python310](https://img.shields.io/badge/python-3.10-green.svg)](https://github.com/NREL/soogo/actions/workflows/python-package.yml)
[![Python311](https://img.shields.io/badge/python-3.11-green.svg)](https://github.com/NREL/soogo/actions/workflows/python-package.yml)
[![Python312](https://img.shields.io/badge/python-3.12-green.svg)](https://github.com/NREL/soogo/actions/workflows/python-package.yml)

# soogo

Surrogate-based 0-th Order Global Optimization for black-box problems.

## Current functionality

<!--- Table below --->

| Optimization algorithm     | Description                                                                                                                                                                                                                                                               | Tags                                           |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `surrogate_optimization()` | Minimize a scalar function using a surrogate and an acquisition function based on [(Björkman & Holmström; 2000)][BjoHol2000] and [(Müller; 2016)][Muller2016].                                                                                                            | `mixed-integer`                                |
| `multistart_msrs()`        | Multistart Local Metric Stochastic Response Surface (LMSRS) [(Regis & Shoemaker; 2007)][RegSho2007]. Applies a derivative-free local search algorithm to obtain new samples. Restarts the surrogate model with new design points whenever the local search has converged. | `multi-start`, `RBF`                           |
| `dycors()`                 | Dynamic Coordinate Search (DYCORS) [(Regis & Shoemaker; 2012)][RegSho2012]. Acquisition cycles between global and local search. Uses the DDS search from [(Tolson & Shoemaker; 2007)][TolSho2007] to generate pools of candidates.                                        | `mixed-integer`, `RBF`                         |
| `cptv()`                   | Minimize a scalar function using rounds of coordinate perturbation (CP) and target value (TV) acquisition functions [(Müller; 2016)][Muller2016]. Derivative-free local search is used to improve a prospective global minimum                                            | `mixed-integer`, `RBF`                         |
| `socemo()`                 | Surrogate-based optimization of computationally expensive multiobjective problems (SOCEMO) [(Müller; 2017a)][Muller2017a].                                                                                                                                                | `multi-objective`, `mixed-integer`, `RBF`      |
| `gosac()`                  | Global optimization with surrogate approximation of constraints (GOSAC) [(Müller; 2017b)][Muller2017b].                                                                                                                                                                   | `mixed-integer`, `black-box-constraint`, `RBF` |
| `bayesian_optimization()`  | Bayesian optimization with dispersion-enhanced expected improvement acquisition [(Müller; 2024)][Muller2024].                                                                                                                                                             | `GP`, `batch`                                  |

<!--- Table below --->

| Acquisition function     | Description                                                                                                                                                                                                                                                                                               |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `WeightedAcquisition`    | Weighted acquisition function based on the predicted value and distance to the nearest sample [(Regis & Shoemaker; 2012)][RegSho2012]. Used in `multistart_msrs()`, `dycors()`, and in the CP step from `cptv()`. It uses average values for the multi-objective scenario [(Müller; 2017a)][Muller2017a]. |
| `TargetValueAcquisition` | Target value acquisition based from [(Gutmann; 2001)][Gut2001]. Used in the TV step from `cptv()`. Cycles through target values as in [(Björkman & Holmström; 2000)][BjoHol2000]. For batched acquisition, uses the strategy from [(Müller; 2016)][Muller2016] to avoid duplicates.                       |
| `MinimizeSurrogate`      | Sample at the local minimum of the surrogate model [(Müller; 2016)][Muller2016]. The original method, Multi-Level Single-Linkage (MLSL), is described in [(Rinnooy Kan & Timmer; 1987)][RinTim1987].                                                                                                      |
| `MaximizeEI`             | Maximize the expected improvement acquisition function for Gaussian processes. Use the dispersion-enhanced strategy from [(Müller; 2024)][Muller2024] for batch sampling.                                                                                                                                 |
| `ParetoFront`            | Sample at the Pareto front of the multi-objective surrogate model to fill gaps in the surface [(Müller; 2017a)][Muller2017a].                                                                                                                                                                             |
| `MinimizeMOSurrogate`    | Obtain pareto-optimal sample points for the multi-objective surrogate model [(Müller; 2017a)][Muller2017a].                                                                                                                                                                                               |
| `GosacSample`            | Minimize a function with surrogate constraints to obtain a single new sample point [(Müller; 2017b)][Muller2017b].                                                                                                                                                                                        |

[BjoHol2000]: https://doi.org/10.1023/A:1011584207202
[Muller2016]: https://doi.org/10.1007/s11081-015-9281-2
[RegSho2007]: https://doi.org/10.1287/ijoc.1060.0182
[RegSho2012]: https://doi.org/10.1080/0305215X.2012.687731
[Muller2017a]: https://doi.org/10.1287/ijoc.2017.0749
[Muller2017b]: https://doi.org/10.1007/s10898-017-0496-y
[Muller2024]: https://doi.org/10.1002/qre.3245
[TolSho2007]: https://doi.org/10.1029/2005WR004723
[Gut2001]: https://doi.org/10.1023/A:1011255519438
[RinTim1987]: https://doi.org/10.1007/BF02592071

## Installation

Use PyPI to install this package:

```sh
pip install soogo
```

See other installation methods below.

### Binaries

The binaries for the latest version are available at https://github.com/NREL/soogo/releases/latest. They can be installed through standard installation, e.g.,

using pip (https://pip.pypa.io/en/stable/cli/pip_install/):

```sh
pip install git+https://github.com/NREL/soogo.git#egg=soogo
```

### From source

This package contains a [pyproject.toml](pyproject.toml) with the list of requirements and dependencies (More about `pyproject.toml` at https://packaging.python.org/en/latest/specifications/pyproject-toml/). With the source downloaded to your local machine, use `pip install [soogo/source/directory]`.

### For developers

This project is configured to use the package manager [pdm](https://pdm-project.org/en/stable/). With pdm installed, run `pdm install` at the root of this repository to install the dependencies. The file [pyproject.toml](pyproject.toml) has the list of dependencies and configurations for the project.

## Documentation

This project uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate the documentation. The latest documentation is available at https://nrel.github.io/soogo. To generate the documentation locally, run `make html` in the `docs` directory. The homepage of the documentation will then be found at `docs/_build/html/index.html`.

## Testing

This project uses [pytest](https://docs.pytest.org/en/stable/) to run the tests. To run the tests, run `pytest` at the root of this repository. Run `pytest --help` to see the available options.

## Contributing

Please, read the [contributing guidelines](CONTRIBUTING.md) before contributing to this project.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

---

_NREL Software Record number: SWR-24-57_
