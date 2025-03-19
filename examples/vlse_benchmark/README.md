# VLSE benchmark

This example shows how to run the VLSE benchmark, that is based on optimization problems from the [Virtual Library of Simulation Experiments](https://www.sfu.ca/~ssurjano/optimization.html). The goal is to test each implementation for single-objective optimization in terms of accuracy. We use two metrics both based on the number of function evaluations $\ell_{\gamma,a}$ that an algorithm $a$ needs to solve a problem $\gamma$. Look at [vlse_bench.ipynb](vlse_bench.ipynb) for some results on experiments performed on Kestrel HPC machine at NREL.

The directory is organized as follows:

```bash
.
├── README.md  # This file
├── benchmark.py  # VLSE benchmark
├── vlse_bench.ipynb  # Notebook that shows results from simulations
├── vlse_bench.py  # Main script for running the benchmark
├── vlse_bench_run.sh  # Auxiliary script to run multiple algorithms and problems
├── vlse_bench_run_test1.py  # Auxiliary script that configures multiple runs of continuous optimization
└── vlse_bench_run_test2.py  # Auxiliary script that configures multiple runs of mixed-integer optimization
```
