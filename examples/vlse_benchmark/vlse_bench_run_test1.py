#!/usr/bin/env python
import subprocess
import sys

# All:
myRfuncs = (
    "branin",
    "hart3",
    "hart6",
    "shekel",
    "ackley",
    "levy",
    "powell",
    "michal",
    "spheref",
    "rastr",
    "mccorm",
    "bukin6",
    "camel6",
    "crossit",
    "drop",
    "egg",
    "griewank",
    "holder",
    "levy13",
)
algorithms = ("SRS", "DYCORS", "CPTV", "CPTVl")

args = sys.argv[1:]
for func in myRfuncs:
    for a in algorithms:
        print(func)
        print(a)
        subprocess.call(["sbatch"] + args + ["./vlse_bench_run.sh", a, func])
