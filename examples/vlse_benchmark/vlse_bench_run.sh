#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=32G
#SBATCH --job-name=vlse-bench-run
#SBATCH --output=vlse-bench-run.%j.out  # %j will be replaced with the job ID

if command -v module &> /dev/null; then
    module load conda
    conda activate $CONDAENV
fi

if [ $# -eq 0 ]; then
    python vlse_bench.py
elif [ $# -eq 1 ]; then
    python vlse_bench.py -a "$1"
elif [ $# -eq 2 ]; then
    python vlse_bench.py -a "$1" -p "$2"
else
    python vlse_bench.py -a "$1" -p "$2" --bounds ${@:3}
fi