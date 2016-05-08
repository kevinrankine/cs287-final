#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-00:30
#SBATCH --job-name=CBOWEncoder
#SBATCH --output=logs/cbow_output.log
#SBATCH --error=logs/cbow_error.log
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th ../main.lua -model cbow -cuda 1 -eta 1e-3 -nepochs 5 -margin 0.2 -nbatches 16
