#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-03:00
#SBATCH --job-name=CBOWEncoder
#SBATCH --output=logs/output.log
#SBATCH --error=logs/error.log
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th main.lua -model cbow++ -cuda 1 -eta 1e-3 -nepochs 15 -margin 0.05 -nbatches 16 -body 0
