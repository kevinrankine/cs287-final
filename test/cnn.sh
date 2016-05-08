#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-01:30
#SBATCH --job-name=CNNEncoder
#SBATCH --output=logs/cnn_output.log
#SBATCH --error=logs/cnn_error.log
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th ../main.lua -model cnn -cuda 1 -eta 3e-4 -d_hid 700 -nepochs 10 -margin 0.2 -nbatches 16 -kernel_width 3 -dropout 0.1
