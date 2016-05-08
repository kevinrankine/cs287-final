#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-01:30
#SBATCH --job-name=RNNEncoder
#SBATCH --output=logs/rnn_output.log
#SBATCH --error=logs/rnn_error.log
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th ../main.lua -model rnn -cuda 1 -eta 3e-4 -d_hid 250 -nepochs 10 -margin 0.2 -nbatches 16 -dropout 0.1
