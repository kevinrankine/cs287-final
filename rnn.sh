#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-00:45
#SBATCH --job-name=RNNEncoder
#SBATCH --output=rnn_output.log
#SBATCH --error=rnn_error.log
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

time th main.lua -model rnn -cuda 1 -eta 1e-3 -d_hid 250 -nepochs 5 -margin 0.2 -nbatches 16 -dropout 0.1
