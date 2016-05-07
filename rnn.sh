#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 1-00:00
#SBATCH --job-name=LSTMEncoder
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th main.lua -model rnn -cuda 1 -eta 1e-3 -d_hid 250 -nepochs 20 -margin 0.2 -nbatches 8
