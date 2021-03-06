#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-01:30
#SBATCH --job-name=MEAN_TITLE
#SBATCH --output=logs/mean_title.log
#SBATCH --error=logs/fuck.fuck
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th main.lua -model rnn -cuda 1 -eta 1e-3 -d_hid 280 -nepochs 5 -margin 0.02 -nbatches 16 -dropout 0.1 -pool mean -to_file gru.dat -body 0

