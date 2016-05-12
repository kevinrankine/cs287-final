#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-06:00
#SBATCH --job-name=MEAN_BODY
#SBATCH --output=logs/mean_body.log
#SBATCH --error=logs/fuck.fuck
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th main.lua -model rnn -cuda 1 -eta 1e-3 -d_hid 280 -nepochs 15 -margin 0.02 -nbatches 8 -dropout 0.1 -pool mean -to_file gru.dat -body 1

