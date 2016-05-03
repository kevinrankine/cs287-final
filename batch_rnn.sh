#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 1-00:00
#SBATCH --job-name=LSTMEncoder
#SBATCH --output=slurm_fuck.out
#SBATCH --error=slurm_fuck.err
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

#th main.lua -model rnn -nepochs 10 -eta 1e-3 -d_hid 200 -cuda 1
#th main.lua -model rnn -cuda 1 -eta 5e-4 -d_hid 150 -nepochs 5 -margin 0.3
th main.lua -model rnn -cuda 1 -eta 1e-3 -d_hid 250 -nepochs 10 -margin 0.2
