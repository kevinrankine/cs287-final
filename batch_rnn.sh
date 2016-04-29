#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 0-10:00
#SBATCH --job-name=LSTMEncoder
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th main.lua -model rnn -nepochs 5 -eta 1e-3 -d_hid 100 -cuda 1
