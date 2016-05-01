#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 1-00:00
#SBATCH --job-name=LSTMEncoder
#SBATCH --output=slurm2.out
#SBATCH --error=slurm2.err
#SBATCH --partition=holyseasgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th main.lua -model rnn -nepochs 10 -eta 1e-3 -d_hid 200 -cuda 1
