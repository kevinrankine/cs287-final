#!/bin/bash

#SBATCH --mail-type=END
#SBATCH --mail-user=rankine@college.harvard.edu
#SBATCH -t 1-00:00
#SBATCH --job-name=LSTMEncoder
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

th main.lua -model cbow -cuda 1 -eta 5e-4 -d_hid 200 -nepochs 5 -margin 0.2
