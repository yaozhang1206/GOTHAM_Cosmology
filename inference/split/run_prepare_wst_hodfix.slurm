#!/bin/bash
#SBATCH --job-name=wst_split_hodfix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=01:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate nn
python -u prepare_wst_train_test_40000_hodfix.py 0.0
