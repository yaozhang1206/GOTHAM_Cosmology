#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=generate_catalog
#SBATCH --partition=ghx4

#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=/u/yzhang116/logs/%x.o%j
#SBATCH --error=/u/yzhang116/logs/%x.o%j


module purge             
module load cudatoolkit/25.5_12.9

# <<< conda initialize <<<
source ~/.bashrc
conda activate nn_test

which python
which nvcc
which gcc
echo $CC

export CC=gcc
export CXX=g++

which python
which nvcc
which gcc
echo $CC

#srun hostname
python generate_catalog_fast.py
