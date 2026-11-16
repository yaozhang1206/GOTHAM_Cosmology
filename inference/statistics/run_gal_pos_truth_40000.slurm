#!/bin/bash
#SBATCH --job-name=gal_pos_truth_40000
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=3
#SBATCH --mem=70g
#SBATCH --time=10:00:00
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=%x.o%j
#SBATCH --error=%x.o%j

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate hod

echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "Start time: $(date)"

# Truth galaxy RSD positions for the test sims in quijote_saved_idx.txt (100 sims x 10 HODs), noise=0.0
srun --mpi=pmi2 python gal_pos_ext_40000_truth.py 0.0

echo "End time: $(date)"
