#!/bin/bash
# Pk + Bk/Qk of TRUTH (Quijote N-body) galaxy catalogs, test sims in quijote_saved_idx.txt only, noise=0.0
#SBATCH --job-name=pk_bk_truth_40000
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

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Python: $(which python)   Start: $(date)"
srun --mpi=pmi2 python gal_pk_bk_ext_40000_truth.py 0.0
echo "End time: $(date)"
