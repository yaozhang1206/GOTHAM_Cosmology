#!/bin/bash
# Pk + Bk/Qk of MOCK (emulator) galaxy catalogs, all 4000 cosmologies x 10 HODs, noise=0.0
#SBATCH --job-name=pk_bk_mock_40000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=2
#SBATCH --mem=60g
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
srun --mpi=pmi2 python gal_pk_bk_ext_40000_mock.py 0 4000 0.0
echo "End time: $(date)"
