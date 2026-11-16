#!/bin/bash
# RSD galaxy positions of MOCK catalogs for LH 2000-3999 (input to the WST code, which runs on DeltaAI)
#SBATCH --job-name=gal_pos_mock_40000
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
srun --mpi=pmi2 python gal_pos_ext_40000_mock.py 2000 4000 0.0
echo "End time: $(date)"
