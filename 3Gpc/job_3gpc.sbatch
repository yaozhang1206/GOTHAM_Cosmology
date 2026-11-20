#!/bin/bash
#SBATCH --account=bdne-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --job-name=gen3gpc
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Fused 3 Gpc/h halo catalogs: DISCO-DJ (1536^3 / 3072^3, 8 GPUs) -> GPU fields in /dev/shm -> emulator -> catalog.
# Submit from this directory (3Gpc/):
#   CIDS="663 188" [CAT_DIR=...] [DISCODJ_DIR=...] sbatch job_3gpc.sbatch     (or CID_FILE=list.txt)
# DISCODJ_DIR: checkout of https://github.com/yaozhang1206/DISCO-DJ, branch multigpu-kit (multi-GPU DISCO-DJ).
set -u
HERE=${SLURM_SUBMIT_DIR}
export CAT_DIR=${CAT_DIR:-/work/hdd/bdne/yzhang116/gen3gpc/catalogs}
export DISCODJ_DIR=${DISCODJ_DIR:-/u/yzhang116/DISCO-DJ-multigpu}
[ -n "${CID_FILE:-}" ] && CIDS=$(cat $CID_FILE)
export CIDS
export LOGP=$HERE/logs/gen3gpc_${SLURM_JOB_ID}
mkdir -p $HERE/logs $CAT_DIR
echo "cids: $CIDS  cat_dir: $CAT_DIR  discodj: $DISCODJ_DIR  $(date)"
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_JOB_NUM_NODES --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=288 \
     bash $HERE/node_3gpc.sh
grep -h "BOX\|SKIP\|exit\|Error\|Traceback\|MAXRSS" ${LOGP}_node*_producer.log | cut -c1-700
grep -h "CATALOG\|worker . box\|exit\|Error\|Traceback" ${LOGP}_node*_worker*.log | cut -c1-400
echo "DONE $(date)"
