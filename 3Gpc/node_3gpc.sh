#!/bin/bash
# One copy per node (srun --ntasks-per-node=1): the JAX producer (one process, 4 GPUs) + 4 emulator workers.
# Needs: CIDS, CAT_DIR, LOGP (log prefix), DISCODJ_DIR (multi-GPU DISCO-DJ checkout).  Called by job_3gpc.sbatch.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DISCODJ_DIR=${DISCODJ_DIR:-/u/yzhang116/DISCO-DJ-multigpu}
NODE=${SLURM_NODEID:-0}
SHM=/dev/shm/gen3gpc_${SLURM_JOB_ID}
mkdir -p $SHM; trap 'rm -rf $SHM' EXIT
echo "node $NODE $(hostname -s): $(df -h /dev/shm | tail -1)"

# ---- producer: DISCO-DJ + GPU fields on the 4 local GPUs (conda pm); jax.distributed over the srun tasks
(
  export PYTHONPATH_EXTRA=$HERE
  # modules, OFI (2 nodes), XLA env, $PY; sets PYTHONPATH=$DISCODJ_DIR/src:$PYTHONPATH_EXTRA (discodj_dist)
  source $DISCODJ_DIR/multigpu/env/setup_delta.sh >/dev/null
  export DISCO_PM_FFT_BACKEND=JAX_RFFT DISCO_HALO_SIZE=${DISCO_HALO_SIZE:-96} OMP_NUM_THREADS=32
  /usr/bin/time -f "PRODUCER MAXRSS_KB %M WALL %e" $PY -u $HERE/run_3gpc.py --cids $CIDS --shm $SHM \
     --cat-dir $CAT_DIR > ${LOGP}_node${NODE}_producer.log 2>&1
  echo "producer exit $?" >> ${LOGP}_node${NODE}_producer.log
) &
PROD=$!

# ---- 4 emulator workers (conda nn_test), one per local GPU
for j in 0 1 2 3; do
(
  module purge; module load cudatoolkit/25.5_12.9
  source ~/.bashrc >/dev/null 2>&1; conda activate nn_test
  export CC=gcc CXX=g++ CUDA_VISIBLE_DEVICES=$j OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=8
  export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_${SLURM_JOB_ID}_$j
  python -u $HERE/emu_worker3.py --local $j --shm $SHM --cat-dir $CAT_DIR > ${LOGP}_node${NODE}_worker$j.log 2>&1
  echo "worker $j exit $?" >> ${LOGP}_node${NODE}_worker$j.log
) &
done
# sample host memory + /dev/shm use every 10 s (peak check)
( while kill -0 $PROD 2>/dev/null; do echo "$(date +%T) $(free -g | awk '/Mem/{print "used "$3" GB"}') shm $(du -sh $SHM 2>/dev/null | cut -f1)"; sleep 10; done ) > ${LOGP}_node${NODE}_mem.log &
wait $PROD
wait
