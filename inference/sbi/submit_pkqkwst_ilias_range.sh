#!/bin/bash
# Submit the full ilias HPO pipeline for combined Pk + Bk/Qk + WST (no nbar),
# restricted to a single Om x sigma8 training/evaluation box.
# Paper Fig. 7 row 2 is:
#
#   bash submit_pkqkwst_ilias_range.sh 0.6 0.0 0.15 0.45 0.65 0.95
#   -> .../all_test/ilias_runs/qk_kmax0.6_wstgrid256_avgangles_noise0.0_Om0.15-0.45_s80.65-0.95
#
# Usage:
#   bash submit_pkqkwst_ilias_range.sh <kmax> <noise> <om_lo> <om_hi> <s8_lo> <s8_hi> [bq] [grid] [average_angles]
#   bq             : bk (bispectrum) or qk (reduced bispectrum, default)
#   grid           : WST grid size (default 256)
#   average_angles : 1 = average WST over orientation bins (default, recommended)
#                    0 = keep full angular info
# Example:
#   bash submit_pkqkwst_ilias_range.sh 0.6 0.0 0.15 0.45 0.65 0.95
#   bash submit_pkqkwst_ilias_range.sh 0.6 0.0 0.15 0.45 0.65 0.95 bk 256 1
#   bash submit_pkqkwst_ilias_range.sh 0.6 0.0 full          # full prior, no box
#   MODEL_SUFFIX=_hodfix bash submit_pkqkwst_ilias_range.sh 0.6 0.0 full
#
# om_lo = "full" drops the box restriction (the remaining range args are then
# not needed; bq/grid/average_angles shift to positions 4-6).
# MODEL_SUFFIX (env, default empty) is appended to the run directory name so a
# rerun does not reuse an existing directory (preprocess skips if x_train.npy
# already exists).
#
# Pipeline:
#   Job 1 – preprocess  (single task)
#   Job 2 – optuna      (40-task array, 5 trials/worker = 200 total)
#            depends on Job 1 completing successfully
#   Job 3 – retrain     (N_TOP_NETS-task array, one net per worker)
#            depends on ALL Job 2 array tasks completing
#   Job 4 – validate    (single task)
#            depends on ALL Job 3 array tasks completing

set -euo pipefail

USAGE="Usage: bash submit_pkqkwst_ilias_range.sh <kmax> <noise> <om_lo> <om_hi> <s8_lo> <s8_hi> [bq] [grid] [average_angles]"
KMAX=${1:?$USAGE}
NOISE=${2:?$USAGE}
OM_LO=${3:?$USAGE}
if [ "$OM_LO" = "full" ]; then
    FULL_RANGE=1
    OM_HI=""; S8_LO=""; S8_HI=""
    BQ=${4:-qk}
    GRID=${5:-256}
    AVG_ANGLES=${6:-1}
else
    FULL_RANGE=0
    OM_HI=${4:?$USAGE}
    S8_LO=${5:?$USAGE}
    S8_HI=${6:?$USAGE}
    BQ=${7:-qk}
    GRID=${8:-256}
    AVG_ANGLES=${9:-1}   # 1 = on (default), 0 = off
fi
MODEL_SUFFIX=${MODEL_SUFFIX:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_inference_pkqkwst_ilias_range.py"
LOG_DIR="$SCRIPT_DIR/logs_ilias"

# model_dir encodes whether angle averaging is on (must match the python
# script's own default-naming logic)
if [ "$AVG_ANGLES" = "1" ]; then
    AVG_TAG="_avgangles"
    AVG_FLAG="--average_angles"
else
    AVG_TAG=""
    AVG_FLAG="--no-average_angles"
fi
if [ "$FULL_RANGE" = "1" ]; then
    RANGE_TAG="_fullrange"
    RANGE_FLAGS=""
else
    RANGE_TAG="_Om${OM_LO}-${OM_HI}_s8${S8_LO}-${S8_HI}"
    RANGE_FLAGS="--om_range $OM_LO $OM_HI --sigma8_range $S8_LO $S8_HI"
fi
MODEL_DIR="/u/yzhang116/NN/inference/all_test/ilias_runs/${BQ}_kmax${KMAX}_wstgrid${GRID}${AVG_TAG}_noise${NOISE}${RANGE_TAG}${MODEL_SUFFIX}"

N_ARRAYS=40
TRIALS_PER_WORKER=5   # 40 × 5 = 200 total trials
N_TOP_NETS=10
BACKEND="lampe"

mkdir -p "$LOG_DIR"

# ── shared SBATCH header fragments ───────────────────────────────────────────
COMMON_SBATCH="
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8g
#SBATCH --partition=cpu
#SBATCH --account=bdne-delta-cpu
#SBATCH --output=${LOG_DIR}/%x_%A_%a.o
#SBATCH --error=${LOG_DIR}/%x_%A_%a.o
"

CONDA_SETUP="
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate nn
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=\$SLURM_CPUS_PER_TASK
echo \"Start time: \$(date)\"
which python
"

PYTHON_COMMON="python $PYTHON_SCRIPT $KMAX $NOISE \
    --bq $BQ \
    --grid $GRID \
    $AVG_FLAG \
    $RANGE_FLAGS \
    --model_dir $MODEL_DIR \
    --n_top_nets $N_TOP_NETS \
    --backend $BACKEND"

JOB_TAG="kmax${KMAX}_noise${NOISE}${RANGE_TAG}${MODEL_SUFFIX}"

# ── Job 1: preprocess ─────────────────────────────────────────────────────────
JID1=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=all_pre_${JOB_TAG}
#SBATCH --time=00:30:00
$COMMON_SBATCH
set -e
$CONDA_SETUP
$PYTHON_COMMON --stage preprocess
echo "End time: \$(date)"
EOF
)
echo "Submitted preprocess   job $JID1"

# ── Job 2: optuna HPO (40-task array) ─────────────────────────────────────────
JID2=$(sbatch --parsable --dependency=afterok:$JID1 <<EOF
#!/bin/bash
#SBATCH --job-name=all_optuna_${JOB_TAG}
#SBATCH --array=0-$((N_ARRAYS - 1))
#SBATCH --time=24:00:00
$COMMON_SBATCH
set -e
$CONDA_SETUP
sleep \$((SLURM_ARRAY_TASK_ID * 2))
$PYTHON_COMMON --stage optuna --n_trials $TRIALS_PER_WORKER
echo "End time: \$(date)"
EOF
)
echo "Submitted optuna array job $JID2 (${N_ARRAYS} workers × ${TRIALS_PER_WORKER} trials)"

# ── Job 3: retrain (array, one net per worker) ────────────────────────────────
JID3=$(sbatch --parsable --dependency=afterok:$JID2 <<EOF
#!/bin/bash
#SBATCH --job-name=all_retrain_${JOB_TAG}
#SBATCH --array=0-$((N_TOP_NETS - 1))
#SBATCH --time=24:00:00
$COMMON_SBATCH
set -e
$CONDA_SETUP
$PYTHON_COMMON --stage retrain --net_index \$SLURM_ARRAY_TASK_ID
echo "End time: \$(date)"
EOF
)
echo "Submitted retrain array job $JID3 (${N_TOP_NETS} workers, 1 net each)"

# ── Job 4: validate ───────────────────────────────────────────────────────────
JID4=$(sbatch --parsable --dependency=afterok:$JID3 <<EOF
#!/bin/bash
#SBATCH --job-name=all_validate_${JOB_TAG}
#SBATCH --time=02:00:00
$COMMON_SBATCH
set -e
$CONDA_SETUP
$PYTHON_COMMON --stage validate
echo "End time: \$(date)"
EOF
)
echo "Submitted validate     job $JID4"

echo ""
echo "Pipeline submitted for kmax=$KMAX  noise=$NOISE  bq=$BQ  grid=$GRID  average_angles=$AVG_ANGLES  Om=[$OM_LO,$OM_HI]  sigma8=[$S8_LO,$S8_HI]"
echo "  Model dir : $MODEL_DIR"
echo "  Log dir   : $LOG_DIR"
echo "  Job chain : $JID1 → $JID2 → $JID3 → $JID4"
echo ""
echo "Monitor with:  squeue -u \$USER"
