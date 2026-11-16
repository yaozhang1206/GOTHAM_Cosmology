#!/bin/bash
# Submit the full ilias HPO pipeline (Pk + Bk/Qk + nbar) for a (kmax, noise, bq) triple.
#
# Usage:
#   bash submit_ilias.sh <kmax> <noise> [bq]
#   bq : bk (bispectrum, default) or qk (reduced bispectrum)
# Example:
#   bash submit_ilias.sh 0.4 0.8
#   bash submit_ilias.sh 0.4 0.8 qk
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

KMAX=${1:?Usage: bash submit_pkqk_ilias.sh <kmax> <noise> [bq]}
NOISE=${2:?Usage: bash submit_pkqk_ilias.sh <kmax> <noise> [bq]}
BQ=${3:-qk}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/run_inference_pkqk_ilias.py"
LOG_DIR="$SCRIPT_DIR/logs_ilias"
MODEL_DIR="/u/yzhang116/NN/inference/bk_test/ilias_runs/${BQ}_kmax${KMAX}_noise${NOISE}_ext_200c_40000"

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
#SBATCH --mem=2g
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
    --model_dir $MODEL_DIR \
    --n_top_nets $N_TOP_NETS \
    --backend $BACKEND"

# ── Job 1: preprocess ─────────────────────────────────────────────────────────
JID1=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=bk_pre_kmax${KMAX}_noise${NOISE}
#SBATCH --time=00:30:00
$COMMON_SBATCH
$CONDA_SETUP
$PYTHON_COMMON --stage preprocess
echo "End time: \$(date)"
EOF
)
echo "Submitted preprocess   job $JID1"

# ── Job 2: optuna HPO (40-task array) ─────────────────────────────────────────
JID2=$(sbatch --parsable --dependency=afterok:$JID1 <<EOF
#!/bin/bash
#SBATCH --job-name=bk_optuna_kmax${KMAX}_noise${NOISE}
#SBATCH --array=0-$((N_ARRAYS - 1))
#SBATCH --time=24:00:00
$COMMON_SBATCH
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
#SBATCH --job-name=bk_retrain_kmax${KMAX}_noise${NOISE}
#SBATCH --array=0-$((N_TOP_NETS - 1))
#SBATCH --time=24:00:00
$COMMON_SBATCH
$CONDA_SETUP
$PYTHON_COMMON --stage retrain --net_index \$SLURM_ARRAY_TASK_ID
echo "End time: \$(date)"
EOF
)
echo "Submitted retrain array job $JID3 (${N_TOP_NETS} workers, 1 net each)"

# ── Job 4: validate ───────────────────────────────────────────────────────────
JID4=$(sbatch --parsable --dependency=afterok:$JID3 <<EOF
#!/bin/bash
#SBATCH --job-name=bk_validate_kmax${KMAX}_noise${NOISE}
#SBATCH --time=02:00:00
$COMMON_SBATCH
$CONDA_SETUP
$PYTHON_COMMON --stage validate
echo "End time: \$(date)"
EOF
)
echo "Submitted validate     job $JID4"

echo ""
echo "Pipeline submitted for kmax=$KMAX  noise=$NOISE  bq=$BQ"
echo "  Model dir : $MODEL_DIR"
echo "  Log dir   : $LOG_DIR"
echo "  Job chain : $JID1 → $JID2 → $JID3 → $JID4"
echo ""
echo "Monitor with:  squeue -u \$USER"
