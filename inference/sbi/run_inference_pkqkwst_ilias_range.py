"""
Ilias-based hyperparameter optimization for Pk + Bk/Qk + WST NPE inference
(no nbar), restricted to a caller-specified Omega_m x sigma_8 training/
evaluation box.

Unrestricted variant: NN/inference/all_test/codes/run_inference_all_ilias_kcpi.py.
The box restriction is the only difference:

--om_range LO HI / --sigma8_range LO HI: if given, every split (train+val,
the internal 1000-pt test used for Optuna scoring and retrain's
log_prob_test, and both validation_mock/validation_quijote sets) is
filtered to keep only rows with Omega_m in [LO, HI] and/or sigma_8 in
[LO, HI]. Pk, Bk/Qk and WST dill files are row-aligned (same theta_all
ordering per split; see the module docstring of the combined driver and the
project's theta_all row-alignment invariant), so the mask computed from
Pk's theta_all is applied identically to the Bk/Qk and WST arrays. The NPE
prior is built at runtime from theta_train.min/max, so restricting the
training rows automatically shrinks the prior to the box -- no separate
prior change needed. When a box leaves fewer rows than the usual subsample
size (1000 for the internal test, 500 for validation), the subsample size
is shrunk to whatever is available (logged explicitly); for validation,
when a box is active the default subsample cap is dropped entirely and
every in-box row is used.

Summary statistic = [signed-log P0 | P2/P0 | P4/P0 | Bk or Qk | signed-log WST (angular)]
loaded from three separate dill files per split:
  Pk_nbar_NGP_galnoise_{noise}_200c_*.dill        — power spectrum multipoles
  Bk_NGP_galnoise_{noise}_200c_*.dill             — bispectrum (bk) or reduced bispectrum (qk)
  WST_nbar_angular_NGP_galnoise_{noise}_*.dill    — wavelet scattering transform coefficients

Use --bq bk  to use the bispectrum (bk_all_*)
Use --bq qk  to use the reduced bispectrum (qk_all_*)
--grid sets the WST field-resolution grid (data key wst_mock_{grid}_all)
--average_angles averages the WST coefficients over orientation bins (on by default)

Stages
------
1. Preprocess : kmax cut on Pk (signed-log P0, P2/P0, P4/P0), raw Bk/Qk,
                signed-log WST (raw or angle-averaged), restricted to the box.
                Saves .npy + Optuna study.
2. Optuna HPO : searches NDE architecture and training hyperparameters.
                Objective: mean log-posterior probability on x_test.
3. Retrain    : retrains the top-N Optuna nets on full train+val data.
4. Validate   : softmax-weighted ensemble, two tests, both restricted to the
                same box:
                  mock     — pk_all_mock + bk/qk_all_mock + wst_mock (self-consistency)
                  quijote  — noiseless Pk + Bk/Qk + WST from the noise_0.0 test files

Usage
-----
    python run_inference_pkqkwst_ilias_range.py <kmax> <noise> \
        --om_range 0.15 0.45 --sigma8_range 0.65 0.95 [options]
Paper Fig. 7 row 2 (Pk+Q0+WST, Om in [0.15,0.45], s8 in [0.65,0.95]) is
    python run_inference_pkqkwst_ilias_range.py 0.6 0.0 --bq qk --grid 256 \
        --average_angles --om_range 0.15 0.45 --sigma8_range 0.65 0.95
run stage by stage through submit_pkqkwst_ilias_range.sh.
"""

import sys
import os

# ── Thread setup (must happen before torch import) ───────────────────────────
NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
print(f"Detected CPUs: {NCPU}", flush=True)
os.environ["OMP_NUM_THREADS"]      = str(NCPU)
os.environ["MKL_NUM_THREADS"]      = str(NCPU)
os.environ["OPENBLAS_NUM_THREADS"] = str(NCPU)
os.environ["NUMEXPR_NUM_THREADS"]  = str(NCPU)

import matplotlib
matplotlib.use("Agg")
import torch
torch.set_num_threads(NCPU)
torch.set_num_interop_threads(NCPU)
print(f"Torch threads: {torch.get_num_threads()}", flush=True)

import numpy as np
import dill
import logging
import time
from os.path import join
from omegaconf import OmegaConf
import optuna

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}", flush=True)

# ── ilias imports ─────────────────────────────────────────────────────────────
ILIAS_ROOT = "/u/yzhang116/ilias"
sys.path.insert(0, ILIAS_ROOT)

from ilias.train import (
    run_training, evaluate_posterior, plot_training_history,
)
from ilias.preprocess import setup_optuna
from ilias.hyperparameters import sample_hyperparameters_optuna
from ilias.validate import load_ensemble, run_validation
from ilias.tools import select_top_trials

# ── Constants ─────────────────────────────────────────────────────────────────
PK_DATA_DIR  = "/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Pk_all_noise"
BK_DATA_DIR  = "/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Bk_all_noise"
WST_DATA_DIR    = "/u/yzhang116/NN/inference/wide_hod_ext/WST_all_noise"
HYPERPRIOR_PATH = join(ILIAS_ROOT, "conf/net/hyperprior.yaml")
WST_TYPE = "angular"

TEST_N_SAMPLES = 1000
TEST_SEED      = 32


def noise_str(noise):
    return f"{noise:.2f}"

INDSEL_P    = [0, 4]
PARAM_NAMES = [r"$\Omega_m$", r"$\sigma_8$"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Data preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def preprocess_pk(pk, k, kmax):
    """kmax cut + signed-log P0, P2/P0, P4/P0 normalization → flatten."""
    k_idx = np.where(k <= kmax)[0]
    pk = np.transpose(pk[:, k_idx, :], (0, 2, 1))   # (N, 3, n_k_cut)
    pk[:, 1:] /= pk[:, 0:1]                          # P2/P0, P4/P0
    pk[:, 0] = (np.sign(pk[:, 0])
                * np.log1p(np.abs(pk[:, 0]))
                / np.log(10))                         # signed-log10
    pk = pk.reshape(pk.shape[0], -1)
    return np.nan_to_num(pk, nan=0.0)


def preprocess_angular(data):
    """
    Average angular WST coefficients over orientation bins.

    Input  : (N, 4683) — S0 (3) | S1 (90 dirs × 2 angles = 180) | S2 (450 × 10 = 4500)
    Output : (N, 543)  — S0 (3) | S1_avg (90) | S2_avg (450)
    """
    s0 = data[:, :3]
    s1 = data[:, 3:183]
    s2 = data[:, 183:]
    s1_avg = s1.reshape(s1.shape[0], 90,  2).mean(axis=2)
    s2_avg = s2.reshape(s2.shape[0], 450, 10).mean(axis=2)
    return np.concatenate([s0, s1_avg, s2_avg], axis=1)


def log_transform_wst(arr):
    """Signed-log10 transform of raw WST coefficients (same as Pk's P0)."""
    return (np.sign(arr) * np.log1p(np.abs(arr)) / np.log(10))


def _wst_features(arr, average_angles):
    """Log-transform WST coefficients, then angle-average if requested."""
    arr = log_transform_wst(arr)
    return preprocess_angular(arr) if average_angles else arr


# ═══════════════════════════════════════════════════════════════════════════════
#  Om/sigma8 box restriction
# ═══════════════════════════════════════════════════════════════════════════════

def restrict_to_box(x, theta, om_range, sigma8_range, label):
    """theta has already been reduced to [Omega_m, sigma_8] columns (INDSEL_P)."""
    if om_range is None and sigma8_range is None:
        return x, theta
    mask = np.ones(len(theta), dtype=bool)
    if om_range is not None:
        mask &= (theta[:, 0] >= om_range[0]) & (theta[:, 0] <= om_range[1])
    if sigma8_range is not None:
        mask &= (theta[:, 1] >= sigma8_range[0]) & (theta[:, 1] <= sigma8_range[1])
    logging.info(f"Box restriction [{label}]: kept {mask.sum()}/{len(mask)} rows "
                 f"(Om in {om_range}, sigma8 in {sigma8_range})")
    return x[mask], theta[mask]


def load_and_preprocess(noise, kmax, bq, grid, average_angles, om_range, sigma8_range):
    """
    Load pre-split dill files (Pk, Bk/Qk, WST), restrict to the Om/sigma8
    box, and return preprocessed arrays (no nbar).

    Summary = [Pk_norm (kmax-cut) | Bk or Qk (raw) | WST_norm (signed-log, raw or angle-averaged)]

    Train uses mock observations; test uses mock (random subsample).
    """
    pk_train_path  = join(PK_DATA_DIR,  f"Pk_nbar_NGP_galnoise_{noise_str(noise)}_200c_train.dill")
    pk_test_path   = join(PK_DATA_DIR,  f"Pk_nbar_NGP_galnoise_{noise_str(noise)}_200c_test.dill")
    bk_train_path  = join(BK_DATA_DIR,  f"Bk_NGP_galnoise_{noise_str(noise)}_200c_train.dill")
    bk_test_path   = join(BK_DATA_DIR,  f"Bk_NGP_galnoise_{noise_str(noise)}_200c_test.dill")
    wst_train_path = join(WST_DATA_DIR, f"WST_nbar_{WST_TYPE}_NGP_kc34pi_40000_galnoise_{noise_str(noise)}_train.dill")
    wst_test_path  = join(WST_DATA_DIR, f"WST_nbar_{WST_TYPE}_NGP_kc34pi_40000_galnoise_{noise_str(noise)}_test.dill")

    logging.info(f"Loading Pk train  : {pk_train_path}")
    logging.info(f"Loading Bk train  : {bk_train_path}")
    logging.info(f"Loading WST train : {wst_train_path}")
    logging.info(f"average_angles : {average_angles}")

    with open(pk_train_path,  "rb") as f: pk_train  = dill.load(f)
    with open(pk_test_path,   "rb") as f: pk_test   = dill.load(f)
    with open(bk_train_path,  "rb") as f: bk_train  = dill.load(f)
    with open(bk_test_path,   "rb") as f: bk_test   = dill.load(f)
    with open(wst_train_path, "rb") as f: wst_train = dill.load(f)
    with open(wst_test_path,  "rb") as f: wst_test  = dill.load(f)

    k = pk_train["k"]
    wst_key_mock = f"wst_mock_{grid}_all"

    x_trainval = np.concatenate([
        preprocess_pk(pk_train["pk_all_mock"], k, kmax),     # Pk (normalised)
        bk_train[f"{bq}_all_mock"],                          # Bk/Qk (raw)
        _wst_features(wst_train[wst_key_mock], average_angles),  # WST
    ], axis=1)
    theta_trainval = pk_train["theta_all"][:, INDSEL_P]
    x_trainval, theta_trainval = restrict_to_box(
        x_trainval, theta_trainval, om_range, sigma8_range, "train+val")

    x_test_full = np.concatenate([
        preprocess_pk(pk_test["pk_all_mock"], k, kmax),
        bk_test[f"{bq}_all_mock"],
        _wst_features(wst_test[wst_key_mock], average_angles),
    ], axis=1)
    theta_test_full = pk_test["theta_all"][:, INDSEL_P]
    x_test_full, theta_test_full = restrict_to_box(
        x_test_full, theta_test_full, om_range, sigma8_range, "internal test pool")

    rng_test = np.random.default_rng(TEST_SEED)
    n_test = min(TEST_N_SAMPLES, len(x_test_full))
    if n_test < TEST_N_SAMPLES:
        logging.warning(f"Internal test pool only has {n_test} rows in the box "
                        f"(< {TEST_N_SAMPLES} default) — using all of them.")
    test_idx = rng_test.choice(len(x_test_full), size=n_test, replace=False)
    x_test     = x_test_full[test_idx]
    theta_test = theta_test_full[test_idx]

    logging.info(f"Summary features : {x_trainval.shape[1]}")
    logging.info(f"Train size : {len(x_trainval)},  Test size : {len(x_test)}")
    return x_trainval, theta_trainval, x_test, theta_test


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 1 – Preprocess
# ═══════════════════════════════════════════════════════════════════════════════

def stage_preprocess(model_dir, x_trainval, theta_trainval,
                     x_test, theta_test, val_frac, seed, cfg):
    """Split train → train+val, save .npy arrays, initialise Optuna study."""
    if os.path.exists(join(model_dir, "x_train.npy")):
        logging.info("Stage 1: preprocessed data found – skipping.")
        return

    rng   = np.random.default_rng(seed)
    idx   = rng.permutation(len(x_trainval))
    n_val = int(val_frac * len(x_trainval))

    val_idx   = idx[:n_val]
    train_idx = idx[n_val:]

    x_train,   theta_train = x_trainval[train_idx], theta_trainval[train_idx]
    x_val,     theta_val   = x_trainval[val_idx],   theta_trainval[val_idx]
    ids_train = train_idx.astype(str)
    ids_val   = val_idx.astype(str)
    ids_test  = np.array([f"test_{i}" for i in range(len(x_test))])

    os.makedirs(model_dir, exist_ok=True)
    np.save(join(model_dir, "x_train.npy"),     x_train)
    np.save(join(model_dir, "x_val.npy"),       x_val)
    np.save(join(model_dir, "x_test.npy"),      x_test)
    np.save(join(model_dir, "theta_train.npy"), theta_train)
    np.save(join(model_dir, "theta_val.npy"),   theta_val)
    np.save(join(model_dir, "theta_test.npy"),  theta_test)
    np.save(join(model_dir, "ids_train.npy"),   ids_train)
    np.save(join(model_dir, "ids_val.npy"),     ids_val)
    np.save(join(model_dir, "ids_test.npy"),    ids_test)
    with open(join(model_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)

    logging.info(f"Stage 1: train={len(x_train)}, val={len(x_val)}, "
                 f"test={len(x_test)}")
    setup_optuna(model_dir, "study", cfg.n_startup_trials)
    logging.info("Stage 1: Optuna study initialised.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 2 – Optuna HPO
# ═══════════════════════════════════════════════════════════════════════════════

def stage_optuna(model_dir, cfg):
    """
    Search for optimal NPE hyperparameters via Optuna TPE.

    Objective: mean log-posterior probability on x_test.
    SQLite study supports parallel SLURM array workers.
    """
    x_train     = np.load(join(model_dir, "x_train.npy"))
    theta_train = np.load(join(model_dir, "theta_train.npy"))
    x_val       = np.load(join(model_dir, "x_val.npy"))
    theta_val   = np.load(join(model_dir, "theta_val.npy"))
    x_test      = np.load(join(model_dir, "x_test.npy"))
    theta_test  = np.load(join(model_dir, "theta_test.npy"))

    study = setup_optuna(model_dir, "study", cfg.n_startup_trials)

    def objective(trial):
        mcfg = sample_hyperparameters_optuna(trial, cfg.net, cfg.embedding_net)
        trial.set_user_attr("mcfg", OmegaConf.to_container(mcfg, resolve=True))
        logging.info(
            f"Trial {trial.number} | "
            f"hf={mcfg.hidden_features} nt={mcfg.num_transforms} "
            f"depth={mcfg.get('fcn_depth','?')} width={mcfg.get('fcn_width','?')} "
            f"bs={mcfg.batch_size} lr={mcfg.learning_rate:.2e}"
        )
        t0 = time.time()
        try:
            posterior, _ = run_training(
                x_train=x_train, theta_train=theta_train,
                x_val=x_val,     theta_val=theta_val,
                out_dir=None, cfg=cfg, mcfg=mcfg,
            )
            score = float(evaluate_posterior(posterior, x_test, theta_test))
        except Exception as exc:
            logging.warning(f"Trial {trial.number} failed: {exc}")
            score = float("-inf")
        elapsed = time.time() - t0
        trial.set_user_attr("timing", elapsed)
        logging.info(f"Trial {trial.number} score={score:.4f}  ({elapsed:.0f}s)")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=cfg.n_trials,
        n_jobs=1,
        timeout=60 * 60 * 23,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if completed:
        logging.info(
            f"Stage 2 done: {len(completed)} trials completed. "
            f"Best value={study.best_value:.4f} (trial {study.best_trial.number})"
        )
    else:
        logging.warning("Stage 2: no trials completed.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 3 – Retrain top-N nets on full train + val
# ═══════════════════════════════════════════════════════════════════════════════

def stage_retrain(model_dir, cfg):
    """Re-train top-N Optuna nets on full train+val; save posterior.pkl."""
    x_train     = np.load(join(model_dir, "x_train.npy"))
    theta_train = np.load(join(model_dir, "theta_train.npy"))
    x_val       = np.load(join(model_dir, "x_val.npy"))
    theta_val   = np.load(join(model_dir, "theta_val.npy"))
    x_test      = np.load(join(model_dir, "x_test.npy"))
    theta_test  = np.load(join(model_dir, "theta_test.npy"))

    study = optuna.load_study(
        storage=f"sqlite:///{join(model_dir, 'optuna_study.db')}",
        study_name="study",
    )
    top_trials = select_top_trials(study, cfg.Nnets)

    net_index = cfg.get("net_index")
    if net_index is not None:
        if net_index >= len(top_trials):
            logging.warning(f"Stage 3: net_index {net_index} out of bounds "
                            f"({len(top_trials)} top trials). Nothing to do.")
            return
        top_trials = [top_trials[net_index]]
        logging.info(f"Stage 3: retraining net_index={net_index} "
                     f"(trial {top_trials[0].number}) on train+val.")
    else:
        logging.info(f"Stage 3: retraining top-{len(top_trials)} nets on train+val.")

    x_all     = np.vstack([x_train, x_val])
    theta_all = np.vstack([theta_train, theta_val])
    n_hold    = max(1, int(0.05 * len(x_all)))
    x_re, theta_re         = x_all[:-n_hold], theta_all[:-n_hold]
    x_re_val, theta_re_val = x_all[-n_hold:], theta_all[-n_hold:]

    for t in top_trials:
        out_dir = join(model_dir, "nets", f"net-{t.number}")
        if os.path.exists(join(out_dir, "posterior.pkl")):
            logging.info(f"  net-{t.number}: already retrained – skipping.")
            continue
        mcfg = OmegaConf.create(t.user_attrs["mcfg"])
        logging.info(f"  net-{t.number}: HPO score={t.value:.4f}")
        try:
            posterior, histories = run_training(
                x_train=x_re,    theta_train=theta_re,
                x_val=x_re_val,  theta_val=theta_re_val,
                out_dir=out_dir, cfg=cfg, mcfg=mcfg,
            )
            plot_training_history(histories, out_dir)
            score = float(evaluate_posterior(posterior, x_test, theta_test))
            with open(join(out_dir, "log_prob_test.txt"), "w") as f:
                f.write(f"{score}\n")
            logging.info(f"  net-{t.number}: retrained, test log_prob={score:.4f}")
        except Exception as exc:
            logging.error(f"  net-{t.number}: retraining failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage 4 – Validate
# ═══════════════════════════════════════════════════════════════════════════════

def _load_combined_test(pk_path, bk_path, wst_path, pk_key, bk_key, wst_key,
                        kmax, average_angles, om_range, sigma8_range,
                        n_samples=TEST_N_SAMPLES, seed=TEST_SEED):
    """Load and preprocess a test set from the three dill files (no nbar),
    restricted to the Om/sigma8 box. n_samples=None means "use every row in
    the box" (no cap) -- used whenever a box restriction is active."""
    with open(pk_path,  "rb") as f: pk_data  = dill.load(f)
    with open(bk_path,  "rb") as f: bk_data  = dill.load(f)
    with open(wst_path, "rb") as f: wst_data = dill.load(f)
    k = pk_data["k"]
    x_full = np.concatenate([
        preprocess_pk(pk_data[pk_key], k, kmax),
        bk_data[bk_key],
        _wst_features(wst_data[wst_key], average_angles),
    ], axis=1)
    theta_full = pk_data["theta_all"][:, INDSEL_P]
    x_full, theta_full = restrict_to_box(
        x_full, theta_full, om_range, sigma8_range, f"validation ({pk_key})")

    rng = np.random.default_rng(seed)
    if n_samples is None:
        n = len(x_full)
        logging.info(f"Validation ({pk_key}): using all {n} rows in the box "
                    f"(no subsample cap).")
        idx = rng.permutation(n)
    else:
        n = min(n_samples, len(x_full))
        if n < n_samples:
            logging.warning(f"Validation ({pk_key}) pool only has {n} rows in the "
                            f"box (< {n_samples} default) — using all of them.")
        idx = rng.choice(len(x_full), size=n, replace=False)
    return (torch.Tensor(x_full[idx]).to(device),
            torch.Tensor(theta_full[idx]).to(device))


def stage_validate(model_dir, cfg, noise, kmax, bq, grid, average_angles,
                   om_range, sigma8_range):
    """
    Build a softmax-weighted ensemble and run two validation tests, both
    restricted to the same Om/sigma8 box as training.

    Test 1 – self-consistency (mock):
        pk_all_mock + {bq}_all_mock + wst_mock (same noise as training)

    Test 2 – noiseless Quijote catalog:
        pk_all_truth + {bq}_all_truth + wst_truth from the noise_0.0 test files
    """
    ensemble = load_ensemble(
        model_dir, cfg.Nnets,
        weighted=True, plot=True, clean=False,
    )

    tests = [
        (
            "mock",
            join(PK_DATA_DIR,  f"Pk_nbar_NGP_galnoise_{noise_str(noise)}_200c_test.dill"),
            join(BK_DATA_DIR,  f"Bk_NGP_galnoise_{noise_str(noise)}_200c_test.dill"),
            join(WST_DATA_DIR, f"WST_nbar_{WST_TYPE}_NGP_kc34pi_40000_galnoise_{noise_str(noise)}_test.dill"),
            "pk_all_mock", f"{bq}_all_mock", f"wst_mock_{grid}_all",
        ),
        (
            "quijote",
            join(PK_DATA_DIR,  "Pk_nbar_NGP_galnoise_0.00_200c_test.dill"),
            join(BK_DATA_DIR,  "Bk_NGP_galnoise_0.00_200c_test.dill"),
            join(WST_DATA_DIR, f"WST_nbar_{WST_TYPE}_NGP_kc34pi_40000_galnoise_0.00_test.dill"),
            "pk_all_truth", f"{bq}_all_truth", f"wst_truth_{grid}_all",
        ),
    ]

    box_active = (om_range is not None) or (sigma8_range is not None)
    val_n_samples = None if box_active else TEST_N_SAMPLES // 2

    for name, pk_path, bk_path, wst_path, pk_key, bk_key, wst_key in tests:
        logging.info(f"Stage 4: test '{name}'")
        x_t, theta_t = _load_combined_test(
            pk_path, bk_path, wst_path, pk_key, bk_key, wst_key, kmax,
            average_angles, om_range, sigma8_range, n_samples=val_n_samples)
        out_dir = join(model_dir, f"validation_{name}")
        os.makedirs(out_dir, exist_ok=True)
        np.save(join(out_dir, "x_test.npy"),     x_t.cpu().numpy())
        np.save(join(out_dir, "theta_test.npy"), theta_t.cpu().numpy())

        run_validation(ensemble, x_t, theta_t, out_dir, names=list(PARAM_NAMES))
        logging.info(f"Stage 4: '{name}' results saved to {out_dir}")

    logging.info("Stage 4: all validation tests complete.")


# ═══════════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def build_cfg(model_dir, n_trials, backend, n_top_nets, om_range, sigma8_range):
    hyperprior = OmegaConf.load(HYPERPRIOR_PATH)
    return OmegaConf.create({
        "model_dir":    model_dir,
        "device":       device,
        "net_index":    None,
        "backend":      backend,
        "engine":       "NPE",
        "embedding_net": "fcn",
        "val_frac":     0.1,
        "test_frac":    0.0,
        "seed":         32,
        "pca_features": None,
        "prior":        "uniform",
        "retrain":      False,
        "verbose":      False,
        "batch_size":   None,
        "learning_rate": None,
        "stop_after_epochs": 30,
        "weight_decay": None,
        "lr_decay_factor": None,
        "lr_patience":  None,
        "validation_smoothing_method": "ema",
        "ema_decay":    0.9,
        "precompress":  False,
        "Nnets":        n_top_nets,
        "n_trials":     n_trials,
        "n_startup_trials": max(10, min(50, n_trials // 3)),
        "cross_val":    False,
        "n_splits":     4,
        "clean_models": False,
        "param_names":  list(PARAM_NAMES),
        # Om/sigma8 box restriction
        "om_range":     list(om_range) if om_range is not None else None,
        "sigma8_range": list(sigma8_range) if sigma8_range is not None else None,
        "net":          hyperprior,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ilias-based NPE HPO for Pk+Bk/Qk+WST inference (no nbar), "
                    "restricted to an Omega_m x sigma_8 box.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("kmax",  type=float, help="Maximum wavenumber for Pk")
    parser.add_argument("noise", type=float, help="Galaxy noise level")
    parser.add_argument("--bq",  type=str,   default="qk",
                        choices=["bk", "qk"],
                        help="Bispectrum type: bk (bispectrum) or qk (reduced)")
    parser.add_argument("--grid", type=int, default=256,
                        help="WST grid size (used in data key wst_mock_{grid}_all)")
    parser.add_argument("--average_angles", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Average WST coefficients over orientation bins "
                             "(reduces 4683 → 543 features, removes angular info).")
    parser.add_argument("--stage", choices=["preprocess", "optuna", "retrain",
                                            "validate", "all"],
                        default="all")
    parser.add_argument("--n_trials",   type=int, default=200)
    parser.add_argument("--model_dir",  type=str, default=None)
    parser.add_argument("--n_top_nets", type=int, default=10)
    parser.add_argument("--net_index",  type=int, default=None,
                        help="If set, retrain only this single top-net index "
                             "(0-based) during --stage retrain. Used to "
                             "parallelize retraining across a SLURM array.")
    parser.add_argument("--backend",    type=str, default="lampe",
                        choices=["lampe", "sbi"])
    parser.add_argument("--om_range",     type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Restrict train/test/validation to Omega_m in [LO, HI].")
    parser.add_argument("--sigma8_range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Restrict train/test/validation to sigma_8 in [LO, HI].")
    args = parser.parse_args()

    if args.model_dir is None:
        avg_tag = "_avgangles" if args.average_angles else ""
        range_tag = ""
        if args.om_range:
            range_tag += f"_Om{args.om_range[0]:g}-{args.om_range[1]:g}"
        if args.sigma8_range:
            range_tag += f"_s8{args.sigma8_range[0]:g}-{args.sigma8_range[1]:g}"
        args.model_dir = (
            f"/u/yzhang116/NN/inference/all_test/ilias_runs/"
            f"{args.bq}_kmax{args.kmax}_wstgrid{args.grid}{avg_tag}_noise{noise_str(args.noise)}{range_tag}"
        )

    cfg_path = join(args.model_dir, "config.yaml")
    if args.stage in ("optuna", "retrain", "validate") and os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        cfg.n_trials = args.n_trials
        logging.info(f"Loaded config from {cfg_path}")
    else:
        cfg = build_cfg(args.model_dir, args.n_trials, args.backend, args.n_top_nets,
                        args.om_range, args.sigma8_range)
    cfg.net_index = args.net_index

    print(
        f"\nstage={args.stage}  kmax={args.kmax}  noise={args.noise}  "
        f"bq={args.bq}  grid={args.grid}  average_angles={args.average_angles}  "
        f"backend={cfg.backend}  n_trials={cfg.n_trials}  n_top_nets={cfg.Nnets}  "
        f"om_range={args.om_range}  sigma8_range={args.sigma8_range}",
        flush=True,
    )
    print(f"Output: {args.model_dir}\n", flush=True)

    if args.stage in ("preprocess", "all"):
        x_trainval, theta_trainval, x_test, theta_test = load_and_preprocess(
            args.noise, args.kmax, args.bq, args.grid, args.average_angles,
            args.om_range, args.sigma8_range)
        logging.info("=== Stage 1: Preprocess ===")
        stage_preprocess(
            args.model_dir, x_trainval, theta_trainval, x_test, theta_test,
            val_frac=0.1, seed=32, cfg=cfg,
        )

    if args.stage in ("optuna", "all"):
        logging.info("=== Stage 2: HPO (Optuna) ===")
        stage_optuna(args.model_dir, cfg)

    if args.stage in ("retrain", "all"):
        logging.info("=== Stage 3: Retrain top nets ===")
        stage_retrain(args.model_dir, cfg)

    if args.stage in ("validate", "all"):
        logging.info("=== Stage 4: Validate ===")
        om_range = tuple(cfg.om_range) if cfg.get("om_range") else None
        sigma8_range = tuple(cfg.sigma8_range) if cfg.get("sigma8_range") else None
        stage_validate(args.model_dir, cfg, args.noise, args.kmax, args.bq,
                       args.grid, args.average_angles, om_range, sigma8_range)

    logging.info(f"Stage '{args.stage}' complete.")
