"""
Ilias-based hyperparameter optimization for Pk + Bk/Qk NPE inference (no nbar).

Summary statistic = [signed-log P0 | P2/P0 | P4/P0 | Bk or Qk]
loaded from two separate dill files per split:
  Pk_nbar_NGP_galnoise_{noise}_*.dill  — power spectrum multipoles
  Bk_NGP_galnoise_{noise}_*.dill       — bispectrum (bk) or reduced bispectrum (qk)

Use --bq bk  to use the bispectrum (bk_all_*)
Use --bq qk  to use the reduced bispectrum (qk_all_*)

Stages
------
1. Preprocess : kmax cut on Pk (signed-log P0, P2/P0, P4/P0), raw Bk/Qk
                concatenated. Saves .npy + Optuna study.
2. Optuna HPO : searches NDE architecture and training hyperparameters.
                Objective: mean log-posterior probability on x_test.
3. Retrain    : retrains the top-N Optuna nets on full train+val data.
4. Validate   : softmax-weighted ensemble, two tests:
                  mock     — pk_all_mock + bk_all_mock (self-consistency)
                  quijote  — noiseless Pk + Bk from Pk_nbar_NGP_noise_0.0 /
                             Bk_NGP_noise_0.0 test files

Paper Fig. 7 row 1 (Pk+Q0, full prior range) is
    python run_inference_pkqk_ilias.py 0.6 0.0 --bq qk
run stage by stage through submit_pkqk_ilias.sh.

Usage
-----
    python run_inference_pkqk_ilias.py <kmax> <noise> [options]
Example:
    python run_inference_pkqk_ilias.py 0.6 0.0 --bq qk --stage all
    python run_inference_pkqk_ilias.py 0.6 0.0 --bq qk --stage optuna --n_trials 5
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
PK_DATA_DIR = "/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Pk_all_noise"
BK_DATA_DIR = "/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Bk_all_noise"
HYPERPRIOR_PATH = join(ILIAS_ROOT, "conf/net/hyperprior.yaml")


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


def load_and_preprocess(noise, kmax, bq):
    """
    Load pre-split dill files and return preprocessed arrays (no nbar).

    Summary = [Pk_norm (kmax-cut) | Bk or Qk (raw)]

    Train uses mock observations; test uses mock (500 random samples).
    """
    pk_train_path = join(PK_DATA_DIR, f"Pk_nbar_NGP_galnoise_{noise_str(noise)}_200c_train.dill")
    pk_test_path  = join(PK_DATA_DIR, f"Pk_nbar_NGP_galnoise_{noise_str(noise)}_200c_test.dill")
    bk_train_path = join(BK_DATA_DIR, f"Bk_NGP_galnoise_{noise_str(noise)}_200c_train.dill")
    bk_test_path  = join(BK_DATA_DIR, f"Bk_NGP_galnoise_{noise_str(noise)}_200c_test.dill")

    logging.info(f"Loading Pk train : {pk_train_path}")
    logging.info(f"Loading Bk train : {bk_train_path}")

    with open(pk_train_path, "rb") as f: pk_train = dill.load(f)
    with open(pk_test_path,  "rb") as f: pk_test  = dill.load(f)
    with open(bk_train_path, "rb") as f: bk_train = dill.load(f)
    with open(bk_test_path,  "rb") as f: bk_test  = dill.load(f)

    k = pk_train["k"]

    x_trainval = np.concatenate([
        preprocess_pk(pk_train["pk_all_mock"], k, kmax),  # Pk (normalised)
        bk_train[f"{bq}_all_mock"],                        # Bk/Qk (raw)
    ], axis=1)
    theta_trainval = pk_train["theta_all"][:, INDSEL_P]

    x_test_full = np.concatenate([
        preprocess_pk(pk_test["pk_all_mock"], k, kmax),
        bk_test[f"{bq}_all_mock"],
    ], axis=1)
    theta_test_full = pk_test["theta_all"][:, INDSEL_P]

    rng_test = np.random.default_rng(32)
    test_idx = rng_test.choice(len(x_test_full), size=500, replace=False)
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

def _load_test_set(pk_path, bk_path, pk_key, bk_key,
                   kmax, bq, n_samples=500, seed=32):
    """Load and preprocess a test set from the two dill files (no nbar)."""
    with open(pk_path, "rb") as f: pk_data = dill.load(f)
    with open(bk_path, "rb") as f: bk_data = dill.load(f)
    k = pk_data["k"]
    x_full = np.concatenate([
        preprocess_pk(pk_data[pk_key], k, kmax),
        bk_data[bk_key],
    ], axis=1)
    theta_full = pk_data["theta_all"][:, INDSEL_P]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_full), size=min(n_samples, len(x_full)), replace=False)
    return (torch.Tensor(x_full[idx]).to(device),
            torch.Tensor(theta_full[idx]).to(device))


def stage_validate(model_dir, cfg, noise, kmax, bq):
    """
    Build a softmax-weighted ensemble and run two validation tests.

    Test 1 – self-consistency (mock):
        pk_all_mock + {bq}_all_mock (same noise as training)

    Test 2 – noiseless Quijote catalog:
        pk_all_truth + {bq}_all_truth from the noise_0.0 test files
    """
    ensemble = load_ensemble(
        model_dir, cfg.Nnets,
        weighted=True, plot=True, clean=False,
    )

    tests = [
        (
            "mock",
            join(PK_DATA_DIR, f"Pk_nbar_NGP_galnoise_{noise_str(noise)}_200c_test.dill"),
            join(BK_DATA_DIR, f"Bk_NGP_galnoise_{noise_str(noise)}_200c_test.dill"),
            "pk_all_mock", f"{bq}_all_mock",
        ),
        (
            "quijote",
            join(PK_DATA_DIR, "Pk_nbar_NGP_galnoise_0.00_200c_test.dill"),
            join(BK_DATA_DIR, "Bk_NGP_galnoise_0.00_200c_test.dill"),
            "pk_all_truth", f"{bq}_all_truth",
        ),
    ]

    for name, pk_path, bk_path, pk_key, bk_key in tests:
        logging.info(f"Stage 4: test '{name}'")
        x_t, theta_t = _load_test_set(pk_path, bk_path, pk_key, bk_key, kmax, bq)
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

def build_cfg(model_dir, n_trials, backend, n_top_nets):
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
        "net":          hyperprior,
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ilias-based NPE HPO for Pk+Bk inference (no nbar).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("kmax",  type=float, help="Maximum wavenumber for Pk")
    parser.add_argument("noise", type=float, help="Galaxy noise level")
    parser.add_argument("--bq",  type=str,   default="qk",
                        choices=["bk", "qk"],
                        help="Bispectrum type: bk (bispectrum) or qk (reduced)")
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
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = (
            f"/u/yzhang116/NN/inference/bk_test/ilias_runs/"
            f"{args.bq}_kmax{args.kmax}_noise{args.noise}_ext_200c_40000"
        )

    cfg_path = join(args.model_dir, "config.yaml")
    if args.stage in ("optuna", "retrain", "validate") and os.path.exists(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        cfg.n_trials = args.n_trials
        logging.info(f"Loaded config from {cfg_path}")
    else:
        cfg = build_cfg(args.model_dir, args.n_trials, args.backend, args.n_top_nets)
    cfg.net_index = args.net_index

    print(
        f"\nstage={args.stage}  kmax={args.kmax}  noise={args.noise}  "
        f"bq={args.bq}  backend={cfg.backend}  "
        f"n_trials={cfg.n_trials}  n_top_nets={cfg.Nnets}",
        flush=True,
    )
    print(f"Output: {args.model_dir}\n", flush=True)

    if args.stage in ("preprocess", "all"):
        x_trainval, theta_trainval, x_test, theta_test = load_and_preprocess(
            args.noise, args.kmax, args.bq)
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
        stage_validate(args.model_dir, cfg, args.noise, args.kmax, args.bq)

    logging.info(f"Stage '{args.stage}' complete.")
