# Inference pipeline (paper Figs. 7 and 8)

Code that produces `fig07_validation_pkqk_full_vs_pkqkwst_box` (Fig. 7) and
`posterior_std` (Fig. 8): SBI (NPE via ilias) validation for

| row | summary statistic | training range | run directory |
|-----|-------------------|----------------|---------------|
| 1 | P_ℓ(k ≤ 0.6) + Q₀ | full prior | `bk_test/ilias_runs/qk_kmax0.6_noise0.0_ext_200c_40000` |
| 2 | P_ℓ(k ≤ 0.6) + Q₀ + WST (angle-averaged, grid 256) | Ωm ∈ [0.15, 0.45], σ8 ∈ [0.65, 0.95] | `all_test/ilias_runs/qk_kmax0.6_wstgrid256_avgangles_noise0.0_Om0.15-0.45_s80.65-0.95_hodfix` |

A full-prior Pk+Q₀+WST run (`..._fullrange_hodfix`) is also available as an
alternative row 2.

Training set: 4000 Quijote LH cosmologies × 10 HOD draws = 40000 mock (emulator)
galaxy catalogs. Test set: the cosmologies in `quijote_saved_idx.txt` (mock +
Quijote N-body "truth" catalogs). Galaxy noise = 0.0 everywhere.

## 1. `statistics/` — generate summary statistics

| step | script | launcher | output |
|------|--------|----------|--------|
| HOD × cosmology LH table | `gen_lh_hod_ext_40000.py` | — | `LH_points_HOD_cosmo_ext_40000.txt` |
| Pk + Bk/Qk, mock catalogs (LH 0–3999) | `gal_pk_bk_ext_40000_mock.py` | `run_pk_bk_mock_40000.slurm` | `wide_hod_ext_40000/{Pk,Bk}_all_noise/*_mock_LH_{i}.dill` |
| Pk + Bk/Qk, truth catalogs (test sims) | `gal_pk_bk_ext_40000_truth.py` | `run_pk_bk_truth_40000.slurm` | `wide_hod_ext_40000/{Pk,Bk}_all_noise/*_truth_LH_{i}.dill` |
| RSD galaxy positions, mock LH 0–3999 | `gal_pos_ext_40000_mock.py` | `run_gal_pos_mock_40000.slurm` (args `2000 4000`; rerun with `0 2000` for LH 0–1999) | `wide_hod_ext_40000/galaxy_rsd_pos_noise/*_mock_LH_{i}.dill` |
| RSD galaxy positions, truth (test sims) | `gal_pos_ext_40000_truth.py` | `run_gal_pos_truth_40000.slurm` | `wide_hod_ext_40000/galaxy_rsd_pos_noise/*_truth_LH_{i}.dill` |

The WST coefficients (`WST_angular_NGP_kc34pi_*`) are computed on DeltaAI
from the galaxy positions; that code is not in this repo.

## 2. `split/` — build train/test sets

Train = every LH sim not in `quijote_saved_idx.txt` (mock only);
test = sims in `quijote_saved_idx.txt` (mock + truth). Rows are ordered
`(isim ascending, ihod 0..9)` in every file. The Pk, Bk and WST `theta_all`
arrays must stay row-aligned, because the SBI drivers concatenate them
column-wise.

| script | output |
|--------|--------|
| `prepare_pk_40000.py [noise]` | `Pk_nbar_NGP_galnoise_{noise}_200c_{train,test}.dill` |
| `prepare_bk_40000.py [noise]` | `Bk_NGP_galnoise_{noise}_200c_{train,test}.dill` |
| `prepare_wst_train_test_40000.py [noise]` | `WST_nbar_angular_NGP_kc34pi_40000_galnoise_{noise}_{train,test}.dill` |
| `prepare_wst_train_test_40000_hodfix.py [noise]` (`run_prepare_wst_hodfix.slurm`) | same files; the version actually run on 2026-11-16 (see below) |

> **Fixed (2026-11-16): HOD mismatch between Pk/Qk and WST for LH 0–1999.**
> The WST per-LH files for isim < 2000 had been painted with
> `LH_points_HOD_cosmo_ext_20000.txt`, so in the original splits the WST rows
> for isim < 2000 (19000/39000 train rows, all 1000 test rows) used different
> HOD parameters (theta columns 6–15) from the Pk/Qk rows at the same
> cosmology. Mock positions for LH 0–1999 and truth positions for the test
> sims were regenerated with the 40000 HOD table
> (`gal_pos_ext_40000_{mock,truth}.py`), their WST recomputed on DeltaAI, and
> the WST splits rebuilt.
>
> `prepare_wst_train_test_40000.py` now reads that layout. The split actually
> used was built by `prepare_wst_train_test_40000_hodfix.py`, which takes the
> isim ≥ 2000 rows from the previous split (their per-LH files had been
> removed; those rows already matched), checks theta against the Pk and Bk
> splits (cosmology and HOD equal in 39000/39000 train and 1000/1000 test
> rows; nbar identical) and only then writes the files, keeping the old ones
> as `*.pre_hodfix.dill`. The row-2 runs were rerun into `*_hodfix`
> directories.

## 3. `sbi/` — run the SBI pipeline

Each submit script chains 4 SLURM jobs: preprocess → Optuna HPO
(40 workers × 5 = 200 trials) → retrain top 10 nets (array) → validate
(softmax-weighted ensemble on `validation_mock` and `validation_quijote`).

```bash
# row 1: Pk+Q0, full prior
bash sbi/submit_pkqk_ilias.sh 0.6 0.0 qk
# row 2: Pk+Q0+WST, restricted box (as in the paper)
MODEL_SUFFIX=_hodfix bash sbi/submit_pkqkwst_ilias_range.sh 0.6 0.0 0.15 0.45 0.65 0.95
# Pk+Q0+WST over the full prior (no box cut)
MODEL_SUFFIX=_hodfix bash sbi/submit_pkqkwst_ilias_range.sh 0.6 0.0 full
```

`MODEL_SUFFIX` is appended to the run directory; use a new one for every
rerun, since preprocessing is skipped if the directory already has
`x_train.npy`. SLURM logs go to `sbi/logs_ilias/` (git-ignored).

Requires the `nn` conda env and ilias at `/u/yzhang116/ilias`.

## 4. `plot/` — the figures

```bash
# Fig. 7 (row 2 = box run with the HOD fix; default)
python plot/plot_validation_pkqk_full_vs_pkqkwst_box.py --row2 box_hodfix
# alternative row 2: full-prior Pk+Q0+WST, or the original (pre-fix) box run
python plot/plot_validation_pkqk_full_vs_pkqkwst_box.py --row2 fullrange_hodfix
python plot/plot_validation_pkqk_full_vs_pkqkwst_box.py --row2 box_orig

# Fig. 8: posterior std vs true Om / sigma8 on the box test set
python plot/errorbar_pkqk_wst_restricted_range.py --wst_suffix _hodfix
```

Fig. 8 also needs the full-range P_ℓ (k_max = 0.3–0.6) and P_ℓ+Q₀ ensembles
evaluated on the box test set (`validation_quijote_box_Om0.15-0.45_s80.65-0.95/`
in those run directories); that evaluation code is not in this repo.

## Data files

- `quijote_saved_idx.txt` — LH sim ids of the test set.
- `quijote_LH_params_4000.txt` — Quijote LH cosmologies (Om Ob h ns s8) for
  LH 0–3999, the cosmology columns of `LH_points_HOD_cosmo_ext_40000.txt`.
