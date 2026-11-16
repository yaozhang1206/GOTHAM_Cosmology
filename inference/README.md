# Inference pipeline (paper Fig. 7)

Code that produces `fig07_validation_pkqk_full_vs_pkqkwst_box`: SBI (NPE via
ilias) validation for

| row | summary statistic | training range | run directory |
|-----|-------------------|----------------|---------------|
| 1 | P_ℓ(k ≤ 0.6) + Q₀ | full prior | `bk_test/ilias_runs/qk_kmax0.6_noise0.0_ext_200c_40000` |
| 2 | P_ℓ(k ≤ 0.6) + Q₀ + WST (angle-averaged, grid 256) | Ωm ∈ [0.15, 0.45], σ8 ∈ [0.65, 0.95] | `all_test/ilias_runs/qk_kmax0.6_wstgrid256_avgangles_noise0.0_Om0.15-0.45_s80.65-0.95` |

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

> **Known issue: HOD mismatch between Pk/Qk and WST for LH 0–1999.**
> For isim < 2000, the WST split reads per-LH files from the older
> `wide_hod_ext/WST_all_noise/` set, which was painted with a different HOD
> table. A check on the saved splits (2026-11-15) found the same cosmology
> columns in every row but different HOD parameters (theta columns 6–15) in
> 19000/39000 train rows (all of isim < 2000) and in 1000/1000 test rows.
> Pk/Qk rows match exactly. So for those rows the Pk+Qk and WST parts of the
> combined vector come from different HOD draws at the same cosmology. LH
> 2000–3999 rows are consistent. The WST rows for isim < 2000 came from
> `LH_points_HOD_cosmo_ext_20000.txt`.
>
> Fix in progress: mock positions for LH 0–1999 and truth positions for the
> test sims have been regenerated with the 40000 HOD table
> (`gal_pos_ext_40000_{mock,truth}.py`). After their WST is computed on
> DeltaAI, `prepare_wst_train_test_40000.py` must read those files for
> isim < 2000 and the row-2 SBI must be rerun.

## 3. `sbi/` — run the SBI pipeline

Each submit script chains 4 SLURM jobs: preprocess → Optuna HPO
(40 workers × 5 = 200 trials) → retrain top 10 nets (array) → validate
(softmax-weighted ensemble on `validation_mock` and `validation_quijote`).

```bash
# row 1: Pk+Q0, full prior
bash sbi/submit_pkqk_ilias.sh 0.6 0.0 qk
# row 2: Pk+Q0+WST, restricted box
bash sbi/submit_pkqkwst_ilias_range.sh 0.6 0.0 0.15 0.45 0.65 0.95
```

Requires the `nn` conda env and ilias at `/u/yzhang116/ilias`.

## 4. `plot/` — the figure

```bash
python plot/plot_validation_pkqk_full_vs_pkqkwst_box.py
```

## Data files

- `quijote_saved_idx.txt` — LH sim ids of the test set.
- `quijote_LH_params_4000.txt` — Quijote LH cosmologies (Om Ob h ns s8) for
  LH 0–3999, the cosmology columns of `LH_points_HOD_cosmo_ext_40000.txt`.
