# GOTHAM_Cosmology

A transformer emulator that generates Quijote halo catalogs from fast
particle-mesh dark-matter fields, conditioned on cosmology. This repo also
holds the code that uses the emulated catalogs for simulation-based
cosmological inference.

The 1 Gpc/h box is split into 64³ sub-boxes (15.625 Mpc/h on a side). For
each sub-box, a ViT/CBAM encoder reads the DM fields from a DISCO-DJ PM run,
and an autoregressive decoder writes a "sentence" listing the halos in that
sub-box. Each halo takes 8 tokens: position (x, y, z inside the sub-box),
log mass, velocity (vx, vy, vz) and concentration.

## Repository layout

| directory | what it does |
|-----------|--------------|
| [`prepare/`](prepare/) | Builds the training data: DM input fields (DISCO-DJ PM runs) and tokenized halo sentences (from Quijote Rockstar catalogs), then the train/val/test split |
| [`model/`](model/) | The encoder–decoder model, multi-node training, and full-box catalog generation |
| [`3Gpc/`](3Gpc/) | Fused multi-GPU DISCO-DJ + emulator pipeline that generates (3 Gpc/h)³ halo catalogs on 2 nodes × 4 GPUs. See [`3Gpc/README.md`](3Gpc/README.md) |
| [`interpretability/`](interpretability/) | Attention maps, embeddings, logits and per-block analysis of a trained checkpoint |
| [`inference/`](inference/) | HOD galaxy painting, summary statistics (P_ℓ, Q₀, WST), SBI with ilias, and paper Figs. 7–8. See [`inference/README.md`](inference/README.md) |

## Pipeline

### 1. Prepare data (`prepare/`)

| script | role |
|--------|------|
| `pm.py <sim_id> [savefull]` | Runs a DISCO-DJ FastPM simulation (512³ particles, 1000 Mpc/h, snapshots at z = 2, 1, 0.5) from the given ICs. It deposits the particles on a 512³ grid and saves the per-sub-box input fields (8³ voxels each) as `dmo_fields_subvols_grid_8_*.npy` plus a meta `.pkl`. Launcher: `run.slurm` (GPU) |
| `process_halo_props_quijote.py` | Turns Rockstar halo catalogs (z = 0.5, log M > 12.7) into token sentences, one per sub-box. MPI-parallel over sims |
| `ngp_funcs.pyx`, `setup_cython_funcs.py` | Cython NGP helper used by the tokenizer. Build it with `python setup_cython_funcs.py build_ext --inplace` |
| `select_20000.py` | Splits 2000 LH sims into 1800 train / 100 val / 100 test (seed 2026). Draws 20000 sub-boxes per training sim, inserts a halo-count token, then shuffles and writes the matched `halos_*.npy` / `fields_*.npy` shards. Launcher: `prepare.slurm` (CPU) |

**Token layout.** The vocabulary has 131 value bins plus special tokens
(`start = 132`, `pad = 134`, `end = 135`). A training sentence is laid out as:

```
[start] [Ωm Ωb h ns σ8] [N_halo] [x y z lgM vx vy vz c] × N_halo [end] [pad …]
```

Each sub-box holds at most 36 halos, so a sentence is at most 296 tokens
long. Value ranges: lgM ∈ [12.7, 15], v ∈ [−1250, 1250] km/s,
c ∈ [1, 16], and 40 position bins per axis inside a sub-box. The training
loss ignores the cosmology tokens. Metadata (token ids, max length) is read
from `sentence_params.pkl`.

### 2. Train and generate (`model/`)

| file | role |
|------|------|
| `model_enc_dec_cos_fast.py` | `HaloDecoderModel`. A 3D ViT encoder (`vit_wpos_embed_v3.py`) with CBAM residual blocks (`cbam_v2.py`, `resnet.py`) feeds an RoPE transformer decoder through cross-attention. Includes KV-cached `generate()` |
| `train.py` | DDP training with an NVIDIA DALI data pipeline, bf16 autocast, and a cosine LR schedule with warmup. Logs to Weights & Biases (project `quijote`). Every 500 steps it saves a checkpoint and generates full boxes for 8 validation sims, then plots the HMF and P(k) ratios (`gen_func.py`, `performance.py`) |
| `train_nodes.slurm` | `torchrun` launcher (4 nodes × 1 GPU) |
| `generate_catalog_fast.py` | Loads a checkpoint and generates all 64³ sub-boxes of a sim in chunks. It decodes the tokens back to a continuous catalog `[x, y, z, M, vx, vy, vz, c]` (`generated_halo_catalog_<simid>_v4.npy`). Launcher: `test.slurm` |

Run training and generation from inside `model/`, because the scripts
import each other by bare module name:

```bash
cd model
sbatch train_nodes.slurm      # train
sbatch test.slurm             # generate catalogs
```

### 3. Large-volume catalogs (`3Gpc/`)

The emulator works one 15.625 Mpc/h sub-box at a time, so it can produce
boxes much larger than the 1 Gpc/h training volume. `3Gpc/` runs a
1536³-particle, 3 Gpc/h DISCO-DJ simulation across 8 GPUs and builds the
emulator input fields on the GPUs. It passes them through `/dev/shm` to one
emulator worker per GPU, which writes a 192³-sub-box halo catalog (about
19 min and 17.5 M halos per box). It uses the multi-GPU DISCO-DJ at
<https://github.com/yaozhang1206/DISCO-DJ> (branch `multigpu-kit`). See
[`3Gpc/README.md`](3Gpc/README.md).

```bash
cd 3Gpc
CIDS="663 188" sbatch job_3gpc.sbatch
```

### 4. Interpretability (`interpretability/`)

The analysis scripts (GPU) run a trained checkpoint and save arrays. The
plot scripts (CPU) then make the paper figures from those arrays, in MNRAS
single-column style.

| script | role |
|--------|------|
| `save_embed_logits_gen.py` | Dumps token, ViT-position, halo-index and property embeddings, per-block hidden states and logits for sim 66 (`sim66_embeddings_logits_blocks.npz`) |
| `visualize_self_attn_gen.py` | Self-attention maps per layer and head, averaged over sub-boxes and cosmologies (`self_attn_per_layer_head_cosmo_66.npy`) |
| `visualize_cross_attn_gen.py` | Cross-attention maps per layer and head, with encoder patches ordered by local density (`cross_attn_per_layer_head_cosmo_66_5halos.npy`) |
| `plot_self_avg.py` | Decoder self-attention averaged over layers and heads, for the header tokens (`<START>`, cosmology, N_halo) and the first 5 halos |
| `plot_cross_avg.py` | Cross-attention averaged over layers and heads, for the header rows and the first 3 halos against the cosmology tokens and DM-field patches |
| `plot_logits.py idx1 idx2 idx3` | Next-token probabilities for all 8 halo properties of three samples, compared with the true tokens |
| `plot_logits_xyzm.py [idx]` | Compact version for one sample (default `idx = 62`): x, y, z and mass for 3 halos in a 4-row panel |
| `plot_embed.py` | Cosine similarity of the ViT positional embeddings to patch 0, shown on the 4×4×4 patch grid |

Input and output paths (`/work/hdd/bdne/yzhang116/cp_visualize/`,
`/u/yzhang116/NN/figures/visualize/results/`) are set directly in the scripts.
The analysis scripts add `/u/yzhang116/NN/quijote` to `sys.path` to import the
model, so point that at `model/` when running them elsewhere.

### 5. Cosmological inference (`inference/`)

Paints galaxies onto the halo catalogs with an HOD (4000 LH cosmologies × 10
HOD draws). It then measures P_ℓ(k), Q₀ and WST, builds the train/test
splits, runs NPE with ilias, and makes paper Figs. 7 and 8. See
[`inference/README.md`](inference/README.md) for the step-by-step pipeline.

## Environment

The code was developed on NCSA Delta / DeltaAI (SLURM, GH200 `ghx4` partition).
Paths to data, checkpoints and outputs under `/work/hdd/bdne/...` and
`/work/nvme/bdne/...` are written directly into the scripts, so edit them
before running elsewhere.

Conda environments referenced by the launchers:

| env | used for |
|-----|----------|
| `pm` | `prepare/pm.py` and the `3Gpc/` producer: JAX, DISCO-DJ, Pylians (`MAS_library`), numba (plus jaxdecomp for multi-GPU) |
| `general` | `prepare/` CPU jobs: numpy, Pylians, colossus, mpi4py, numba, Cython |
| `nn` | training and SBI: PyTorch (DDP, NCCL), NVIDIA DALI, wandb, ilias |
| `nn_test` | `generate_catalog_fast.py` and the `3Gpc/` emulator workers |

Main external dependencies: PyTorch ≥ 2.x, `nvidia-dali`, `wandb`, JAX +
DISCO-DJ, Pylians3, colossus, mpi4py, numba, Cython, and ilias (for SBI).

The multi-GPU DISCO-DJ used by `3Gpc/` is available at
<https://github.com/yaozhang1206/DISCO-DJ> on the `multigpu-kit` branch. It
provides `discodj_dist` and the Delta environment script
`multigpu/env/setup_delta.sh`.

## Data

The training inputs are the Quijote latin-hypercube high-resolution Rockstar
halo catalogs at z = 0.5 and DISCO-DJ PM runs from the matching initial
conditions. These datasets are not included in the repo. Index files for the
inference test set (`inference/quijote_saved_idx.txt`) and the LH cosmologies
(`inference/quijote_LH_params_4000.txt`) are tracked.
