# 3 Gpc/h halo catalogs (fused PM + emulator)

Generates halo catalogs for (3 Gpc/h)³ boxes. Each box is 27 times the
volume of the 1 Gpc/h training boxes. The pipeline runs a multi-GPU
DISCO-DJ simulation and then the trained transformer emulator
([`../model/`](../model/)) on the same 2 nodes × 4 GH200 GPUs. The 18-channel
emulator input fields are built on the GPUs and passed through `/dev/shm`, so
no field files are written to disk.

**Multi-GPU DISCO-DJ** is available at
<https://github.com/yaozhang1206/DISCO-DJ> on the **`multigpu-kit`** branch.
That branch provides `discodj_dist` (distributed nLPT + slab-mesh PM) and the
Delta environment script `multigpu/env/setup_delta.sh`. The `main` branch
lacks the fixed `pad_fourier_full`, and `run_3gpc.py` refuses to start
without it.

## What one box involves

| | |
|---|---|
| box / particles / PM mesh | 3000 Mpc/h, 1536³, 3072³ |
| snapshots | FastPM 10 + 5 + 5 steps to z = 2, 1, 0.5 (2LPT ICs at a = 1/128) |
| emulator grid | 192³ sub-volumes of 8³ voxels (15.625 Mpc/h, same as training) |
| decomposition | 8 x-slabs of 24 sub-volume rows, one per GPU |
| output | `catalog_3gpc_<cid>.npy`, float32 `(N, 8)`: `x y z` [Mpc/h], `M` [M☉/h], `vx vy vz` [km/s], `c`, plus `catalog_3gpc_<cid>_meta.json` |

Cosmology ids `cid` index the 4000 LH cosmologies in
`LH_points_HOD_cosmo_ext_40000.txt`. These are the same ids as the `inference/`
pipeline. The IC phases use N-GenIC with seed = `cid`, and the z = 0 linear
P(k) comes from Quijote for `cid` < 2000 and from CAMB otherwise.

Reference run (job 3614452, cid 663): 17.5 M halos in 19 min wall-clock.
That breaks down into about 3.8 min building the IC on the CPU, 2.5 min for
DISCO-DJ plus fields, and about 12 min for the emulator. Peak memory was
about 100 GB host RSS per node and 71 GB per GPU in the emulator.

## How it works

```
job_3gpc.sbatch  (2 nodes)
└─ node_3gpc.sh  (one per node)
   ├─ run_3gpc.py     producer: 1 JAX process, 4 local GPUs, jax.distributed across nodes
   │    for each cid: ic_worker.py (CPU subprocess, prefetched one box ahead)
   │                  → pm_multigpu_gpufields.run_one_simulation_multigpu
   │                  → /dev/shm/gen3gpc_<job>/<cid>/slab<g>.npy  (fp16, 884736×8×8×8×18)
   └─ emu_worker3.py ×4  consumer: 1 torch process per GPU (CUDA_VISIBLE_DEVICES=j)
        waits for its slab → emu.Emulator.generate → emu.decode
        → CAT_DIR/.parts_<cid>/cat<g>.npy ; the worker holding slab 0 merges the 8 parts
```

JAX and torch never hold a GPU at the same time. Before starting the next
box, the producer waits for every local worker's `gpu_free_<cid>_<j>` flag
and then for a cross-node barrier. The IC for box n+1 is built on the CPU
while box n is still running on the GPUs. Boxes that already have a catalog
are skipped, so you can resubmit the same `CIDS` after a timeout. If the PM
halo region is too small for a cosmology, that box is skipped and recorded in
`CAT_DIR/needs_halo_<cid>.json`; rerun it with a larger `DISCO_HALO_SIZE`.

| file | role |
|------|------|
| `job_3gpc.sbatch` | SLURM entry point (DeltaAI `ghx4`, 2 nodes, exclusive) |
| `node_3gpc.sh` | Per-node launcher: the producer, 4 emulator workers, and a memory monitor |
| `run_3gpc.py` | Producer: IC scaling, cosmology setup, DISCO-DJ run in field-sink mode, box synchronization |
| `pm_multigpu_gpufields.py` | Multi-GPU DISCO-DJ driver with GPU-side CIC painting and sub-volume field building. Produces the same channels as [`../prepare/pm.py`](../prepare/pm.py) |
| `ic_worker.py`, `ic_dj.py` | CPU N-GenIC z = 0 linear density field (uses the compiled `discodj_native` from the conda env) |
| `emu_worker3.py` | Consumer: generates one slab per box, decodes it and writes a partial catalog, merges on slab 0 |
| `emu.py` | Wraps `HaloDecoderModel` from `../model/` for any `grid³` box or slab, with a vectorized decoder. Same checkpoint and sampling as `generate_catalog_fast.py` |
| `common3.py` | 3 Gpc/h constants, paths and seeds |
| `lh_common.py` | Cosmology table, P(k) files and IC seeds shared with the 2 Gpc/h LH4000 run |

## Running

Submit from this directory:

```bash
cd 3Gpc
CIDS="663 188" sbatch job_3gpc.sbatch
# or: CID_FILE=cids.txt sbatch job_3gpc.sbatch
```

| variable | default | meaning |
|----------|---------|---------|
| `CIDS` / `CID_FILE` | — (required) | cosmology ids to run, in order |
| `CAT_DIR` | `/work/hdd/bdne/yzhang116/gen3gpc/catalogs` | output catalogs |
| `DISCODJ_DIR` | `/u/yzhang116/DISCO-DJ-multigpu` | checkout of `yaozhang1206/DISCO-DJ` @ `multigpu-kit` |
| `DISCO_HALO_SIZE` | `96` | PM halo, in cells |

Logs go to `logs/gen3gpc_<job>.out`, plus a per-node producer, worker and
memory log under `logs/`. Lines starting with `BOX` and `CATALOG` give the
per-box timings. Only single-box runs have been timed so far, so set
`--time` with that ~19 min per box in mind, and resubmit if a job times out
(finished boxes are skipped).

Environments: the producer uses conda `pm` (JAX + jaxdecomp, Pylians, numba),
set up by `setup_delta.sh`. The workers use conda `nn_test` (PyTorch 2.5.1)
with `cudatoolkit/25.5_12.9`. The checkpoint path, the cosmology table and
the IC/P(k) directories are set in `emu.py` and `lh_common.py`.
