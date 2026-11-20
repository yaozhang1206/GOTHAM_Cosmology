"""Shared constants / helpers for the fused 3 Gpc/h pipeline (no JAX / torch imports).

DISCO-DJ (1536^3 particles, 3072^3 PM, 8 GPUs on 2 nodes) -> GPU-built emulator fields in /dev/shm ->
transformer emulator (generate_catalog_fast.py model, via emu.py) -> halo catalog.  No field files.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lh_common as LC                                     # noqa: E402  (cosmology table, Pk files, IC seeds)

BOX, RES, RES_PM = 3000.0, 1536, 3072
GRID, GS = 192, 8                       # 192^3 subvolumes of 8^3 voxels (15.625 Mpc/h, same as the 1 Gpc/h training)
NSLAB = 8                               # one x-slab of GRID/NSLAB = 24 subvolume rows per GPU
NPN = 4                                 # GPUs (= emulator workers) per node
ROWS = GRID ** 3 // NSLAB               # subvolumes per slab (884,736)
CAT_DIR = "/work/hdd/bdne/yzhang116/gen3gpc/catalogs"

cosmologies = LC.cosmologies            # (4000, 5) Om Ob h ns s8, same ids as the 2 Gpc/h LH4000 run
pk_file = LC.pk_file
ic_seed = LC.ic_seed


def cat_path(cid, cat_dir=CAT_DIR):
    return os.path.join(cat_dir, f"catalog_3gpc_{cid}.npy")


def parts_dir(cid, cat_dir=CAT_DIR):
    """Partial (per-slab) catalogs on the shared file system: the 8 slabs live on two nodes."""
    return os.path.join(cat_dir, f".parts_{cid}")


def emu_seed(cid, slab):
    return 3_000_000 + 1000 * int(cid) + 100 * int(slab)      # + chunk index (< 100 chunks per slab)


def gpu_free_flag(shm, cid, local):
    """Written by local emulator worker `local` when it has released its GPU after box `cid`."""
    return os.path.join(shm, f"gpu_free_{int(cid)}_{int(local)}")


def wait_for(pred, timeout, poll=2.0, what=""):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(poll)
