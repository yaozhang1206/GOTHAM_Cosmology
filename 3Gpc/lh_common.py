"""Shared constants / helpers for the 2 Gpc/h LH4000 production (no JAX / torch imports)."""
import os
import time

import numpy as np

COSMO_FILE = "/work/hdd/bdne/yzhang116/quijote_galaxy/LH_points_HOD_cosmo_ext_40000.txt"
QUIJOTE_IC = "/work/hdd/bdne/yzhang116/ICsQuijote"
CAMB_DIR = "/work/hdd/bdne/yzhang116/lh4000/camb"
CAT_DIR = "/work/hdd/bdne/yzhang116/lh4000/catalogs_2gpc"

BOX, RES, RES_PM = 2000.0, 1024, 2048
GRID, GS = 128, 8                       # 128^3 subvolumes of 8^3 voxels (15.625 Mpc/h)
NSLAB = 4                               # one x-slab of GRID/NSLAB subvolume rows per GPU
SNAP_NCH = [2, 8, 8]
NCH = sum(SNAP_NCH)


def cosmologies():
    a = np.loadtxt(COSMO_FILE)
    ids, first = np.unique(a[:, 15].astype(int), return_index=True)
    assert np.array_equal(ids, np.arange(len(ids)))
    return a[first, 10:15]              # (4000, 5): Om Ob h ns s8


def pk_file(cid):
    if cid < 2000:
        return os.path.join(QUIJOTE_IC, str(cid), "Pk_mm_z=0.000.txt")
    return os.path.join(CAMB_DIR, str(cid), "Pk_mm_z=0.000.txt")


def cat_path(cid, cat_dir=CAT_DIR):
    return os.path.join(cat_dir, f"catalog_2gpc_{cid}.npy")


def ic_seed(cid):
    return int(cid)                     # N-GenIC seed = cosmology id (ids < 2000: Quijote LH seed)


def emu_seed(cid, slab):
    return 1_000_000 + 1000 * int(cid) + 100 * int(slab)      # + chunk index


def gpu_free_flag(shm, cid, slab):
    """Written by emulator worker `slab` when it has released its GPU after box `cid` (kept outside
    SHM/{cid}, which worker 0 deletes after merging)."""
    return os.path.join(shm, f"gpu_free_{int(cid)}_{int(slab)}")


def wait_for(pred, timeout, poll=2.0, what=""):
    t0 = time.time()
    while not pred():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"timed out after {timeout:.0f} s waiting for {what}")
        time.sleep(poll)
