"""3 Gpc/h consumer: transformer emulator on one GPU for one x-slab of every box produced by run_3gpc.py.

Local worker j (0..3, CUDA_VISIBLE_DEVICES=j) on a node: for each cid in SHM/cids.txt wait for
SHM/{cid}/local_slabs.json, take global slab g = local_slabs[j], memory-map SHM/{cid}/slab{g}.npy, generate tokens
(emu.py = ../model/generate_catalog_fast.py model, sampling, 16384-subvolume chunks), delete the slab, release the GPU
(empty_cache + SHM/gpu_free_{cid}_{j}), decode the 24 x-rows of subvolumes to a partial catalog on the shared FS
(CAT_DIR/.parts_{cid}/cat{g}.npy).  The worker holding slab 0 merges the 8 parts into CAT_DIR/catalog_3gpc_{cid}.npy
(+ _meta.json).  Catalog columns: x y z [Mpc/h] M [Msun/h] vx vy vz [km/s] c.

Env: conda nn_test (torch), module cudatoolkit/25.5_12.9.
"""
import argparse
import json
import os
import shutil
import signal
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common3 as C       # noqa: E402
import emu                # noqa: E402  (emu.py, on sys.path via common3)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def producer_pids(shm):
    key = shm.rstrip("/").encode()
    pids = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/cmdline", "rb") as fh:
                args = fh.read().split(b"\0")
        except OSError:
            continue
        if any(x.endswith(b"run_3gpc.py") for x in args) and any(x.rstrip(b"/") == key for x in args):
            pids.append(int(p))
    return pids


def kill_producer(shm):
    for pid in producer_pids(shm):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def wait_box(shm, cid, timeout, check_every=30.0):
    """-> list of this node's global slab ids, "skip", or None (producer gone)."""
    d = os.path.join(shm, str(cid))
    f = os.path.join(d, "local_slabs.json")
    t0 = t_chk = time.time()
    missing = 0
    while not os.path.exists(f):
        if os.path.exists(os.path.join(d, "SKIP")):
            return "skip"
        if os.path.exists(os.path.join(shm, "producer.failed")) or os.path.exists(os.path.join(shm, "producer.done")):
            if not os.path.exists(f):
                return None
        if time.time() - t_chk > check_every:
            t_chk = time.time()
            missing = 0 if producer_pids(shm) else missing + 1
            if missing >= 2 and not os.path.exists(f):
                log(f"producer process is gone while waiting for box {cid}")
                open(os.path.join(shm, "producer.failed"), "a").close()
                return None
        if time.time() - t0 > timeout:
            log(f"no slabs for box {cid} after {timeout:.0f} s with the producer alive -> killing it")
            open(os.path.join(shm, "producer.failed"), "a").close()
            kill_producer(shm)
            return None
        time.sleep(2.0)
    return json.load(open(f))


def merge(cid, par, cat_dir, timing):
    d = C.parts_dir(cid, cat_dir)
    C.wait_for(lambda: all(os.path.exists(os.path.join(d, f"cat{g}.npy")) for g in range(C.NSLAB)),
               7200, poll=5.0, what=f"partial catalogs of {cid}")
    parts = [np.load(os.path.join(d, f"cat{g}.npy")) for g in range(C.NSLAB)]
    tms = [json.load(open(os.path.join(d, f"timing{g}.json"))) for g in range(C.NSLAB)]
    cat = np.concatenate(parts).astype(np.float32)
    out = C.cat_path(cid, cat_dir)
    np.save(out + ".tmp.npy", cat)
    meta = dict(cid=int(cid), params_Om_Ob_h_ns_s8=[float(x) for x in par], ic_seed=C.ic_seed(cid),
                box=C.BOX, res=C.RES, res_pm=C.RES_PM, grid=C.GRID, z=0.5, n_halos=int(len(cat)),
                columns="x y z M vx vy vz c", emu_seed_base=C.emu_seed(cid, 0), timing_per_slab=tms)
    with open(out.replace(".npy", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(out + ".tmp.npy", out)
    shutil.rmtree(d, ignore_errors=True)
    return len(cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=int, required=True, help="local GPU / worker index 0..3")
    ap.add_argument("--shm", required=True)
    ap.add_argument("--cat-dir", default=C.CAT_DIR)
    ap.add_argument("--chunk", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args()
    j = a.local
    C.wait_for(lambda: os.path.exists(os.path.join(a.shm, "cids.txt")), 1800, what="cids.txt")
    cids = [int(x) for x in open(os.path.join(a.shm, "cids.txt")).read().split()]
    pars = C.cosmologies()
    t0 = time.time()
    em = emu.Emulator()
    torch = em.torch
    log(f"worker {j}: model ready ({time.time() - t0:.0f} s), {len(cids)} boxes")
    rows_x = C.GRID // C.NSLAB
    for cid in cids:
        tt = time.time()
        slabs = wait_box(a.shm, cid, a.timeout)
        if slabs is None:
            log(f"worker {j}: producer stopped before box {cid}; exiting")
            break
        if slabs == "skip":
            log(f"worker {j}: box {cid} skipped by producer (halo too small)")
            open(C.gpu_free_flag(a.shm, cid, j), "w").close()
            continue
        g = slabs[j]
        f = os.path.join(a.shm, str(cid), f"slab{g}.npy")
        t_wait = time.time() - tt
        t1 = time.time()
        fields = np.load(f, mmap_mode="r")
        assert fields.shape == (C.ROWS, C.GS, C.GS, C.GS, 18), fields.shape
        tok = em.generate(fields, pars[cid], C.emu_seed(cid, g), chunk=a.chunk)
        del fields
        os.remove(f)
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        open(C.gpu_free_flag(a.shm, cid, j), "w").close()
        t_gen = time.time() - t1
        t1 = time.time()
        cat = emu.decode(tok, C.GRID, C.BOX, x0=g * rows_x, rng=np.random.default_rng(C.emu_seed(cid, g)))
        timing = dict(slab=g, host=os.uname().nodename.split(".")[0], wait=round(t_wait, 1),
                      generate=round(t_gen, 1), decode=round(time.time() - t1, 1), peak_gpu_gb=round(peak, 1),
                      n_halos=int(len(cat)))
        d = C.parts_dir(cid, a.cat_dir)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"timing{g}.json"), "w") as fh:
            json.dump(timing, fh)
        np.save(os.path.join(d, f"cat{g}.tmp.npy"), cat)
        os.replace(os.path.join(d, f"cat{g}.tmp.npy"), os.path.join(d, f"cat{g}.npy"))
        log(f"worker {j} box {cid}: " + json.dumps(timing))
        if g == 0:
            n = merge(cid, pars[cid], a.cat_dir, timing)
            log(f"CATALOG {cid}: {n} halos -> {C.cat_path(cid, a.cat_dir)} ({time.time() - tt:.0f} s since box wait)")


if __name__ == "__main__":
    main()
