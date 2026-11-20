"""3 Gpc/h producer: one JAX process per node (2 nodes x 4 GPUs), DISCO-DJ + GPU emulator fields -> /dev/shm.

For each cosmology id (cid) without a finished catalog:
  IC    : ic_worker.py (N-GenIC phases, seed = cid, CAMB/Quijote P(k), z = 0) as a CPU subprocess on every
          node, prefetched one box ahead; scaled to a_ic = 1/128 for pm_multigpu's convention.
  N-body + fields: pm_multigpu_gpufields.run_one_simulation_multigpu in sink mode (DISCO_FIELD_SINK_DIR):
          IC -> fphi -> lean 2LPT -> slab-mesh fastpm 10 + 5 + 5 steps to z = 2, 1, 0.5; after each snapshot the
          fields of each GPU's x-slab are copied into SHM/{cid}/slab{g}.npy (fp16, (884736, 8, 8, 8, 18)).
          Then SHM/{cid}/local_slabs.json lists this node's slabs g (sorted) for the 4 local emulator workers.
  sync  : before box n the producer waits for SHM/gpu_free_{cid_{n-1}}_{j} of its 4 local workers, then for the
          other node (distributed-runtime barrier), so JAX and torch never share a GPU.
Writes SHM/producer.done (or producer.failed) at exit.  Box timings: lines starting with "BOX".
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common3 as C                                # noqa: E402

os.environ.update({
    "DISCO_BOXSIZE": str(C.BOX), "DISCO_RES": str(C.RES), "DISCO_PM_FACTOR": str(C.RES_PM // C.RES),
    "DISCO_LPT_ORDER": "2", "DISCO_GRID": str(C.GRID), "DISCO_GRID_SBOX": str(C.GS),
    "DISCO_PM_CHECKPOINT": "0", "DISCO_OUTPUT_TAG": "gen3gpc"})
os.environ.setdefault("DISCO_PM_FFT_BACKEND", "JAX_RFFT")
os.environ.setdefault("DISCO_HALO_SIZE", "96")
_stub = types.ModuleType("pm_checkpoint_io")
for _n in ("atomic_json", "restore_process_shards", "save_process_shards"):
    setattr(_stub, _n, lambda *a, **k: (_ for _ in ()).throw(RuntimeError("checkpoint stub")))
sys.modules["pm_checkpoint_io"] = _stub

import argparse                                   # noqa: E402
import json                                       # noqa: E402
import shutil                                     # noqa: E402
import subprocess                                 # noqa: E402
import time                                       # noqa: E402

import numpy as np                                # noqa: E402
import jax                                        # noqa: E402

import pm_multigpu_gpufields as M                 # noqa: E402  (initialises jax.distributed from Slurm)
import ic_worker                                  # noqa: E402  (CPU IC subprocess)
import inspect                                    # noqa: E402
import discodj_dist.lpt.nlpt_distributed as _nd   # noqa: E402

assert "lax.slice_in_dim" in inspect.getsource(_nd.pad_fourier_full), "old (miscompiling) pad_fourier_full"
assert M.res == C.RES and M.res_pm == C.RES_PM and M.boxsize == C.BOX and M.grid == C.GRID
assert M.pm_pdims == (C.NSLAB, 1), f"need a slab PM mesh over {C.NSLAB} GPUs, got {M.pm_pdims}"

PID, NPROC = jax.process_index(), jax.process_count()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] [proc {PID}/{NPROC}] {msg}", flush=True)


def barrier(name, timeout_s=7200):
    """Cross-node barrier in the distributed runtime (no device collective, long timeout)."""
    if NPROC > 1:
        from jax._src import distributed
        distributed.global_state.client.wait_at_barrier(name, timeout_in_ms=int(timeout_s * 1000))


def dplus_aic(par):
    om, ob, h, ns, s8 = (float(x) for x in par)
    dj = M.DiscoDJ(dim=3, res=M.res, device=M.discodj_device, precision=M.precision, boxsize=M.boxsize,
                   cosmo=dict(Omega_c=om - ob, Omega_b=ob, h=h, n_s=ns, sigma8=s8), name="dplus").with_timetables()
    tt = dj.cosmo._timetables
    return float(np.interp(np.log10(M.a_ic), np.log10(np.asarray(tt["a"])), np.asarray(tt["Dplus"])))


def set_cosmology(par):
    om, ob, h, ns, s8 = (float(x) for x in par)
    M.Om, M.Ob, M.h, M.ns, M.sigma8 = om, ob, h, ns, s8
    M.cosmo = dict(Omega_c=om - ob, Omega_b=ob, h=h, n_s=ns, sigma8=s8)
    M.cosmo_source = "gen3gpc cosmology table"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cids", type=int, nargs="+", required=True)
    ap.add_argument("--shm", required=True)
    ap.add_argument("--cat-dir", default=C.CAT_DIR)
    ap.add_argument("--ic-workers", type=int, default=64)
    ap.add_argument("--gpu-timeout", type=float, default=7200.0)
    a = ap.parse_args()
    pars = C.cosmologies()
    todo = [c for c in a.cids if not os.path.exists(C.cat_path(c, a.cat_dir))]
    with open(os.path.join(a.shm, "cids.txt.tmp"), "w") as fh:
        fh.write("\n".join(map(str, todo)) + "\n")
    os.replace(os.path.join(a.shm, "cids.txt.tmp"), os.path.join(a.shm, "cids.txt"))
    log(f"producer: {len(todo)} boxes; {len(jax.devices())} devices, pm_pdims {M.pm_pdims}, halo {M.halo_size} "
        f"(usable {M.halo_usable_cells} PM cells)")
    M.root_out = os.path.join(a.shm, "unused") + "/"
    M._preflight_pm_rfft_backend()
    ic_path = lambda c: os.path.join(a.shm, f"ic_{c}.npy")

    def start_ic(c):
        argv, env = ic_worker.spawn_cmd(c, ic_path(c))
        argv += ["--n", str(C.RES), "--box", str(C.BOX), "--workers", str(a.ic_workers)]
        return subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, text=True)

    def finish_ic(c, proc):
        out, _ = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ic_worker for {c} failed with exit code {proc.returncode}")
        r = json.loads(out.strip().splitlines()[-1])
        return r["path"], r["std"], r["t"]

    procs = {todo[0]: start_ic(todo[0])} if todo else {}
    prev = None
    try:
        for n, cid in enumerate(todo):
            tt = time.perf_counter()
            t = {}
            path, std, t["ic_cpu"] = finish_ic(cid, procs.pop(cid))
            t["ic_wait"] = time.perf_counter() - tt
            if n + 1 < len(todo):
                procs[todo[n + 1]] = start_ic(todo[n + 1])
            t1 = time.perf_counter()
            ic = np.load(path, mmap_mode="r+")          # z = 0 delta -> a_ic (pm_multigpu divides by D+(a_ic))
            ic *= np.float32(dplus_aic(pars[cid]))
            ic.flush()
            del ic
            t["ic_scale"] = time.perf_counter() - t1
            t1 = time.perf_counter()
            if prev is not None:                        # local emulator workers must have released the GPUs
                C.wait_for(lambda: all(os.path.exists(C.gpu_free_flag(a.shm, prev, j)) for j in range(C.NPN)),
                           a.gpu_timeout, what=f"gpu_free for {prev}")
                shutil.rmtree(os.path.join(a.shm, str(prev)), ignore_errors=True)
            barrier(f"box_start_{cid}")
            t["wait_gpu"] = time.perf_counter() - t1
            set_cosmology(pars[cid])
            os.environ["DISCO_IC_PATH"] = path
            sink = os.path.join(a.shm, str(cid))
            os.environ["DISCO_FIELD_SINK_DIR"] = sink
            t1 = time.perf_counter()
            try:
                r = M.run_one_simulation_multigpu(cid, savefull=True)
            except RuntimeError as e:
                if "halo is too small" not in str(e):
                    raise
                os.makedirs(a.cat_dir, exist_ok=True)
                if PID == 0:
                    with open(os.path.join(a.cat_dir, f"needs_halo_{cid}.json"), "w") as fh:
                        json.dump(dict(cid=cid, halo_size=M.halo_size, usable=M.halo_usable_cells, err=str(e)), fh)
                os.makedirs(sink, exist_ok=True)
                open(os.path.join(sink, "SKIP"), "w").close()
                log(f"SKIP {cid}: {e}")
                os.remove(path)
                continue
            t["djfields"] = time.perf_counter() - t1
            t["dj"] = r["t_disco"]
            t["fields"] = sum(r["t_fields"])
            with open(os.path.join(sink, "local_slabs.json.tmp"), "w") as fh:
                json.dump(sorted(r["slabs"]), fh)
            os.replace(os.path.join(sink, "local_slabs.json.tmp"), os.path.join(sink, "local_slabs.json"))
            os.remove(path)
            t["total"] = time.perf_counter() - tt
            log(f"BOX {cid} params {np.round(pars[cid], 5).tolist()} slabs {sorted(r['slabs'])} ic_std {std:.4e} "
                + json.dumps({k: round(v, 1) for k, v in t.items()}))
            prev = cid
        open(os.path.join(a.shm, "producer.done"), "w").close()
    except BaseException:
        open(os.path.join(a.shm, "producer.failed"), "w").close()
        raise
    finally:
        for p in procs.values():
            p.kill()


if __name__ == "__main__":
    main()
    os._exit(0)          # skip jax.distributed shutdown handshakes after the last box
