"""CPU-side IC builder, run as an independent subprocess of run_2gpc.py (no JAX, no GPU, own GIL).

    python ic_worker.py CID OUT.npy [--n N --box L]      -> prints one JSON line {"path", "std", "t"}

z = 0 linear delta(x), float32 N^3 (default 1024^3, 2000 Mpc/h), written atomically to OUT.npy.
Uses the compiled discodj_native from the conda env: the caller must NOT put the workspace DISCO-DJ/src on
PYTHONPATH (its discodj_native is an uncompiled shim); spawn_cmd() builds the right environment.
Env: conda pm, LD_LIBRARY_PATH=/sw/user/gsl/lib.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/u/yzhang116/.conda/envs/pm/bin/python"


def spawn_cmd(cid, out_path):
    """(argv, env) for subprocess.Popen: same env minus any DISCO-DJ checkout src/ (it holds an uncompiled
    discodj_native shim that would shadow the compiled one in the conda env) on PYTHONPATH."""
    env = dict(os.environ)
    pp = [p for p in env.get("PYTHONPATH", "").split(":")
          if p and not os.path.isdir(os.path.join(p, "discodj_native"))]
    env["PYTHONPATH"] = ":".join([HERE] + pp)
    return [PY, "-u", os.path.join(HERE, "ic_worker.py"), str(cid), out_path], env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cid", type=int)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--box", type=float, default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    import numpy as np
    import discodj_native
    assert hasattr(discodj_native, "_discodj_native"), f"uncompiled discodj_native shim: {discodj_native.__file__}"
    import ic_dj
    import lh_common as C
    t0 = time.time()
    n, box = a.n or C.RES, a.box or C.BOX
    s8 = float(C.cosmologies()[a.cid, 4])
    d = ic_dj.delta_x_host(C.ic_seed(a.cid), n, box, C.pk_file(a.cid), s8, workers=a.workers)
    tmp = a.out + ".tmp.npy"
    np.save(tmp, d)
    os.replace(tmp, a.out)
    print(json.dumps(dict(path=a.out, std=float(d.std()), t=time.time() - t0)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
