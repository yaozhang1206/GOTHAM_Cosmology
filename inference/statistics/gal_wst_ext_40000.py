"""WST (angular, NGP, kc=3pi/4, J=6, Q=1, grid 256) of mock and truth galaxies.

Same operator and mesh as gal_wst_angular_grid256_mock_rerun.py, run over every
Galaxy_pos_galnoise_{noise}_{kind}_LH_{isim}.dill in IN_DIRS. When a sim exists in
several input dirs, the first dir in IN_DIRS wins (the /work copies of the sims
duplicated in gal_position/ are truncated).

After each sim's output is written, it is reloaded and validated; if it passes,
every copy of that sim's galaxy-position file is deleted (DELETE_INPUTS).
Inputs of failed sims are kept.

Usage:
  python gal_wst_ext_40000.py <kind: mock|truth|both> [shard] [nshards] [noise]
  shard/nshards default to SLURM_PROCID/SLURM_NTASKS (0/1 outside srun).
Outputs that already exist are skipped, so the script can be rerun to resume.
"""
import dill
import sys, os, glob, re, time
import numpy as np

DATA_DIR = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000'
OUT_DIR = '/u/yzhang116/NN/analyze/WST_all_noise'  # /work/hdd near quota
IN_DIRS = [f'{OUT_DIR}/gal_position',               # complete copies, preferred
           f'{DATA_DIR}/galaxy_rsd_pos_noise']
NHOD = 10
GRID = 256
DELETE_INPUTS = True


def get_gal_mesh(pos, boxsize=1000., grid=256, MAS='NGP'):
    import MAS_library as MASL
    mesh = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.MA(pos, mesh, boxsize, MAS)
    mesh /= np.mean(mesh, dtype=np.float32)
    mesh -= 1.0
    return mesh


def in_name(kind, noise, isim):
    return f'Galaxy_pos_galnoise_{noise:.2f}_{kind}_LH_{isim}.dill'


def input_copies(kind, noise, isim):
    """All existing copies of this sim's position file, in IN_DIRS priority order."""
    return [p for p in (f'{d}/{in_name(kind, noise, isim)}' for d in IN_DIRS) if os.path.exists(p)]


def outfile(kind, noise, isim):
    return f'{OUT_DIR}/WST_angular_NGP_kc34pi_galnoise_{noise:.2f}_{kind}_galaxy_LH_{isim}.dill'


def list_isims(kind, noise):
    pat = re.compile(rf'_{kind}_LH_(\d+)\.dill$')
    isims = set()
    for d in IN_DIRS:
        for f in glob.glob(f'{d}/Galaxy_pos_galnoise_{noise:.2f}_{kind}_LH_*.dill'):
            isims.add(int(pat.search(f).group(1)))
    return sorted(isims)


def validate_output(path, kind, isim, nhod=NHOD, grid=GRID):
    """Reload a WST output file; return None if OK, else a reason string."""
    try:
        d = dill.load(open(path, 'rb'))
    except Exception as e:
        return f'unreadable output: {type(e).__name__}: {e}'
    if d.get('isim') != isim:
        return f'isim mismatch: {d.get("isim")}'
    n1 = n2 = None
    for ih in range(nhod):
        for k in (f'theta_hod_{ih}', f'theta_cosmo_{ih}', f'galsum_{kind}_{ih}'):
            if k not in d:
                return f'missing key {k}'
        for s in ('s0', 's1', 's2'):
            k = f'rsd_{s}_{kind}_{ih}_{grid:d}'
            if k not in d:
                return f'missing key {k}'
            a = np.asarray(d[k])
            if a.size == 0 or not np.all(np.isfinite(a)):
                return f'empty or non-finite {k}'
        l1, l2 = d[f'rsd_s1_{kind}_{ih}_{grid:d}'].size, d[f'rsd_s2_{kind}_{ih}_{grid:d}'].size
        if n1 is None:
            n1, n2 = l1, l2
        elif (l1, l2) != (n1, n2):
            return f'inconsistent s1/s2 length at hod {ih}: {(l1, l2)} vs {(n1, n2)}'
    return None


def get_wavelets_all_hods(kind, isim, noise, wst_op, device, src, nhod_LH_samp=NHOD, grid=GRID):
    import torch
    gal_pos = dill.load(open(src, 'rb'))
    saved_j = {'isim': isim}

    for ihod in range(nhod_LH_samp):
        saved_j[f'theta_hod_{ihod}'] = gal_pos[f'theta_hod_{ihod}']
        saved_j[f'theta_cosmo_{ihod}'] = gal_pos[f'theta_cosmo_{ihod}']
        saved_j[f'galsum_{kind}_{ihod}'] = gal_pos[f'galsum_{kind}_{ihod}']

        pos = np.ascontiguousarray(gal_pos[f'pos_rsd_{kind}_{ihod}'], dtype=np.float32)
        mesh = get_gal_mesh(pos, grid=grid)

        df = torch.from_numpy(mesh).to(device)
        s0, s1, s2 = wst_op(df)
        saved_j[f'rsd_s0_{kind}_{ihod}_{grid:d}'] = s0.cpu().numpy()
        saved_j[f'rsd_s1_{kind}_{ihod}_{grid:d}'] = s1.cpu().numpy().flatten()
        saved_j[f'rsd_s2_{kind}_{ihod}_{grid:d}'] = s2.cpu().numpy().flatten()

    out = outfile(kind, noise, isim)
    tmp = out + '.tmp'
    with open(tmp, 'wb') as f:
        dill.dump(saved_j, f)
    os.replace(tmp, out)


def verify_and_delete(kind, isim, noise, tag=''):
    """Validate the output; on success delete all input copies. Returns True if valid."""
    reason = validate_output(outfile(kind, noise, isim), kind, isim)
    if reason is not None:
        print(f'{tag} INVALID {kind} isim={isim}: {reason}; inputs kept', flush=True)
        return False
    if DELETE_INPUTS:
        for p in input_copies(kind, noise, isim):
            os.remove(p)
            print(f'{tag} deleted {p}', flush=True)
    return True


if __name__ == "__main__":
    import torch
    from galactic_wavelets.scattering_operator import ScatteringOp

    kind_arg = sys.argv[1]
    shard = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get('SLURM_PROCID', 0))
    nshards = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get('SLURM_NTASKS', 1))
    noise = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    kinds = ['truth', 'mock'] if kind_arg == 'both' else [kind_arg]
    tag = f'[shard {shard}]'

    # truth first (only 100 sims), then mock; interleave across shards
    jobs = [(k, i) for k in kinds for i in list_isims(k, noise)]
    jobs = jobs[shard::nshards]

    # outputs already present from an earlier session: validate, clean up inputs left behind
    todo = []
    for k, i in jobs:
        if os.path.exists(outfile(k, noise, i)):
            if input_copies(k, noise, i):
                verify_and_delete(k, i, noise, tag)
        else:
            todo.append((k, i))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"{tag} nshards={nshards} device={device} assigned={len(jobs)} todo={len(todo)}", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    wst_op = ScatteringOp(torch.Size([GRID, GRID, GRID]), 6, 1,
                          moments=[1/2, 1, 2],
                          kc=np.pi * 0.75,
                          scattering=True,
                          angular_width=np.pi/4.,
                          device=device,
                          los=[1, 0, 0],
                          use_mp=False)

    for n, (kind, isim) in enumerate(todo):
        t0 = time.time()
        src = input_copies(kind, noise, isim)
        if not src:
            print(f"{tag} FAILED {kind} isim={isim}: input vanished", flush=True)
            continue
        try:
            get_wavelets_all_hods(kind, isim, noise, wst_op, device, src[0])
        except Exception as e:
            print(f"{tag} FAILED {kind} isim={isim} src={src[0]}: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"{tag} Done {kind} isim={isim} ({n+1}/{len(todo)}, {time.time()-t0:.1f}s)", flush=True)
        verify_and_delete(kind, isim, noise, tag)
