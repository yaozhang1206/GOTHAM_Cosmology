"""
Rebuild the WST 40000 train/test splits after the HOD-mismatch fix, then
check theta_all row alignment against the Pk and Bk splits.

isim 0-1999: new per-LH WST files (computed on DeltaAI from positions painted
             with LH_points_HOD_cosmo_ext_40000.txt), mock and truth in
             separate files, in /u/yzhang116/NN/inference/wide_hod_ext/WST_all_noise/
isim 2000-3999: the per-LH files in wide_hod_ext_40000/WST_all_noise/ are gone,
             so these rows are copied from the previous 40000 train split
             (their HOD already matched Pk; verified in job 24029369).

The previous splits are renamed to *.pre_hodfix.dill (once) before the new
ones are written under the canonical names that the SBI drivers read.

Usage: python prepare_wst_train_test_40000_hodfix.py [noise]
"""
import os
import sys

import dill
import numpy as np

noise = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
grid, model, nhod = 256, 'angular', 10

WDIR = '/u/yzhang116/NN/inference/wide_hod_ext/WST_all_noise/'
MOCK = WDIR + f'WST_{model}_NGP_kc34pi_galnoise_{noise:.2f}_mock_galaxy_LH_{{}}.dill'
TRUTH = WDIR + f'WST_{model}_NGP_kc34pi_galnoise_{noise:.2f}_truth_galaxy_LH_{{}}.dill'
SPLIT = WDIR + f'WST_nbar_{model}_NGP_kc34pi_40000_galnoise_{noise:.2f}_{{}}.dill'
PK = f'/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Pk_all_noise/Pk_nbar_NGP_galnoise_{noise:.2f}_200c_{{}}.dill'
BK = f'/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Bk_all_noise/Bk_NGP_galnoise_{noise:.2f}_200c_{{}}.dill'
LH40 = '/work/hdd/bdne/yzhang116/quijote_galaxy/LH_points_HOD_cosmo_ext_40000.txt'
saved_idx = set(np.loadtxt('/u/yzhang116/NN/inference/quijote_saved_idx.txt', dtype=int).tolist())


def load(path):
    with open(path, 'rb') as f:
        return dill.load(f)


def wst_vec(s, kind, ihod):
    return np.concatenate([s[f'rsd_s{j}_{kind}_{ihod}_{grid}'] for j in range(3)])


def theta_of(s, ihod):
    return np.array(list(s[f'theta_cosmo_{ihod}'].values()) + list(s[f'theta_hod_{ihod}'].values()), dtype=float)


# Show the key layout of the new files once
for p in (MOCK.format(0), TRUTH.format(min(saved_idx))):
    s = load(p)
    print(os.path.basename(p), 'keys:', sorted(s.keys())[:12], '...', len(s), 'keys total', flush=True)

# ---- previous splits (source for isim >= 2000 rows) ----
old = {sp: (SPLIT.format(sp).replace('.dill', '.pre_hodfix.dill')
            if os.path.exists(SPLIT.format(sp).replace('.dill', '.pre_hodfix.dill')) else SPLIT.format(sp))
       for sp in ('train', 'test')}
old_train = load(old['train'])
old_theta = np.asarray(old_train['theta_all'], dtype=float)
old_isim = old_theta[:, 5].astype(int)
hi = old_isim >= 2000
print(f'old train split {old["train"]}: {len(old_theta)} rows, {hi.sum()} with isim>=2000', flush=True)

# ---- build isim < 2000 from new per-LH files ----
tr_w, tr_t, tr_n = [], [], []
te_wm, te_wt, te_t, te_nm, te_nt = [], [], [], [], []
truth_theta_mismatch = 0
errors = []
for isim in range(2000):
    try:
        sm = load(MOCK.format(isim))
        is_test = isim in saved_idx
        st = load(TRUTH.format(isim)) if is_test else None
        for ihod in range(nhod):
            th = theta_of(sm, ihod)
            wm = wst_vec(sm, 'mock', ihod)
            nm = sm[f'galsum_mock_{ihod}']['number_density']
            if is_test:
                if not np.allclose(theta_of(st, ihod), th, rtol=0, atol=1e-8):
                    truth_theta_mismatch += 1
                te_wm.append(wm); te_wt.append(wst_vec(st, 'truth', ihod)); te_t.append(th)
                te_nm.append(nm); te_nt.append(st[f'galsum_truth_{ihod}']['number_density'])
            else:
                tr_w.append(wm); tr_t.append(th); tr_n.append(nm)
    except Exception as e:
        errors.append(isim)
        print(f'ERROR isim={isim}: {type(e).__name__}: {e}', flush=True)

print(f'per-LH read errors: {len(errors)} {errors[:20]}', flush=True)
print(f'truth-file theta != mock-file theta rows: {truth_theta_mismatch}', flush=True)
if errors:
    sys.exit('Aborting: missing/bad per-LH WST files; splits not written.')

wkey = f'wst_mock_{grid}_all'
new_lo = np.array(tr_w)
if new_lo.shape[1] != np.asarray(old_train[wkey]).shape[1]:
    sys.exit(f'Aborting: WST length {new_lo.shape[1]} != old {np.asarray(old_train[wkey]).shape[1]}')

train = {
    wkey: np.concatenate([new_lo, np.asarray(old_train[wkey])[hi]]),
    'theta_all': np.concatenate([np.array(tr_t), old_theta[hi]]),
    'nbar_mock': np.concatenate([np.array(tr_n), np.asarray(old_train['nbar_mock'])[hi]]),
}
test = {
    wkey: np.array(te_wm),
    f'wst_truth_{grid}_all': np.array(te_wt),
    'theta_all': np.array(te_t),
    'nbar_mock': np.array(te_nm),
    'nbar_truth': np.array(te_nt),
}
for name, d in (('train', train), ('test', test)):
    print(name, {k: np.shape(v) for k, v in d.items()}, flush=True)
    if not np.isfinite(d[wkey]).all():
        print(f'  WARNING: non-finite values in {name} {wkey}', flush=True)

# How much did the isim<2000 WST change vs the old (mismatched-HOD) split?
olo = ~hi
d = np.abs(new_lo - np.asarray(old_train[wkey])[olo]) / (np.abs(np.asarray(old_train[wkey])[olo]) + 1e-30)
print(f'isim<2000 train WST relative change vs old split: median {np.median(d):.3g}, '
      f'rows identical {(d.max(axis=1) == 0).sum()}/{len(d)}', flush=True)

# ---- theta alignment checks ----
lh40 = np.loadtxt(LH40)
ok_all = True
for name, d in (('train', train), ('test', test)):
    tw = d['theta_all']
    pk = load(PK.format(name)); tp = np.asarray(pk['theta_all'], dtype=float)
    tb = np.asarray(load(BK.format(name))['theta_all'], dtype=float)
    print(f'=== {name}: Pk {tp.shape}  Bk {tb.shape}  WST {tw.shape} ===')
    for lbl, a, b in (('Pk vs Bk ', tp, tb), ('Pk vs WST', tp, tw), ('Bk vs WST', tb, tw)):
        if a.shape != b.shape:
            print(f'  {lbl}: SHAPE MISMATCH'); ok_all = False; continue
        cos = np.isclose(a[:, :6], b[:, :6], rtol=0, atol=1e-6).all(axis=1)
        hod = np.isclose(a[:, 6:], b[:, 6:], rtol=0, atol=1e-6).all(axis=1)
        print(f'  {lbl}: cosmology equal {cos.sum()}/{len(cos)}, HOD equal {hod.sum()}/{len(hod)}, '
              f'max |diff| {np.abs(a - b).max():.2e}')
        ok_all &= bool(cos.all() and hod.all())
        if not hod.all():
            isim = a[:, 5].astype(int)
            print(f'    mismatched isim: {np.unique(isim[~hod])[:20]}')
    isim = tw[:, 5].astype(int); ihod = np.tile(np.arange(nhod), len(tw) // nhod)
    m40 = np.isclose(tw[:, 6:], lh40[isim + 4000 * ihod, :10], rtol=0, atol=1e-6).all(axis=1)
    print(f'  WST HOD == LH_ext_40000 table: {m40.sum()}/{len(tw)}')
    ok_all &= bool(m40.all())
    for nk in ('nbar_mock', 'nbar_truth'):
        if nk in pk and nk in d:
            r = np.asarray(d[nk]) / np.asarray(pk[nk])
            print(f'  {nk} WST/Pk: min {r.min():.6f} max {r.max():.6f}')

print('ALL THETA CHECKS PASS' if ok_all else 'THETA CHECKS FAILED', flush=True)
if not ok_all:
    sys.exit('Not writing splits because the theta checks failed.')

# ---- write: keep previous splits as *.pre_hodfix.dill ----
for sp, d in (('train', train), ('test', test)):
    path = SPLIT.format(sp)
    bak = path.replace('.dill', '.pre_hodfix.dill')
    if not os.path.exists(bak):
        os.rename(path, bak)
        print(f'renamed old {sp} split -> {bak}')
    with open(path, 'wb') as f:
        dill.dump(d, f)
    print(f'saved {sp} -> {path}', flush=True)
