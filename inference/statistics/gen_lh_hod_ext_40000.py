"""
Generate the HOD x cosmology Latin-hypercube table used by the paper's
galaxy statistics: LH_points_HOD_cosmo_ext_40000.txt.

  40000 rows = 4000 Quijote LH cosmologies x 10 HOD draws.
  Columns 0-9 : 10 extended-HOD parameters (LHS, criterion='cm', seed=0)
  Columns 10-15: Om Ob h ns s8 LH_cosmo_id (cosmology table tiled 10x)

Row index = isim + 4000 * ihod (the indexing get_theta_hod() assumes in
statistics/gal_*_40000_*.py).

Usage:  python gen_lh_hod_ext_40000.py
"""

import numpy as np
from scipy.stats import truncnorm
from smt.sampling_methods import LHS

data_path = '/work/hdd/bdne/yzhang116/quijote_galaxy/'

# ── Parameter definitions ──────────────────────────────────────────────────
# dist: 'uniform' or 'truncnorm'
# For 'uniform':   lo, hi
# For 'truncnorm': lo, hi, loc, scale  (truncation bounds + Gaussian params)

param_defs = {
    'logMmin':    ('truncnorm', 12.0, 14.0, 12.35, 0.13),  # N(12.35,0.13) in [12,14]
    'sigma_logM': ('uniform',   0.1,  0.6),
    'logM0':      ('uniform',  13.0, 15.0),
    'logM1':      ('uniform',  13.0, 15.0),
    'alpha':      ('uniform',   0.0,  1.5),
    'abias_cen':  ('truncnorm', -1.0,  1.0, 0.0, 0.2),     # N(0, 0.2) in [-1, 1]
    'abias_sat':  ('truncnorm', -1.0,  1.0, 0.0, 0.2),     # N(0, 0.2) in [-1, 1]
    'eta_cen':    ('uniform',   0.0,  0.7),
    'eta_sat':    ('uniform',   0.2,  2.0),
    'eta_conc':   ('uniform',   0.2,  2.0),
}

param_names = list(param_defs.keys())
nvar = len(param_names)

# ── LH design in [0,1]^nvar ────────────────────────────────────────────────
num_LHS = 40000
xlimits = np.zeros((nvar, 2))
xlimits[:, 1] = 1.0

sampling = LHS(xlimits=xlimits, criterion='cm', seed=0)
LHS_points = sampling(num_LHS)   # (40000, 10), uniform in [0,1]

# ── Transform each column to its target distribution ──────────────────────
#
# Uniform:   u -> lo + (hi - lo) * u           (linear scale)
# Truncnorm: u -> truncnorm.ppf(u, a, b, loc, scale)   (inverse CDF)
#
# Both preserve the LH stratification because any monotone transform of
# a 1-D LH design is still a valid LH design for the transformed variable.

LHS_points_final = np.zeros_like(LHS_points)

for jv, name in enumerate(param_names):
    u = LHS_points[:, jv]
    defn = param_defs[name]

    if defn[0] == 'uniform':
        lo, hi = defn[1], defn[2]
        LHS_points_final[:, jv] = lo + (hi - lo) * u

    elif defn[0] == 'truncnorm':
        lo, hi, loc, scale = defn[1], defn[2], defn[3], defn[4]
        a = (lo - loc) / scale
        b = (hi - loc) / scale
        LHS_points_final[:, jv] = truncnorm.ppf(u, a, b, loc=loc, scale=scale)

# ── Attach cosmology params (same tiling as original code) ────────────────
LH_cosmo_val_file = '/u/yzhang116/NN/prepare/quijote_LH_params_4000.txt'
LH_cosmo_val_all  = np.loadtxt(LH_cosmo_val_file)              # (4000, 5)
id_LH_cosmo_all   = np.arange(len(LH_cosmo_val_all))
LH_cosmo_val_all_wid = np.concatenate(
    (LH_cosmo_val_all, id_LH_cosmo_all[:, None]), axis=1)      # (4000, 6)

num_tiles = num_LHS // len(LH_cosmo_val_all)                   # 10
LH_cosmo_tiled = np.tile(LH_cosmo_val_all_wid, (num_tiles, 1)) # (40000, 6)

all_params = np.concatenate((LHS_points_final, LH_cosmo_tiled), axis=1)  # (40000, 16)

# ── Save ──────────────────────────────────────────────────────────────────
cosmo_param_names = ['Om', 'Ob', 'h', 'ns', 's8', 'LH_cosmo_id']
header = '     '.join(param_names + cosmo_param_names)

out_file = data_path + 'LH_points_HOD_cosmo_ext_' + str(num_LHS) + '.txt'
np.savetxt(out_file, all_params, header=header)
print(f'Saved {all_params.shape} -> {out_file}')
