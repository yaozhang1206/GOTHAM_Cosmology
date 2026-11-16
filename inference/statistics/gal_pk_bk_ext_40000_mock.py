"""
Mock-only Pk/Bk generator — 4000 cosmologies, 10 HOD realizations each.

Reads cosmology from LH_points_HOD_cosmo_ext_40000.txt (columns 10-14).
Saves Pk and Bk for mock galaxy catalogs only.

Usage (via srun):
  srun --mpi=pmi2 python gal_pk_bk_ext_40000_mock.py <start> <end> [noise]
"""

import dill
import sys
import os
import numpy as np
import Pk_library as PKL
import MAS_library as MASL
from astropy.cosmology import FlatLambdaCDM
from mpi4py import MPI

sys.path.insert(0, '/u/yzhang116/ltu-cmass')
from cmass.bias.tools.hod_models import Zheng07
from halotools.sim_manager import UserSuppliedHaloCatalog
from halotools.empirical_models import halo_mass_to_halo_radius, NFWProfile

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MOCK_DIR    = '/work/hdd/bdne/spandey3/quijote_LH_discodj/quijote_generated_halo_cats'
SAVE_PK_DIR = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Pk_all_noise/'
SAVE_BK_DIR = '/work/hdd/bdne/yzhang116/quijote_galaxy/wide_hod_ext_40000/Bk_all_noise/'
LH_FILE     = '/work/hdd/bdne/yzhang116/quijote_galaxy/LH_points_HOD_cosmo_ext_40000.txt'

N_COSMO  = 4000
N_HOD    = 10
Z        = 0.5
BOXSIZE  = 1000.
K1_ARRAY = [0.08, 0.16, 0.32]
THETA_BK = np.linspace(0.1, np.pi - 0.1, 8)

nthreads = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

# Load LH table once to avoid repeated I/O
LH_POINTS = np.loadtxt(LH_FILE)

print(f"Rank {rank}/{size}: nthreads={nthreads}", flush=True)


# ---------------------------------------------------------------------------
# Halo catalog helpers
# ---------------------------------------------------------------------------

def _build_halo_catalog(pos, vel, mass, cosmo, conc, mdef='200c'):
    BoxSize = np.array([BOXSIZE, BOXSIZE, BOXSIZE])
    mkey = f'halo_m{mdef}'
    rkey = f'halo_r{mdef}'
    radius = halo_mass_to_halo_radius(mass, cosmo, Z, mdef)
    kws = dict(
        halo_x=pos[:, 0], halo_y=pos[:, 1], halo_z=pos[:, 2],
        halo_vx=vel[:, 0], halo_vy=vel[:, 1], halo_vz=vel[:, 2],
        halo_nfw_conc=conc,
        halo_redshift=np.full(len(mass), Z),
        halo_id=np.arange(len(mass)),
        halo_hostid=np.zeros(len(mass), dtype=int),
        halo_upid=np.full(len(mass), -1.0),
        halo_local_id=np.arange(len(mass), dtype='i8'),
        cosmology=cosmo, redshift=Z, particle_mass=1, Lbox=BoxSize, mdef=mdef,
    )
    kws[mkey] = mass
    kws[rkey] = radius
    if mdef != 'vir':
        kws['halo_mvir'] = np.full(len(mass), np.nan)
        kws['halo_rvir'] = np.full(len(mass), np.nan)
    return UserSuppliedHaloCatalog(**kws)


def _build_HOD_model(cosmo, theta_hod, mdef='200c'):
    model = Zheng07(mass_def=mdef, assem_bias=True, vel_assem_bias=True)
    model.set_parameters(dict(theta_hod))
    model.set_occupation()
    model.set_profiles(cosmology=cosmo, zf=Z)
    return model.get_model()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_theta_hod(isim, ihod):
    """Return (theta_hod dict, theta_cosmo dict) for (isim, ihod)."""
    row = LH_POINTS[isim + N_COSMO * ihod]
    logMmin, sigma_logM, logM0, logM1, alpha = row[:5]
    abias_cen, abias_sat, eta_cen, eta_sat, eta_conc = row[5:10]
    Om, Ob, h, ns, s8, LH_id_cosmo = row[10:]
    theta_hod = {
        'logMmin': logMmin, 'sigma_logM': sigma_logM,
        'logM0': logM0, 'logM1': logM1, 'alpha': alpha,
        'mean_occupation_centrals_assembias_param1':   abias_cen,
        'mean_occupation_satellites_assembias_param1': abias_sat,
        'eta_vb_centrals':          eta_cen,
        'eta_vb_satellites':        eta_sat,
        'conc_gal_bias_satellites': eta_conc,
    }
    theta_cosmo = dict(Om=Om, Ob=Ob, h=h, ns=ns, sigma8=s8, LH_id_cosmo=LH_id_cosmo)
    return theta_hod, theta_cosmo


def load_mock_halos(isim):
    mock = np.load(MOCK_DIR + f'/generated_halo_catalog_{isim}.npy')
    return dict(pos=mock[:, :3].copy(), vel=mock[:, 4:7],
                mass=mock[:, 3], conc=mock[:, 7])


# ---------------------------------------------------------------------------
# Galaxy catalog
# ---------------------------------------------------------------------------

def get_gal_cats(halos, cosmo_astropy, theta_hod, seed=0):
    catalog = _build_halo_catalog(halos['pos'], halos['vel'], halos['mass'],
                                   cosmo_astropy, halos['conc'])
    hod_factory = _build_HOD_model(cosmo_astropy, theta_hod)
    hod_factory.populate_mock(catalog, seed=seed, halo_mass_column_key='halo_m200c')
    gal = hod_factory.mock.galaxy_table
    pos = np.array([gal['x'], gal['y'], gal['z']]).T
    vel = np.array([gal['vx'], gal['vy'], gal['vz']]).T
    gal_type = (np.array(gal['gal_type']) == 'satellites').astype(int)
    n_total = len(pos)
    n_sat   = int(gal_type.sum())
    galsum  = dict(total=n_total, number_density=n_total / BOXSIZE**3,
                   centrals=n_total - n_sat, satellites=n_sat,
                   fsat=n_sat / n_total)
    return pos, vel, galsum


def get_rsd_pos(pos, vel, rsd_factor, noise=0., los=(1, 0, 0)):
    out = (pos + vel * rsd_factor * np.asarray(los)).astype(np.float32) % BOXSIZE
    if noise > 0:
        out += np.random.normal(0, noise, size=out.shape).astype(np.float32)
        out %= BOXSIZE
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def get_gal_Pk(gal_pos, grid=512, MAS='NGP', kmax=1.0):
    mesh = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.MA(gal_pos, mesh, BOXSIZE, MAS)
    mesh /= np.mean(mesh, dtype=np.float32)
    mesh -= 1.0
    Pk  = PKL.Pk(mesh, BOXSIZE, axis=0, MAS=None, threads=nthreads)
    sel = np.where((Pk.k3D >= 0.01) & (Pk.k3D <= kmax))[0]
    return Pk.Pk[sel, :], Pk.k3D[sel]


def get_gal_Bk(k1, pos, grid=256, MAS='NGP'):
    mesh = np.zeros((grid, grid, grid), dtype=np.float32)
    MASL.MA(pos, mesh, BOXSIZE, MAS)
    mesh /= np.mean(mesh, dtype=np.float32)
    mesh -= 1.0
    return PKL.Bk(mesh, BOXSIZE, k1, k1, THETA_BK, None, nthreads)


# ---------------------------------------------------------------------------
# Main per-sim function
# ---------------------------------------------------------------------------

def process_sim(isim, noise=0., kmax=1.0):
    pk_path = SAVE_PK_DIR + f'Pk_NGP_galnoise_{noise:.2f}_mock_LH_{isim}.dill'
    if os.path.exists(pk_path):
        print(f"Rank {rank}: skipping {isim} (exists).", flush=True)
        return

    # Cosmology from LH file (ihod=0 row; cosmo columns are identical for all ihod)
    _, theta_cosmo0 = get_theta_hod(isim, ihod=0)
    Om, Ob, h = theta_cosmo0['Om'], theta_cosmo0['Ob'], theta_cosmo0['h']
    Ol  = 1.0 - Om
    Hz  = 100.0 * np.sqrt(Om * (1.0 + Z)**3 + Ol)
    rsd_factor   = (1.0 + Z) / Hz
    cosmo_astropy = FlatLambdaCDM(H0=100.0 * h, Om0=Om, Ob0=Ob)

    halos_mock = load_mock_halos(isim)

    saved_k = {'isim': isim}
    saved_j = {'isim': isim, 'theta_Bk': THETA_BK, 'k_array': K1_ARRAY}

    for ihod in range(N_HOD):
        theta_hod, theta_cosmo = get_theta_hod(isim, ihod)
        for d in (saved_k, saved_j):
            d[f'theta_hod_{ihod}']   = theta_hod
            d[f'theta_cosmo_{ihod}'] = theta_cosmo

        pos_mock, vel_mock, galsum_mock = get_gal_cats(halos_mock, cosmo_astropy, theta_hod)
        for d in (saved_k, saved_j):
            d[f'galsum_mock_{ihod}'] = galsum_mock

        rsd_pos_mock = get_rsd_pos(pos_mock, vel_mock, rsd_factor, noise=noise)

        pk_vals, k_vals = get_gal_Pk(rsd_pos_mock, kmax=kmax)
        saved_k[f'rsd_Pk_mock_{ihod}'] = pk_vals
        saved_k[f'k_Pk_{ihod}']        = k_vals

        for k1 in K1_ARRAY:
            Bk_mock = get_gal_Bk(k1, rsd_pos_mock)
            tag = f'0p{int(k1 * 100):02d}'
            saved_j[f'rsd_Bk_mock_{tag}_{ihod}'] = Bk_mock.B
            saved_j[f'rsd_Qk_mock_{tag}_{ihod}'] = Bk_mock.Q

    dill.dump(saved_k, open(pk_path, 'wb'))
    bk_path = SAVE_BK_DIR + f'Bk_NGP_galnoise_{noise:.2f}_mock_LH_{isim}.dill'
    dill.dump(saved_j, open(bk_path, 'wb'))
    print(f"Rank {rank}: saved isim={isim}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    start = int(sys.argv[1])
    end   = int(sys.argv[2])
    noise = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

    print(f"Hello from rank {rank}/{size}", flush=True)
    for simid in range(start + rank, end, size):
        print(f"Rank {rank} processing mock sim {simid}", flush=True)
        try:
            process_sim(simid, noise=noise)
        except Exception as e:
            print(f"Rank {rank}: ERROR on isim={simid}: {e}", flush=True)
