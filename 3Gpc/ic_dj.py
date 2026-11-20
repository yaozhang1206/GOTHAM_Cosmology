"""Quijote-equivalent initial conditions generated with DISCO-DJ (CAMB P(k) table + N-GenIC white noise).

Reproduces 2LPTic (WhichSpectrum 2, SphereMode 0, RayleighSampling 1, Phase_flip 0, Nmesh = Nsample = N)
to float precision (Phase 0, validate_ics.py: P(k) ratio 1.00000, cross-corr 1.000000 vs the 2LPTic field):
  * P(k) is interpolated log-log onto a dense table (DISCO-DJ's jnp.interp is linear in (k, P): +0.09 %),
  * the table is normalised to sigma8 with 2LPTic's TopHatSigma2 (DISCO-DJ's trapezoid on the 400 table
    points differs by up to 2e-4), and DISCO-DJ's fix_sigma8 is NOT used.
The growth conventions (2LPTic GrowthFactor vs DISCO-DJ timetables) agree to 6e-6, so the z = 0 field is
used directly (no /Dplus(a_ic) round trip).

    dj = with_quijote_ics(dj, pkfile, seed, sigma8)           # single GPU (pm.py)
    dk = delta_k_host(seed, N, L, pkfile, sigma8)             # host numpy delta(k) at z = 0 (multi-GPU)
"""
import numpy as np

N_TABLE = 20000


def read_table(pkfile):
    return np.loadtxt(pkfile, unpack=True)[:2]


def pk_loglog(table, k):
    kt, pt = table
    return np.exp(np.interp(np.log(np.maximum(k, kt[0])), np.log(kt), np.log(pt)))


def sigma8_tophat(table, r=8.0):
    """2LPTic TopHatSigma2 over the log-log interpolated table (as make_ics.sigma8_of)."""
    k = np.geomspace(table[0][0], min(table[0][-1], 500.0 / r), 200000)
    x = k * r
    w = 3 * (np.sin(x) - x * np.cos(x)) / x ** 3
    return np.sqrt(np.trapezoid(k ** 3 * pk_loglog(table, k) * w ** 2 / (2 * np.pi ** 2), np.log(k)))


def normalised_table(pkfile, sigma8, n=N_TABLE):
    """Dense log-spaced (k, P) at z = 0 [h/Mpc, (Mpc/h)^3], normalised to sigma8 like 2LPTic."""
    table = read_table(pkfile)
    k = np.geomspace(table[0][0], table[0][-1], n)
    return k, pk_loglog(table, k) * (sigma8 / sigma8_tophat(table)) ** 2


def with_quijote_ics(dj, pkfile, seed, sigma8):
    """DiscoDJ object (timetables computed) -> same object with pk_table and ICs (fphi)."""
    import jax.numpy as jnp
    k, p = normalised_table(pkfile, sigma8)
    dj = dj.update(pk_table={"k": jnp.asarray(k, dtype=dj.dtype), "Pk": jnp.asarray(p, dtype=dj.dtype)})
    return dj.with_ics(white_noise_space="real", white_noise_field=dj.get_ngenic_noise(int(seed)))


def kmag_half(N, L, dtype=np.float32):
    k1 = (2 * np.pi * np.fft.fftfreq(N, L / N)).astype(dtype)
    kz = (2 * np.pi * np.fft.rfftfreq(N, L / N)).astype(dtype)
    return np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)


def delta_k_host(seed, N, L, pkfile, sigma8, dtype=np.complex64):
    """numpy-rfftn-convention delta(k) at z = 0 on the N^3 mesh of a box L (Mpc/h); same math as
    DISCO-DJ generate_grf_core(real, base_field=ngenic): W * sqrt(P / L^3) * N^3, DC = Nyquist = 0."""
    from discodj_native import rng_ngenic
    w = np.asarray(rng_ngenic(int(seed), int(N)).get_field()).astype(dtype)   # Nyquist planes already 0
    k, p = normalised_table(pkfile, sigma8)
    kk = kmag_half(N, L)
    amp = np.sqrt(np.interp(kk, k, p).astype(np.float32) / np.float32(L ** 3)) * np.float32(N ** 3)
    del kk
    amp[0, 0, 0] = 0
    w *= amp
    return w


def delta_x_host(seed, N, L, pkfile, sigma8, workers=None):
    """z = 0 linear delta(x), float32 N^3, lattice at i * L / N."""
    import scipy.fft
    dk = delta_k_host(seed, N, L, pkfile, sigma8)
    return scipy.fft.irfftn(dk, s=(N,) * 3, workers=workers, overwrite_x=True).astype(np.float32, copy=False)
