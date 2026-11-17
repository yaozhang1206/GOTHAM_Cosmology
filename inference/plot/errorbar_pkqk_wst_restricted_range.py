"""
Self-contained figure, 1 row x 2 panels (Om, sigma8): posterior std vs true
parameter, overlaying two families of runs in the same axes.  Adapted from
../compare_errorbar_pkqk_full_wst_box/errorbar_combined_pkqk_full_wst_box.py
with two changes:

  * the full-range-trained pk+qk+WST series is NOT plotted -- only the
    restricted-range one is;
  * that series' legend entry spells out the restricted training range, so the
    figure is self-explanatory: unlike every other series here, this model saw
    only a sub-box of the prior during training.

Series (marker, colour), all FULL-RANGE trained except the last:

  - pk kmax family ('o', Blues ramp):
        kmax = 0.3, 0.4, 0.5, 0.6      (pk_test/.../kmax*_noise0.0_ext_200c_40000)
  - pk + qk ('s', orange):
        kmax = 0.6                     (bk_test/.../qk_kmax0.6_noise0.0_ext_200c_40000)
  - pk + qk + WST ('^', red), RESTRICTED-RANGE trained:
        kmax = 0.6                     (all_test/.../qk_kmax0.6_wstgrid256_
                                        avgangles_noise0.0_Om0.15-0.45_s80.65-0.95)

Every series is evaluated on the SAME box-restricted quijote test set --
Om in [0.15, 0.45], sigma8 in [0.65, 0.95], 560 cosmologies -- so the
comparison is like-for-like even though the WST model saw only the box in
training while the pk / pk+qk models saw the full prior:

  * the restricted run's own validation_quijote/ IS that test set;
  * for the full-range runs, the same test set was pushed through the
    full-range ensemble into validation_quijote_box_Om0.15-0.45_s80.65-0.95/
    by pk_test/codes/range_sweep/eval_fullrange_ensemble_on_boxes.py
    (kmax=0.6 and pk+qk) and by
    ../compare_errorbar_pkqk_full_wst_box/eval_fullrange_pk_on_box.py
    (kmax = 0.3, 0.4, 0.5).

All of those come from run_inference_ilias_range.py's _load_test_set with
n_samples=None and seed=32, which fixes both the membership and the row
order, so the theta_test arrays are byte-for-byte identical across series --
asserted at load time rather than assumed. The parameter bins therefore span
the box, not the full prior.

This does NOT re-sample posteriors from the trained ensembles: it reads the
already-saved posterior_samples.npy, so no feature preprocessing has to be
replicated here.

Usage
-----
    python errorbar_pkqk_wst_restricted_range.py [--out fig.png] [--n_bins 4]
"""

import argparse
import logging
import os
from os.path import join

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import glob
# Nimbus Roman (URW Times clone): register explicitly so it is found and embedded.
for _f in glob.glob('/usr/share/fonts/urw-base35/NimbusRoman-*.otf'):
    font_manager.fontManager.addfont(_f)

# MNRAS journal style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_ROOT = "/u/yzhang116/NN/inference"

# ── The Om x sigma8 box the WST run was trained on, and that every series
#    is evaluated on ────────────────────────────────────────────────────────
OM_RANGE     = ("0.15", "0.45")
SIGMA8_RANGE = ("0.65", "0.95")
NOISE = "0.0"
BQ    = "qk"
GRID  = "256"

BOX_TAG = f"_Om{OM_RANGE[0]}-{OM_RANGE[1]}_s8{SIGMA8_RANGE[0]}-{SIGMA8_RANGE[1]}"
# Sub-directory holding a full-range ensemble's predictions on the box test set.
BOX_VDIR = f"validation_quijote_box{BOX_TAG}"

PK_KMAXES = ["0.3", "0.4", "0.5", "0.6"]
# Full-range-trained pk runs; their box predictions live in BOX_VDIR.
PK_RUN_DIRS = [
    (kmax, join(INFER_ROOT, "pk_test", "ilias_runs",
                f"kmax{kmax}_noise{NOISE}_ext_200c_40000"), BOX_VDIR)
    for kmax in PK_KMAXES
]
# Full-range-trained pk+qk run, likewise.
PKQK_RUN_DIR = (join(INFER_ROOT, "bk_test", "ilias_runs",
                     f"{BQ}_kmax0.6_noise{NOISE}_ext_200c_40000"), BOX_VDIR)
# Restricted-range-trained pk+qk+WST run: its ordinary validation_quijote IS
# the box test set.
def pkqkwst_run_dir(suffix="_hodfix"):
    return (join(INFER_ROOT, "all_test", "ilias_runs",
                 f"{BQ}_kmax0.6_wstgrid{GRID}_avgangles_noise{NOISE}{BOX_TAG}{suffix}"),
            "validation_quijote")


PKQKWST_RUN_DIR = pkqkwst_run_dir()

PARAM_NAMES = [r"$\Omega_m$", r"$\sigma_8$"]

# The WST label carries its training range on a second line: it is the only
# series here that did not see the full prior, and the figure is misread
# without that caveat.
WST_RANGE_NOTE = (rf"trained on $\Omega_m\in[{OM_RANGE[0]},{OM_RANGE[1]}]$, "
                  rf"$\sigma_8\in[{SIGMA8_RANGE[0]},{SIGMA8_RANGE[1]}]$")

EXTRA_STYLES = {
    "pkqk":    {"marker": "s", "color": "tab:orange",
                "label": r"$P_\ell(k_{\rm max}=0.6)+Q_0$"},
    "pkqkwst": {"marker": "^", "color": "tab:red",
                "label": (r"$P_\ell(k_{\rm max}=0.6)+Q_0+{\rm WST}$" "\n"
                          rf"({WST_RANGE_NOTE})")},
}


# ──────────────────────────────────────────────────────────────────────────────
#  Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_std_and_theta(run_dir, vsubdir):
    """Posterior std per test obs, from an already-saved validation directory."""
    vdir = join(run_dir, vsubdir)
    samples = np.load(join(vdir, "posterior_samples.npy"))   # (n_post, n_obs, n_params)
    theta   = np.load(join(vdir, "theta_test.npy"))          # (n_obs, n_params)
    std = np.std(samples, axis=0)                             # (n_obs, n_params)
    return std, theta


def build_kmax_run_data():
    run_data = []
    for kmax, run_dir, vsubdir in PK_RUN_DIRS:
        vdir = join(run_dir, vsubdir)
        if not os.path.isdir(vdir):
            logging.warning(
                f"  pk kmax={kmax}: box validation not found: {vdir}\n"
                f"    generate it with:  python ../compare_errorbar_pkqk_full_wst_box/"
                f"eval_fullrange_pk_on_box.py --kmax {kmax}")
            continue
        label = rf"$P_\ell(k_{{\rm max}}={kmax})$"
        logging.info(f"── {label}  {vdir}")
        std, theta = load_std_and_theta(run_dir, vsubdir)
        run_data.append({"label": label, "std": std, "theta": theta})
    return run_data


def build_extra_run_data(wst_suffix=""):
    cases = [("pkqk", PKQK_RUN_DIR), ("pkqkwst", pkqkwst_run_dir(wst_suffix))]
    run_data = []
    for key, (run_dir, vsubdir) in cases:
        style = EXTRA_STYLES[key]
        vdir = join(run_dir, vsubdir)
        if not os.path.isdir(vdir):
            logging.warning(f"  {style['label']}: validation dir not found: {vdir}")
            continue
        logging.info(f"── {key}  {vdir}")
        std, theta = load_std_and_theta(run_dir, vsubdir)
        run_data.append({"key": key, "label": style["label"], "std": std, "theta": theta})
    return run_data


def assert_same_test_cosmologies(run_data):
    """The whole point of this figure: every series must be the same test set.

    All of these validation dirs were built by _load_test_set(n_samples=None,
    seed=32) on the same box, which fixes membership and row order, so the
    theta arrays should match exactly -- not merely as sets.
    """
    ref_label, ref_theta = run_data[0]["label"], run_data[0]["theta"]
    for rd in run_data[1:]:
        if rd["theta"].shape != ref_theta.shape or not np.array_equal(rd["theta"], ref_theta):
            raise ValueError(
                f"test cosmologies differ between {ref_label} {ref_theta.shape} and "
                f"{rd['label']} {rd['theta'].shape} -- this figure requires the "
                f"identical box test set for every series.")
    logging.info(f"All {len(run_data)} series share the same {len(ref_theta)} "
                 f"test cosmologies.")


# ──────────────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _percentiles(rd, pi, bins):
    std   = rd["std"][:, pi]
    theta = rd["theta"][:, pi]
    idx_bins = [
        np.where((theta >= bins[l]) & (theta < bins[l + 1]))[0]
        for l in range(len(bins) - 1)
    ]
    p16 = np.array([np.percentile(std[ib], 16) if len(ib) else np.nan for ib in idx_bins])
    p50 = np.array([np.percentile(std[ib], 50) if len(ib) else np.nan for ib in idx_bins])
    p84 = np.array([np.percentile(std[ib], 84) if len(ib) else np.nan for ib in idx_bins])
    return p16, p50, p84


def plot_overlay(run_data_kmax, run_data_extra, param_bins, out_path):
    bin_centers = [
        [0.5 * (param_bins[pi][l] + param_bins[pi][l + 1]) for l in range(len(param_bins[pi]) - 1)]
        for pi in range(2)
    ]

    cmap   = plt.get_cmap("Blues")
    n_kmax = max(len(run_data_kmax), 1)
    kmax_colors = [cmap(0.4 + 0.6 * i / max(n_kmax - 1, 1)) for i in range(n_kmax)]

    n_total = len(run_data_kmax) + len(run_data_extra)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained")

    for pi in range(2):
        ax      = axes[pi]
        bins    = param_bins[pi]
        centers = np.array(bin_centers[pi])
        bin_width = bins[1] - bins[0]
        offsets = np.linspace(-0.25, 0.25, max(n_total, 1)) * bin_width

        slot = 0
        for i, rd in enumerate(run_data_kmax):
            p16, p50, p84 = _percentiles(rd, pi, bins)
            x = centers + offsets[slot]
            ax.errorbar(
                x, p50, yerr=[p50 - p16, p84 - p50],
                fmt="o", color=kmax_colors[i], label=rd["label"],
                capsize=2, linewidth=1.0, markersize=3.5,
                markeredgewidth=0.75, capthick=1.0,
            )
            slot += 1

        for rd in run_data_extra:
            style = EXTRA_STYLES[rd["key"]]
            p16, p50, p84 = _percentiles(rd, pi, bins)
            x = centers + offsets[slot]
            ax.errorbar(
                x, p50, yerr=[p50 - p16, p84 - p50],
                fmt=style["marker"], color=style["color"], label=rd["label"],
                capsize=2, linewidth=1.0, markersize=4,
                markeredgewidth=0.75, capthick=1.0,
            )
            slot += 1

        ax.set_xlabel(PARAM_NAMES[pi])
        ax.set_ylabel("Posterior std")

    # One figure-level legend under both panels: the WST entry spells out its
    # training range and is far wider than the others, so an in-axes legend
    # sits on top of the data.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncol=3, framealpha=0., edgecolor="0.8", labelspacing=0.4,
               columnspacing=1.2, handlelength=1.0, handletextpad=0.4,
               borderpad=0.2, fontsize=7.5)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.06)
    base, _ext = os.path.splitext(out_path)
    for ext in ("pdf", "png"):
        fig.savefig(f"{base}.{ext}", dpi=300)
    plt.close(fig)
    logging.info(f"Figure saved → {base}.pdf / .png")


# ──────────────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Overlay the full-range-trained pk-kmax family and pk+qk with "
                    "the restricted-range-trained pk+qk+WST run, all on the same "
                    "box test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_bins", type=int, default=4,
                        help="Number of bins to split each parameter's box range into.")
    parser.add_argument("--wst_suffix", default="_hodfix",
                        help="Run-directory suffix of the pk+qk+WST box run.")
    parser.add_argument("--out", default=None,
                        help="Output figure path (default: "
                             "errorbar_pkqk_wst_restricted_range<box><wst_suffix>.png here)")
    args = parser.parse_args()
    if args.out is None:
        args.out = join(SAVE_DIR, f"errorbar_pkqk_wst_restricted_range{BOX_TAG}.png")

    om_bins = list(np.linspace(float(OM_RANGE[0]), float(OM_RANGE[1]), args.n_bins + 1))
    s8_bins = list(np.linspace(float(SIGMA8_RANGE[0]), float(SIGMA8_RANGE[1]), args.n_bins + 1))
    param_bins = [om_bins, s8_bins]

    logging.info("Loading full-range pk-kmax family (0.3, 0.4, 0.5, 0.6) on the box test set …")
    run_data_kmax = build_kmax_run_data()

    logging.info("Loading pk+qk (full range) and pk+qk+WST (restricted range) …")
    run_data_extra = build_extra_run_data(args.wst_suffix)

    if not run_data_kmax or not run_data_extra:
        logging.error("Missing data for one or both series groups, aborting.")
        raise SystemExit(1)

    assert_same_test_cosmologies(run_data_kmax + run_data_extra)

    out_path = os.path.normpath(args.out)
    plot_overlay(run_data_kmax, run_data_extra, param_bins, out_path)
    logging.info("Done.")


if __name__ == "__main__":
    main()
