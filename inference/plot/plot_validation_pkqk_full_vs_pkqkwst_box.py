"""
MNRAS-ready validation figure comparing two summary statistics -- Pk+Qk and
Pk+Qk+WST -- where the two rows come from runs trained over DIFFERENT
cosmological parameter ranges:

  row 1  Pk+Qk        NPE trained over the FULL Om x sigma8 prior range
                      (bk_test/ilias_runs/qk_kmax0.6_noise0.0_ext_200c_40000)
  row 2  Pk+Qk+WST    NPE trained on the restricted box
                      Om in [0.15, 0.45], sigma8 in [0.65, 0.95]
                      (all_test/ilias_runs/qk_kmax0.6_wstgrid256_avgangles_
                       noise0.0_Om0.15-0.45_s80.65-0.95)

Each row uses that run's own top-level validation_quijote (blue, "Quijote")
and validation_mock (red, "Emulator") directories -- NOT the
validation_*_box_* sub-directories, so row 1 really is the full-range
validation set while row 2 is the box one. The rows are therefore NOT
evaluated on the same test points: the prediction panels span different
axis ranges by construction, and the coverage/TARP panels are each
internally consistent but describe different priors. Read the comparison as
"how well does each model do on its own domain", not as a like-for-like
statistic-vs-statistic contest (that needs a box-vs-box comparison where
both rows share the same box).

Panels per row: [Om predictions] [sigma8 predictions] [Om coverage]
[sigma8 coverage] [TARP].

This produces paper Fig. 7 (fig07_validation_pkqk_full_vs_pkqkwst_box).
Inputs are the validation_quijote / validation_mock outputs of
../sbi/run_inference_pkqk_ilias.py (row 1) and
../sbi/run_inference_pkqkwst_ilias_range.py (row 2).

WST features going into the Pk+Qk+WST run ARE signed-log10 transformed (see
log_transform_wst in ../sbi/run_inference_pkqkwst_ilias_range.py) -- already
baked into the saved posterior_samples.npy, so nothing extra is needed here.

Usage
-----
    python plot_validation_pkqk_full_vs_pkqkwst_box.py [--row2 box_hodfix|fullrange_hodfix|box_orig]
                                                       [--no-tarp] [--out fig.pdf]
"""

import argparse
import os
from os.path import join, dirname

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import glob
# Nimbus Roman (URW Times clone): register explicitly so it is found and embedded.
for _f in glob.glob('/usr/share/fonts/urw-base35/NimbusRoman-*.otf'):
    font_manager.fontManager.addfont(_f)
import matplotlib.ticker as mticker

# ── MNRAS figure settings (matches ../plot_validation.py) ────────────────────
FIG_W_IN = 7.0        # printed width (double column); included at scale 1

# Fixed panel grid, in inches. Every panel is the same square size; the gaps
# before each panel column hold that panel's y tick labels / y label.
GAP_LEFT    = 0.46    # before (a): tick labels + "Predicted"
GAP_TICKS   = 0.30    # before (b): tick labels only
GAP_YLABEL  = 0.48    # before (c) and TARP: tick labels + y label
GAP_SHARED  = 0.15    # between the two coverage panels (shared y axis)
GAP_RIGHT   = 0.08
ROW_TITLE_H = 0.21    # row title line
PANEL_TTL_H = 0.19    # per-panel title
XLABEL_H    = 0.40    # x tick labels + x label
TOP_PAD     = 0.03
XLABEL_Y_IN = 0.20    # top of every x label, measured down from the axes

# Quijote (blue) / Emulator (red): the original colours.
C_QUIJOTE  = "#4878CF"
C_EMULATOR = "#D65F5F"

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Nimbus Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "axes.labelsize":     9,
    "axes.titlesize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    7.5,
    "axes.linewidth":     0.75,
    "xtick.major.width":  0.75,
    "ytick.major.width":  0.75,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.minor.size":   1.5,
    "ytick.minor.size":   1.5,
    "axes.labelpad":      2.0,
    "axes.titlepad":      3.0,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
})

PARAM_LABELS = [r"$\Omega_\mathrm{m}$", r"$\sigma_8$"]
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))

KMAX  = "0.6"
INFER_ROOT = "/u/yzhang116/NN/inference"

# Row 1 is always Pk+Qk over the full prior.
ROW1 = ("Pk+Qk (full range)",
        rf"$P_\ell(k_{{\rm max}}={KMAX})+Q_0$, trained on the full prior range",
        join(INFER_ROOT, "bk_test", "ilias_runs",
             "qk_kmax0.6_noise0.0_ext_200c_40000"))

_WST_RUNS = join(INFER_ROOT, "all_test", "ilias_runs")
_BOX_LABEL = (rf"$P_\ell(k_{{\rm max}}={KMAX})+Q_0+{{\rm WST}}$, trained on "
              r"$\Omega_\mathrm{m}\in[0.15,0.45]$, $\sigma_8\in[0.65,0.95]$")
_FULL_LABEL = (rf"$P_\ell(k_{{\rm max}}={KMAX})+Q_0+{{\rm WST}}$, "
               "trained on the full prior range")
# Row 2 choices (--row2). "box_orig" is the run behind the original Fig. 7,
# whose WST for LH 0-1999 used a different HOD draw than Pk/Qk; the *_hodfix
# runs use WST recomputed with the 40000 HOD table (row-aligned with Pk/Qk).
ROW2_CHOICES = {
    "box_orig": ("Pk+Qk+WST (box, original)", _BOX_LABEL,
                 join(_WST_RUNS, "qk_kmax0.6_wstgrid256_avgangles_noise0.0_Om0.15-0.45_s80.65-0.95"),
                 "validation_pkqk_full_vs_pkqkwst_box.pdf"),
    "box_hodfix": ("Pk+Qk+WST (box, HOD fixed)", _BOX_LABEL,
                   join(_WST_RUNS, "qk_kmax0.6_wstgrid256_avgangles_noise0.0_Om0.15-0.45_s80.65-0.95_hodfix"),
                   "validation_pkqk_full_vs_pkqkwst_box_hodfix.pdf"),
    "fullrange_hodfix": ("Pk+Qk+WST (full range, HOD fixed)", _FULL_LABEL,
                         join(_WST_RUNS, "qk_kmax0.6_wstgrid256_avgangles_noise0.0_fullrange_hodfix"),
                         "validation_pkqk_full_vs_pkqkwst_fullrange_hodfix.pdf"),
}


# ── Compute predictions ───────────────────────────────────────────────────────

def compute_predictions(samples):
    """Return median, 16th and 84th percentiles over the posterior axis.

    Parameters
    ----------
    samples : (n_post, n_obs, n_params)

    Returns
    -------
    med, lo, hi : each (n_obs, n_params)
    """
    med = np.median(samples, axis=0)
    lo  = np.percentile(samples, 16, axis=0)
    hi  = np.percentile(samples, 84, axis=0)
    return med, lo, hi


# ── Compute coverage ──────────────────────────────────────────────────────────

def compute_ranks(samples, theta_true):
    """Marginal rank of each true parameter within its posterior samples.

    Parameters
    ----------
    samples    : (n_post, n_obs, n_params)
    theta_true : (n_obs, n_params)

    Returns
    -------
    ranks : (n_obs, n_params)  integer counts in [0, n_post]
    """
    return (samples < theta_true[None, ...]).sum(axis=0)


def bootstrap_coverage_band(ranks, n_boot=1000, percentiles=(2.5, 16, 84, 97.5), seed=None):
    """Bootstrap uncertainty band for this curve's own sorted-rank ECDF.

    Resamples `ranks` with replacement n_boot times and applies the same
    sort + normalise-by-max transform used to draw the curve itself, then
    takes percentiles across replicates at each grid position. Default
    percentiles give the 1-sigma (16/84) and 2-sigma (2.5/97.5) intervals,
    matching draw_tarp's mean +/- 1,2 std bands.
    """
    rng = np.random.default_rng(seed)
    ranks = np.asarray(ranks)
    ndata = len(ranks)
    idx = rng.integers(0, ndata, size=(n_boot, ndata))
    boots = np.sort(ranks[idx], axis=1).astype(float)
    boots /= boots[:, -1:]
    return np.percentile(boots, percentiles, axis=0)   # (4, ndata)


# ── Panel drawing helpers ─────────────────────────────────────────────────────

def _param_axis_range(true, lo, hi):
    vmin = min(true.min(), lo.min())
    vmax = max(true.max(), hi.max())
    pad  = 0.04 * (vmax - vmin)
    return (vmin - pad, vmax + pad)


def draw_predictions(ax, med, lo, hi, true, label, ylabel=True):
    """Scatter of posterior median vs. true with 68% error bars."""
    err_lo = med - lo
    err_hi = hi - med
    ax.errorbar(
        true, med,
        yerr=[err_lo, err_hi],
        fmt="none",
        elinewidth=0.75,
        ecolor=C_QUIJOTE,
        alpha=0.3,
        zorder=1,
        rasterized=True,
    )
    ax.scatter(
        true, med,
        s=5, color=C_QUIJOTE, alpha=0.7,
        linewidths=0, zorder=2, rasterized=True,
    )
    prange = _param_axis_range(true, lo, hi)
    ax.plot(prange, prange, color="0.2", lw=0.9, ls="--", zorder=3)
    ax.set_xlim(prange)
    ax.set_ylim(prange)
    ax.set_xlabel("True")
    if ylabel:
        ax.set_ylabel("Predicted")
    ax.set_title(f"{label}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))


def draw_coverage(ax, series, label, show_legend=False, n_boot=1000, band_seed=42,
                  ylabel=True, xlabel=True):
    """CDF of marginal posterior ranks vs. expected uniform CDF.

    Replicates the ili PosteriorCoverage._plot_coverage logic, with a
    bootstrap uncertainty band drawn around each curve (data-driven, per
    bootstrap_coverage_band -- not a null-hypothesis reference band).

    Parameters
    ----------
    series : list of (ranks, name, color)
             ranks : (n_obs,) array for this curve
             name  : legend label, or None to omit from the legend
             color : line color
             A legend is only drawn when more than one named series is given.
    label  : panel title (e.g. a parameter name)
    show_legend : if False, never draw the legend even with multiple named series.
    """
    ndata = len(series[0][0])
    cdf   = np.linspace(0, 1, ndata)

    ax.plot(cdf, cdf, color="0.2", lw=0.9, ls="--", zorder=2)

    for si, (ranks, name, color) in enumerate(series):
        xr = np.sort(ranks)
        xr = xr / xr[-1]   # normalise to [0, 1]
        series_cdf = np.linspace(0, 1, len(ranks))   # own length, may differ across series
        bands = bootstrap_coverage_band(ranks, n_boot=n_boot, seed=band_seed + si)
        ax.fill_betweenx(series_cdf, bands[0], bands[3], color=color, alpha=0.15,
                         zorder=1, edgecolor="none", linewidth=0)
        ax.fill_betweenx(series_cdf, bands[1], bands[2], color=color, alpha=0.3,
                         zorder=1, edgecolor="none", linewidth=0)
        ax.plot(xr, series_cdf, color=color, lw=1.2, zorder=3, label=name)

    # (the Quijote / Emulator legend is drawn at figure level, outside the grid)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if xlabel:
        ax.set_xlabel("Predicted percentile")
    if ylabel:
        ax.set_ylabel("Empirical percentile")
    else:
        ax.tick_params(labelleft=False)
    ax.set_title(f"{label}")
    _unit_ticks(ax)


def draw_tarp(ax, samples, theta_true, references="random", metric="euclidean",
              bootstrap=True, norm=True, num_alpha_bins=None, num_bootstrap=100):
    """TARP credibility plot (joint coverage across all parameters).

    Replicates ili.validation.metrics.PosteriorCoverage._plot_TARP.

    Parameters
    ----------
    samples    : (n_post, n_obs, n_params)
    theta_true : (n_obs, n_params)
    """
    import tarp
    ecp, alpha = tarp.get_tarp_coverage(
        samples, theta_true,
        references=references, metric=metric,
        norm=norm, bootstrap=bootstrap,
        num_alpha_bins=num_alpha_bins,
        num_bootstrap=num_bootstrap,
    )

    ax.plot([0, 1], [0, 1], color="0.2", lw=0.9, ls="--", zorder=2)
    if bootstrap:
        ecp_mean = np.mean(ecp, axis=0)
        ecp_std  = np.std(ecp, axis=0)
        ax.plot(alpha, ecp_mean, color=C_QUIJOTE, lw=1.2, zorder=3, label="TARP")
        ax.fill_between(alpha, ecp_mean - ecp_std, ecp_mean + ecp_std,
                         alpha=0.3, color=C_QUIJOTE, zorder=1,
                         edgecolor="none", linewidth=0)
        ax.fill_between(alpha, ecp_mean - 2 * ecp_std, ecp_mean + 2 * ecp_std,
                         alpha=0.2, color=C_QUIJOTE, zorder=1,
                         edgecolor="none", linewidth=0)
    else:
        ax.plot(alpha, ecp, color=C_QUIJOTE, lw=1.2, zorder=3, label="TARP")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Credibility level")
    ax.set_ylabel("Expected coverage")
    ax.set_title("TARP")
    _unit_ticks(ax)


def _unit_ticks(ax):
    """Thinned ticks for [0, 1] x [0, 1] panels (0, 0.5, 1; minor every 0.25)."""
    fmt = mticker.FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(mticker.MultipleLocator(0.5))
        axis.set_minor_locator(mticker.MultipleLocator(0.25))
        axis.set_major_formatter(fmt)


# ── Main figure builder ───────────────────────────────────────────────────────

def make_figure(datasets, outpath, include_tarp=True, tarp_kwargs=None, row_labels=None):
    """Build and save the MNRAS validation figure (4 panels, or 5 with TARP).

    One row is drawn per entry in `datasets`, sharing the same panel layout.

    Parameters
    ----------
    datasets     : list of (samples, theta_true, mock) triples
                   samples    : (n_post, n_obs, n_params)
                   theta_true : (n_obs, n_params)
                   mock       : optional (mock_samples, mock_theta_true) pair, overlaid
                                as an "Emulator" coverage curve alongside the row's
                                "Quijote" curve; None for an ordinary single-curve row
    outpath      : output file (.pdf or .png)
    include_tarp : if True, append a 5th panel with the TARP coverage plot
    tarp_kwargs  : optional dict of kwargs forwarded to draw_tarp
    row_labels   : optional list of strings, one per row, shown left of each row
    """
    n_rows   = len(datasets)
    n_panels = 5 if include_tarp else 4

    # Built at printed size (7.0 in, included at scale 1) on a FIXED grid:
    # every panel has identical width and height. Row descriptions are
    # horizontal row titles; the legend sits outside the panels.
    gaps = [GAP_LEFT, GAP_TICKS, GAP_YLABEL, GAP_SHARED, GAP_YLABEL][:n_panels]
    P = (FIG_W_IN - sum(gaps) - GAP_RIGHT) / n_panels          # panel side, in
    row_h = ROW_TITLE_H + PANEL_TTL_H + P + XLABEL_H
    fig_h = TOP_PAD + n_rows * row_h
    fig = plt.figure(figsize=(FIG_W_IN, fig_h))
    lefts = np.cumsum(gaps) + P * np.arange(n_panels)          # panel left edges, in

    axes, row_tops = [], []
    for i in range(n_rows):
        top = fig_h - TOP_PAD - i * row_h                       # top of row block, in
        bottom = top - ROW_TITLE_H - PANEL_TTL_H - P            # panel bottom, in
        row_tops.append(top)
        row_axes = [fig.add_axes([x / FIG_W_IN, bottom / fig_h, P / FIG_W_IN, P / fig_h])
                    for x in lefts]
        row_axes[3].sharey(row_axes[2])
        axes.append(row_axes)

    letter = 0
    for i, (samples, theta_true, mock) in enumerate(datasets):
        med, lo, hi = compute_predictions(samples)
        ranks = compute_ranks(samples, theta_true)   # (n_obs, n_params)
        row = axes[i]

        for j, ax in enumerate(row[:2]):
            draw_predictions(ax, med[:, j], lo[:, j], hi[:, j], theta_true[:, j],
                             PARAM_LABELS[j], ylabel=(j == 0))

        if mock is not None:
            mock_ranks = compute_ranks(*mock)   # (n_obs, n_params)

        for j, ax in enumerate(row[2:4]):
            series = [(ranks[:, j], "Quijote" if mock is not None else None, C_QUIJOTE)]
            if mock is not None:
                series.append((mock_ranks[:, j], "Emulator", C_EMULATOR))
            draw_coverage(ax, series, PARAM_LABELS[j], ylabel=(j == 0),
                          xlabel=(j == 0))

        if include_tarp:
            draw_tarp(row[4], samples, theta_true, **(tarp_kwargs or {}))

        for ax in row:
            ax.text(0.05, 0.95, f"({chr(97 + letter)})",
                    transform=ax.transAxes,
                    fontsize=8, va="top", ha="left",
                    fontweight="bold")
            letter += 1

        # Align every x label at the same depth below its axes; the coverage
        # pair shares one x label centred under both panels.
        ylab = -XLABEL_Y_IN / P
        for ax in row:
            ax.xaxis.set_label_coords(0.5, ylab)
        row[2].xaxis.set_label_coords(1 + GAP_SHARED / (2 * P), ylab)

        if row_labels is not None:
            fig.text(0.05 / FIG_W_IN, (row_tops[i] - 0.15) / fig_h, row_labels[i],
                     ha="left", va="baseline", fontsize=9)

    # Quijote / Emulator legend, outside the panels: right end of the first
    # row-title line, above the coverage panels.
    if any(m is not None for _, _, m in datasets):
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color=C_QUIJOTE, lw=1.2, label="Quijote"),
                   Line2D([], [], color=C_EMULATOR, lw=1.2, label="Emulator")]
        x_right = (lefts[3] + P) / FIG_W_IN
        fig.legend(handles=handles, loc="lower right", ncol=2, frameon=False,
                   bbox_to_anchor=(x_right, (row_tops[0] - ROW_TITLE_H) / fig_h),
                   fontsize=8, handlelength=1.5, columnspacing=1.2,
                   borderaxespad=0.0, borderpad=0.1)

    sizes = {(round(ax.get_position().width * FIG_W_IN, 4),
              round(ax.get_position().height * fig_h, 4))
             for row in axes for ax in row}
    for i, row in enumerate(axes):
        for k, ax in enumerate(row):
            pos = ax.get_position()
            print(f"[panel] row {i} col {k}: w = {pos.width * FIG_W_IN:.4f} in, "
                  f"h = {pos.height * fig_h:.4f} in")
    print(f"[panel] distinct (w, h) sizes: {sorted(sizes)}  -> "
          f"{'ALL IDENTICAL' if len(sizes) == 1 else 'MISMATCH'}")

    for ext_path in ({outpath, os.path.splitext(outpath)[0] + ".png"}
                     if outpath.endswith(".pdf") else {outpath}):
        fig.savefig(ext_path, dpi=300)   # no bbox_inches="tight": keep 7.0 in
    plt.close(fig)
    print(f"Saved: {outpath}")


# ── Loading ────────────────────────────────────────────────────────────────────

def load_dataset(model_dir, seed=42, n_samples_cap=None):
    """Load a (samples, theta_true, mock) triple from a run dir, using its
    top-level validation_quijote and validation_mock directories."""
    vdir = join(model_dir, "validation_quijote")
    samples_path = join(vdir, "posterior_samples.npy")
    theta_path   = join(vdir, "theta_test.npy")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"posterior_samples.npy not found in {vdir}")
    if not os.path.exists(theta_path):
        raise FileNotFoundError(f"theta_test.npy not found in {vdir}")

    samples    = np.load(samples_path)     # (n_post, n_obs, n_params)
    theta_true = np.load(theta_path)       # (n_obs, n_params)
    if n_samples_cap is not None and n_samples_cap < theta_true.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(theta_true.shape[0], size=n_samples_cap, replace=False)
        samples    = samples[:, idx, :]
        theta_true = theta_true[idx, :]
    print(f"[{vdir}] samples: {samples.shape}, theta_true: {theta_true.shape}")

    mock = None
    mock_dir = join(dirname(vdir), "validation_mock")
    mock_samples_path = join(mock_dir, "posterior_samples.npy")
    mock_theta_path   = join(mock_dir, "theta_test.npy")
    if os.path.exists(mock_samples_path) and os.path.exists(mock_theta_path):
        mock_samples = np.load(mock_samples_path)
        mock_theta   = np.load(mock_theta_path)
        if n_samples_cap is not None and n_samples_cap < mock_theta.shape[0]:
            rng = np.random.default_rng(seed)
            mock_idx = rng.choice(mock_theta.shape[0], size=n_samples_cap, replace=False)
            mock_samples = mock_samples[:, mock_idx, :]
            mock_theta   = mock_theta[mock_idx, :]
        print(f"[{vdir}] auto-detected mock overlay at {mock_dir}: "
              f"samples {mock_samples.shape}, theta_true {mock_theta.shape}")
        mock = (mock_samples, mock_theta)

    return samples, theta_true, mock


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MNRAS validation figure: full-range Pk+Qk (row 1) vs "
                    "box-trained Pk+Qk+WST (row 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--row2", choices=sorted(ROW2_CHOICES), default="box_hodfix",
                        help="Which Pk+Qk+WST run goes in row 2.")
    parser.add_argument("--out", default=None,
                        help="Output path (.pdf; a .png twin is also saved). "
                             f"Default: {SAVE_DIR}/<file name set by --row2>")
    parser.add_argument("--tarp", dest="tarp", action="store_true", default=True,
                        help="Include the TARP panel (default: on).")
    parser.add_argument("--no-tarp", dest="tarp", action="store_false",
                        help="Omit the TARP panel.")
    parser.add_argument("--n_samples_cap", type=int, default=None,
                        help="Subsample each validation set to at most this many "
                             "test points. Default: no cap (use all available "
                             "test points).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    row_labels = []
    datasets = []
    key2, label2, dir2, default_name = ROW2_CHOICES[args.row2]
    for key, label, model_dir in (ROW1, (key2, label2, dir2)):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"{key}: model dir not found: {model_dir}")
        print(f"── {key}  {model_dir}")
        datasets.append(load_dataset(model_dir, seed=args.seed,
                                     n_samples_cap=args.n_samples_cap))
        row_labels.append(label)

    out_path = args.out or join(SAVE_DIR, default_name)

    make_figure(datasets, out_path, include_tarp=args.tarp, row_labels=row_labels)
