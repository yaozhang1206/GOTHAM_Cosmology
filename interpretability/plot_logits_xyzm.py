"""Next-token probabilities for x, y, z and mass of one sample, stacked in a
single compact panel (4 rows, 3 halos per row).

Usage: python plot_logits_xyzm.py [idx]   (default idx = 62)
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import font_manager

# Nimbus Roman (URW Times clone) registered explicitly so the PDF embeds a Times-like font
for _f in glob.glob('/usr/share/fonts/urw-base35/NimbusRoman-*.otf'):
    font_manager.fontManager.addfont(_f)
import matplotlib.gridspec as gridspec
import sys

# MNRAS journal style (matches plot_logits.py)
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
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

NHALO = 3
NPROP = 8
# properties to show: (index into the 8 per-halo tokens, row label)
PROPS = [(0, r'$x$'), (1, r'$y$'), (2, r'$z$'), (3, 'mass')]
COLORS = ['tab:blue', 'tab:orange', 'tab:green']   # original colours (author's choice)
LW = 1.0


def make_logits_figure_xyzm(idx, logits, truth_sentences, outdir='.'):
    """Save a 4-row (x, y, z, mass) logits figure for one sample index."""
    logits_sel = logits[idx]  # (L-1, vocab_size)
    truth_sel = truth_sentences[idx]  # (L,)
    prop = np.exp(logits_sel) / np.exp(logits_sel).sum(axis=-1, keepdims=True)  # (L-1, vocab_size)
    prop = prop[6:6 + NHALO * NPROP].reshape((NHALO, NPROP, -1))  # (nhalo, nprop, vocab_size)
    truth_sel = truth_sel[7:7 + NHALO * NPROP].reshape((NHALO, NPROP))  # (nhalo, nprop)

    # MNRAS single-column width
    # final PDF is exactly 3.4 in wide (saved without bbox_inches='tight')
    fig = plt.figure(figsize=(3.4, 3.6))
    gs = gridspec.GridSpec(len(PROPS), 1, figure=fig, hspace=0,
                           left=0.155, right=0.97, top=0.885, bottom=0.105)

    axes = []
    for k, (i, label) in enumerate(PROPS):
        # x, y, z share the 0-41 token range; mass keeps its own x-axis
        share = axes[0] if (axes and i in (0, 1, 2)) else None
        ax = fig.add_subplot(gs[k], sharex=share)
        axes.append(ax)
        for h in range(NHALO):
            ax.plot(prop[h, i, :], color=COLORS[h], linewidth=LW,
                    label=f"Halo {h + 1}" if k == 0 else None)
        ymin, ymax = ax.get_ylim()
        for h in range(NHALO):
            ax.vlines(x=truth_sel[h, i], ymin=ymin, ymax=prop[:, i, :].max(),
                      color=COLORS[h], linewidth=LW)
        ax.set_ylim(ymin, ymax)
        if i in (0, 1, 2):
            ax.set_xlim(0, 41)
        ax.text(0.03, 0.92, label, transform=ax.transAxes,
                ha='left', va='top', fontsize=9)

    # x/y/z tick labels go on top of the panel, mass tick labels at the bottom,
    # so both token ranges are readable without splitting the panel
    axes[0].tick_params(axis='x', labeltop=True, labelbottom=False)
    axes[0].xaxis.set_label_position('top')
    axes[0].set_xlabel(r"Token index ($x$, $y$, $z$)", labelpad=2)
    for ax in axes[1:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel("Token index (mass)", labelpad=1)

    # single y label centred on the panel (between rows 2 and 3)
    axes[1].set_ylabel("Probability")
    axes[1].yaxis.set_label_coords(-0.13, 0.0)

    # legend in the mass panel, whose right half is empty
    axes[-1].legend(*axes[0].get_legend_handles_labels(), loc='upper right', framealpha=0., edgecolor='0.8',
                   handlelength=1.5, borderaxespad=0.3, labelspacing=0.25)

    fig.savefig(f"{outdir}/logits_halo_{idx}_xyzm.pdf", dpi=300)
    fig.savefig(f"{outdir}/logits_halo_{idx}_xyzm.png", dpi=300)
    plt.close(fig)
    print(f"Saved logits_halo_{idx}_xyzm.pdf / .png")


if __name__ == "__main__":
    data = np.load('/work/hdd/bdne/yzhang116/cp_visualize/during_gen/sim66_embeddings_logits_blocks.npz')
    truth = np.load('/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits_blocks.npz')
    logits = data['logits_100']
    truth_sentences = truth['sentence']

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 62
    make_logits_figure_xyzm(idx, logits, truth_sentences)
