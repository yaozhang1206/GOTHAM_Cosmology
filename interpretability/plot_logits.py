import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import font_manager

# Nimbus Roman (URW Times clone) registered explicitly so the PDF embeds a Times-like font
for _f in glob.glob('/usr/share/fonts/urw-base35/NimbusRoman-*.otf'):
    font_manager.fontManager.addfont(_f)
import matplotlib.gridspec as gridspec
import sys
from matplotlib.ticker import FixedLocator, MaxNLocator

# MNRAS journal style (matches plot_wte_similarity.py)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.75,
    'xtick.major.width': 0.75,
    'xtick.major.size': 3,
    'ytick.major.width': 0.75,
    'ytick.major.size': 3,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

NHALO = 3
NPROP = 8
LABELS = [r'$x$', r'$y$', r'$z$', 'mass', r'$v_x$', r'$v_y$', r'$v_z$', 'concentration']
COLORS = ['tab:blue', 'tab:orange', 'tab:green']   # original colours (author's choice)
LW = 1.0


def make_logits_figure(indices, logits, sentences, truth_sentences, outdir='.'):
    """Save a stacked logits figure for the 3 given sample indices."""
    prop_list = []
    sent_list = []
    truth_list = []
    for idx in indices:
        logits_sel = logits[idx]  # (L-1, vocab_size)
        sentences_sel = sentences[idx]  # (L,)
        truth_sel = truth_sentences[idx]  # (L,)
        prop = np.exp(logits_sel) / np.exp(logits_sel).sum(axis=-1, keepdims=True)  # (L-1, vocab_size)
        prop = prop[6:6 + NHALO * NPROP].reshape((NHALO, NPROP, -1))  # (nhalo, nprop, vocab_size)
        sentences_sel = sentences_sel[7:7 + NHALO * NPROP].reshape((NHALO, NPROP))  # (nhalo, nprop)
        truth_sel = truth_sel[7:7 + NHALO * NPROP].reshape((NHALO, NPROP))  # (nhalo, nprop)
        prop_list.append(prop)
        sent_list.append(sentences_sel)
        truth_list.append(truth_sel)

    # final PDF is exactly 7.0 in wide (saved without bbox_inches='tight')
    fig = plt.figure(figsize=(7.0, 5.2))
    outer = gridspec.GridSpec(2, 4, figure=fig, wspace=0.32, hspace=0.30,
                              left=0.078, right=0.99, top=0.905, bottom=0.075)

    for i in range(NPROP):
        row_outer, col_outer = divmod(i, 4)
        inner = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer[row_outer, col_outer], hspace=0)
        axes_group = []
        for k in range(3):
            ax = fig.add_subplot(inner[k], sharex=axes_group[0] if axes_group else None)
            axes_group.append(ax)
            for h in range(NHALO):
                label = f"Halo {h + 1}" if (i == 3 and k == 0) else None
                ax.plot(prop_list[k][h, i, :], color=COLORS[h], linewidth=LW, label=label)
            ymin_panel, ymax_panel = ax.get_ylim()
            for h in range(NHALO):
                #ax.vlines(x=sent_list[k][h, i], ymin=ymin_panel, ymax=prop_list[k][:, i, :].max(),
                #          color=COLORS[h], linestyle='dashed', linewidth=0.7)
                ax.vlines(x=truth_list[k][h, i], ymin=ymin_panel, ymax=prop_list[k][:, i, :].max(),
                          color=COLORS[h], linewidth=LW)
            ax.set_ylim(ymin_panel, ymax_panel)
            if k > 0:  # no y ticks in the top 15% so labels cannot collide with the panel above
                for _nb in (3, 4, 6):  # coarsest tick set that leaves >= 2 labelled ticks
                    _t = MaxNLocator(nbins=_nb, steps=[1, 2, 2.5, 5, 10]).tick_values(ymin_panel, ymax_panel)
                    _t = [t for t in _t if ymin_panel <= t <= ymin_panel + 0.85 * (ymax_panel - ymin_panel)]
                    if len(_t) >= 2:
                        break
                ax.yaxis.set_major_locator(FixedLocator(_t))
            if i in (0, 1, 2):
                ax.set_xlim(0, 41)
            if k == 0:
                ax.set_title(LABELS[i])
            if k < 2:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Token index", labelpad=1)
        if col_outer == 0:
            axes_group[1].set_ylabel("Probability")
        if i == 3:  # one shared legend above all panels
            handles, labels = axes_group[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=NHALO,
                       bbox_to_anchor=(0.53, 1.0), frameon=False, fontsize=8,
                       handlelength=2.0, columnspacing=2.0)


    suffix = "_".join(str(idx) for idx in indices)
    fig.savefig(f"{outdir}/logits_halo_{suffix}.pdf", dpi=300)
    fig.savefig(f"{outdir}/logits_halo_{suffix}.png", dpi=300)
    plt.close(fig)
    print(f"Saved logits_halo_{suffix}.pdf / .png")


if __name__ == "__main__":
    data = np.load('/work/hdd/bdne/yzhang116/cp_visualize/during_gen/sim66_embeddings_logits_blocks.npz')
    truth = np.load('/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits_blocks.npz')
    logits = data['logits_100']
    sentences = data['sentence']
    truth_sentences = truth['sentence']

    idx1, idx2, idx3 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    make_logits_figure([idx1, idx2, idx3], logits, sentences, truth_sentences)
