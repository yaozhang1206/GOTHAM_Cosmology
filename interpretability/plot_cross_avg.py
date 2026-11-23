import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import font_manager

# Nimbus Roman (URW Times clone) registered explicitly so the PDF embeds a Times-like font
for _f in glob.glob('/usr/share/fonts/urw-base35/NimbusRoman-*.otf'):
    font_manager.fontManager.addfont(_f)
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, FixedLocator

# MNRAS journal style: serif/Times + STIX math fonts
# (referee polish: figure is built at its printed width, 3.4 in, and every
#  label is >= 7 pt; nothing here changes the plotted data)
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
    'ytick.major.width': 0.75,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.top': False,
    'ytick.right': False,
})

# ============================================================
# Config — edit here
# ============================================================
OUT_DIR        = '/u/yzhang116/NN/figures/visualize/results/'
OUT_NPY_DIR    = '/work/hdd/bdne/yzhang116/cp_visualize/during_gen/'

N_COSMO_TOKENS = 5   # encoder cosmology tokens (columns)
N_HEADER       = 7   # decoder header rows: <START> + 5 cosmo tokens + N_halo
N_HALO_SHOW    = 3   # number of halo blocks to display
chosen_sim = 66
plot_rows      = N_HEADER + N_HALO_SHOW * 8

LABEL_FS       = 8   # token labels (pt)

# --- figure geometry (inches); final PDF is exactly FIG_W wide ---
FIG_W          = 3.4
MARGIN_L       = 0.68    # row labels
MARGIN_R       = 0.66    # colour bar + its labels
MARGIN_B       = 0.42    # cosmo-token bracket + DM-field arrow
MARGIN_T       = 0.06
MAT_H          = 2.39    # same row pitch as the self-attention figure

COSMO_TOKENS  = [r'$\mathrm{\Omega_m}$', r'$\sigma_8$', r'$\mathrm{\Omega_b}$', r'$h$', r'$n_s$']
cosmo_labels = r'$\mathrm{\Omega_m}$' + r',' + r'$\sigma_8$' + r',' + r'$\mathrm{\Omega_b}$' + r',' + r'$h$' + r',' + r'$n_s$'
HEADER_LABELS = ['<START>'] + COSMO_TOKENS + [r'$N_\mathrm{halo}$']

cross_avg = np.load(os.path.join(OUT_NPY_DIR, f'cross_attn_per_layer_head_cosmo_{chosen_sim}_5halos.npy'))  # (n_layers, n_heads, valid_rows, Te)
# keep the header rows plus the first N_HALO_SHOW halo blocks; all
# encoder columns are kept since they are unrelated to the halo count
cross_avg = cross_avg[:, :, :plot_rows, :]
n_layers, n_heads, valid_rows, Te = cross_avg.shape

# mean over both layers and heads
attn_all_avg = cross_avg.mean(axis=(0, 1))   # (valid_rows, Te)

# MNRAS single-column width 3.4 in
MAT_W = FIG_W - MARGIN_L - MARGIN_R
FIG_H = MARGIN_B + MAT_H + MARGIN_T
fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([MARGIN_L / FIG_W, MARGIN_B / FIG_H, MAT_W / FIG_W, MAT_H / FIG_H])
im = ax.imshow(attn_all_avg, cmap='Blues', aspect='auto',
               origin='upper', interpolation='nearest',
               norm=LogNorm(vmin=0.01, vmax=0.1))

ax.axvline(x=N_COSMO_TOKENS - 0.5, color='black', linewidth=1.0, alpha=1.)

# x-axis: no ticks; the cosmological-parameter columns are bracketed and
# labelled below the axis; remaining columns are DM-field tokens, marked
# instead with an arrow
ax.set_xticks([])
trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
pt = 1.0 / 72 / MAT_H             # 1 pt in axes-fraction units (vertical)

# bracket under the 5 cosmology columns, label right-aligned to its end
yb = -3 * pt
ax.plot([-0.5, -0.5, N_COSMO_TOKENS - 0.5, N_COSMO_TOKENS - 0.5],
        [yb + 2 * pt, yb, yb, yb + 2 * pt], transform=trans, color='black',
        linewidth=0.75, clip_on=False)
ax.text(N_COSMO_TOKENS - 0.5, yb - 2 * pt, cosmo_labels, transform=trans,
        ha='right', va='top', fontsize=LABEL_FS, clip_on=False)

y_arrow = -9 * pt
ax.annotate('', xy=(Te - 0.5, y_arrow), xytext=(N_COSMO_TOKENS + 0.5, y_arrow),
            xycoords=trans, textcoords=trans,
            arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
            annotation_clip=False)
ax.text((N_COSMO_TOKENS - 0.5 + Te - 0.5) / 2, y_arrow - 3 * pt,
        'DM fields (density decreasing)',
        transform=trans, ha='center', va='top', fontsize=9, clip_on=False)

# y-axis: header rows get real ticks with labels staggered in two columns
# (adjacent 8-pt labels would otherwise collide); halo rows get one label
# per row group, with dashed / dotted separators
assert (valid_rows - N_HEADER) % 8 == 0, f'unexpected valid_rows={valid_rows}'
n_halo = (valid_rows - N_HEADER) // 8

ax.set_yticks([])

XYZ_COLOR = 'tab:blue'
V_COLOR   = 'tab:blue'
ytrans = ax.get_yaxis_transform()  # x in axes fraction, y in data coords
off = 3.0 / 72 / MAT_W             # 3 pt gap from the axis

# header rows are only ~6 pt apart: stagger their labels in two columns,
# odd rows next to the axis, even rows pushed out with a thin leader line
OUTER = 14.0 / 72 / MAT_W
for j, lab in enumerate(HEADER_LABELS):
    d = OUTER if j % 2 == 0 else off
    ax.text(-d, j, lab, transform=ytrans, ha='right', va='center',
            fontsize=LABEL_FS, clip_on=False)
    if j % 2 == 0:
        ax.plot([0, -(OUTER - 0.5 * off)], [j, j], transform=ytrans,
                color='0.4', linewidth=0.75, clip_on=False)

for i in range(n_halo):
    row0 = N_HEADER + 8 * i

    # per-row-group labels, aligned with their corresponding rows
    ax.text(-off, row0 + 1, f'$x_{{{i + 1}}}, y_{{{i + 1}}}, z_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=LABEL_FS)
    ax.text(-off, row0 + 3, f'$M_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=LABEL_FS)
    ax.text(-off, row0 + 5, f'$v_{{x,{i + 1}}}, v_{{y,{i + 1}}}, v_{{z,{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=LABEL_FS)
    ax.text(-off, row0 + 7, f'$c_{{{i + 1}}}$',
            transform=ytrans, ha='right', va='center', fontsize=LABEL_FS)

    # dashed lines bracketing (x, y, z)
    ax.axhline(y=row0 - 0.5, color=XYZ_COLOR, linestyle='--', linewidth=0.9, zorder=5)

    # dotted lines bracketing (vx, vy, vz)
    ax.axhline(y=row0 + 3.5, color=V_COLOR, linestyle=':', linewidth=1.0, zorder=5)

cax = fig.add_axes([(MARGIN_L + MAT_W + 0.07) / FIG_W, MARGIN_B / FIG_H,
                    0.09 / FIG_W, MAT_H / FIG_H])
cbar = fig.colorbar(im, cax=cax)
cbar.outline.set_linewidth(0.75)
cbar.set_label('Attention weight', fontsize=9, labelpad=3)
cbar.set_ticks([0.01, 0.02, 0.03, 0.05, 0.07, 0.1])
cax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
cax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: ''))
cax.tick_params(which='major', labelsize=8, length=2.5, width=0.75, pad=1.5)
cax.tick_params(which='minor', length=1.5, width=0.5)

save_path = os.path.join(OUT_DIR, f'cross_avg_all_cosmo_{chosen_sim}_{N_HALO_SHOW}halo')
plt.savefig(save_path + '.pdf', dpi=300)
plt.savefig(save_path + '.png', dpi=300)
plt.close(fig)
print(f'Saved: {save_path}.pdf / .png')
