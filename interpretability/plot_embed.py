import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib import font_manager

# Nimbus Roman (URW Times clone) registered explicitly so the PDF embeds a Times-like font
for _f in glob.glob('/usr/share/fonts/urw-base35/NimbusRoman-*.otf'):
    font_manager.fontManager.addfont(_f)

# MNRAS journal style: serif/Times + STIX math fonts
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
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

data = np.load(f'/work/hdd/bdne/yzhang116/cp_visualize/sim66_embeddings_logits.npz')

# positional embeddings from ViT
pos_emb = data['pos_embed'][0]          # (64, n_embd)
pos_emb_norm = pos_emb / np.linalg.norm(pos_emb, axis=1, keepdims=True)
ref_index = 0
ref_norm = pos_emb_norm[ref_index]      # (D,)
similarity = pos_emb_norm @ ref_norm
similarity = similarity[:64].reshape((4, 4, 4))

# MNRAS single-column width ~3.4 in
# final PDF is exactly 3.4 in wide (saved without bbox_inches='tight')
fig, axs = plt.subplots(2, 2, figsize=(3.4, 3.1))
fig.subplots_adjust(left=0.10, right=0.78, bottom=0.11, top=0.93, wspace=0.28, hspace=0.42)

for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        im = axs[i, j].imshow(similarity[idx][:4, :4], cmap='Blues', vmin=0, vmax=1)
        axs[i, j].set_title(f'$x={idx}$')
        axs[i, j].set_xticks(range(4))
        axs[i, j].set_yticks(range(4))
        axs[i, j].set_xticklabels(range(4))
        axs[i, j].set_yticklabels(range(4))
        if i == 1:
            axs[i, j].set_xlabel('$z$')
        if j == 0:
            axs[i, j].set_ylabel('$y$')

# colour bar spans exactly the height of the (aspect-locked) image panels
for a in axs.flat:
    a.apply_aspect()
_top = axs[0, 1].get_position().y1
_bot = axs[1, 1].get_position().y0
_x1 = axs[0, 1].get_position().x1
cbar_ax = fig.add_axes([_x1 + 0.04, _bot, 0.035, _top - _bot])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Cosine similarity')
cbar.outline.set_linewidth(0.75)
cbar.ax.tick_params(labelsize=8, width=0.75)

plt.savefig('pos_embedding.pdf', dpi=300)
plt.savefig('pos_embedding.png', dpi=300)
plt.close()
print("Saved pos_embedding.pdf / .png")
