import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import mplcursors

# ----- load data -----
data = np.loadtxt("puzzle_morphospace_v01.csv", delimiter=",", skiprows=1)

def int_to_board(x):
    bits = np.array([(x >> i) & 1 for i in range(9)], dtype=int)
    return bits.reshape((3,3))

boards = [int_to_board(i) for i in range(512)]

# ----- t-SNE -----
coords = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    max_iter=2000,
    init='pca',
    random_state=42
).fit_transform(data)

# ---------- collision-free layout ----------
def collision_layout(points, min_dist=10.0, steps=200):
    pts = points.copy()
    for _ in range(steps):
        disp = np.zeros_like(pts)
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                delta = pts[j] - pts[i]
                dist = np.linalg.norm(delta)
                if dist < 1e-6:
                    delta = np.random.randn(*delta.shape)
                    dist = np.linalg.norm(delta)
                overlap = min_dist - dist
                if overlap > 0:
                    push = (overlap / dist) * 0.5
                    disp[i] -= push * delta
                    disp[j] += push * delta
        pts += disp
    return pts

coords2 = collision_layout(coords, min_dist=10.0, steps=150)

# ----- figure -----
fig, ax = plt.subplots(figsize=(10,10))
ax.set_title("Puzzle Morphospace (Padding + Collision-Free + Hover Zoom)", fontsize=16)

# ----- fit axis -----
ax.set_xlim(coords2[:,0].min() - 50, coords2[:,0].max() + 50)
ax.set_ylim(coords2[:,1].min() - 50, coords2[:,1].max() + 50)

# ----- thumbnails -----
THUMB_ZOOM = 2.0
SMALL_THUMBS = []
for (x, y), board in zip(coords2, boards):
    img = OffsetImage(board, cmap='gray_r', zoom=THUMB_ZOOM)
    ab = AnnotationBbox(
        img, (x, y),
        frameon=True,
        pad=0.2,  # pretty padding
        bboxprops=dict(edgecolor='black', linewidth=0.4)
    )
    art = ax.add_artist(ab)
    SMALL_THUMBS.append((art, board))

ax.set_xticks([]); ax.set_yticks([])

# ---- hover: show large thumbnail ----
cursor = mplcursors.cursor(highlight=False)

@cursor.connect("add")
def on_hover(sel):
    idx = sel.index
    artist, board = SMALL_THUMBS[idx]

    big_img = OffsetImage(board, cmap='gray_r', zoom=7.0)

    sel.annotation.set(text="")
    sel.annotation.set_offset((40, 40))

    sel.annotation.box_patch.set(fc="white", ec="black", lw=2)
    sel.annotation.set_image(big_img.get_array())
plt.savefig("tsne_thumb.svg", dpi=50)
plt.savefig("tsne_thumb.png", dpi=50)
plt.show()
