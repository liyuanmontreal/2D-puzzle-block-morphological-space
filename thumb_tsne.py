import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from scipy.spatial import cKDTree

# ---------------------------------------------------
# Load data
# ---------------------------------------------------
data = np.loadtxt("puzzle_morphospace_v01.csv", delimiter=",", skiprows=1)

def int_to_board(x):
    bits = np.array([(x >> i) & 1 for i in range(9)], dtype=int)
    return bits.reshape((3,3))

boards = [int_to_board(i) for i in range(512)]

def board_to_rgb(board):
    """convert 0/1 board into uint8 RGB for matplotlib"""
    return np.stack([board * 255] * 3, axis=-1).astype(np.uint8)

boards_rgb = [board_to_rgb(b) for b in boards]

# ---------------------------------------------------
# t-SNE
# ---------------------------------------------------
coords = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    max_iter=2000,
    init='pca',
    random_state=42
).fit_transform(data)

# KDTree for fast hover checking
tree = cKDTree(coords)

# ---------------------------------------------------
# Plot base figure
# ---------------------------------------------------
fig, ax = plt.subplots(figsize=(10,10))
ax.set_title("Puzzle Morphospace", fontsize=16)

ax.set_xticks([]); ax.set_yticks([])

# invisible scatter for hit test only
scatter = ax.scatter(coords[:,0], coords[:,1], s=20, alpha=0)

# ---------------------------------------------------
# Draw small thumbnails
# ---------------------------------------------------
THUMB_ZOOM = 2.0

for (x, y), img in zip(coords, boards_rgb):
    small = OffsetImage(img, zoom=THUMB_ZOOM)
    ab = AnnotationBbox(
        small, (x, y),
        frameon=True,
        pad=0.25,
        bboxprops=dict(edgecolor='black', linewidth=0.4)
    )
    ax.add_artist(ab)

# ---------------------------------------------------
# Hover: show big popup
# ---------------------------------------------------
popup = None
HOVER_DIST = 10
BIG_ZOOM = 7.0

def on_move(event):
    global popup

    if not event.inaxes:
        # remove popup when mouse leaves axes
        if popup is not None:
            popup.remove()
            popup = None
            fig.canvas.draw_idle()
        return

    # nearest point
    mx, my = event.xdata, event.ydata
    dist, idx = tree.query([mx, my])

    # too far: hide popup
    if dist > HOVER_DIST:
        if popup is not None:
            popup.remove()
            popup = None
            fig.canvas.draw_idle()
        return

    # close to a point → show popup
    if popup is not None:
        try:
            popup.remove()
        except:
            pass
        popup = None

    big_img = OffsetImage(boards_rgb[idx], zoom=BIG_ZOOM)

    popup = AnnotationBbox(
        big_img,
        (coords[idx, 0] + 40, coords[idx, 1] + 40),
        frameon=True,
        pad=0.4,
        bboxprops=dict(edgecolor='black', linewidth=2)
    )
    ax.add_artist(popup)
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_move)

plt.tight_layout()


plt.savefig("tsne_thumb.png", dpi=50)
plt.savefig("tsne_thumb.svg", dpi=50)
plt.show()

