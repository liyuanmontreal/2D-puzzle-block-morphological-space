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

# ----- figure -----
fig, ax = plt.subplots(figsize=(10,10))
ax.set_title("Puzzle Morphospace (Hover Popup)", fontsize=16)

# ----- invisible scatter (for hover detection only) -----
scatter = ax.scatter(coords[:,0], coords[:,1], s=20, alpha=0)

# ----- draw small thumbnails -----
THUMB_ZOOM = 2.0
for (x, y), board in zip(coords, boards):
    img = OffsetImage(board, cmap='gray_r', zoom=THUMB_ZOOM)
    ab = AnnotationBbox(
        img, (x, y),
        frameon=True,
        pad=0.2,
        bboxprops=dict(edgecolor='black', linewidth=0.4)
    )
    ax.add_artist(ab)

ax.set_xticks([]); ax.set_yticks([])

# ----- Popup state -----
popup = None

# ----- hover: show manual popup -----
popup = None   # global

cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_hover(sel):
    global popup

    # 全部禁用默认annotation
    sel.annotation.set_visible(False)

    idx = sel.index
    board = boards[idx]

    # 如果有旧 popup，就删除
    if popup is not None:
        try:
            popup.remove()
        except:
            pass
        popup = None

    big_img = OffsetImage(board, cmap='gray_r', zoom=8.0)

    popup = AnnotationBbox(
        big_img,
        (coords[idx,0] + 40, coords[idx,1] + 40),
        frameon=True,
        pad=0.3,
        bboxprops=dict(edgecolor='black', linewidth=1.5)
    )
    ax.add_artist(popup)

    plt.draw()



@cursor.connect("remove")
def on_unhover(sel):
    global popup
    if popup is not None:
        popup.remove()
        popup = None
        plt.draw()


plt.savefig("tsne_thumb.svg", dpi=50)
plt.savefig("tsne_thumb.png", dpi=50)
plt.show()
