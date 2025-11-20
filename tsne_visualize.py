import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = np.loadtxt("puzzle_morphospace_v01.csv", delimiter=",", skiprows=1)
print("数据形状:", data.shape)

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    max_iter=2000,     # 用这个取代 n_iter
    init="pca",
    random_state=42,
)

coords = tsne.fit_transform(data)

# 拆出几个特征（和之前 header 对应）
black_ratio       = data[:, 0]
mirror_lr         = data[:, 1]
n_components      = data[:, 4]
boundary_complexity = data[:, 5]

plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.scatter(coords[:, 0], coords[:, 1], c=black_ratio, s=10, cmap="viridis")
plt.title("colored by black_ratio ")
plt.colorbar()

plt.subplot(2, 2, 2)
plt.scatter(coords[:, 0], coords[:, 1], c=mirror_lr, s=10, cmap="viridis")
plt.title("colored by mirror_lr(symmetrical) ")
plt.colorbar()

plt.subplot(2, 2, 3)
plt.scatter(coords[:, 0], coords[:, 1], c=n_components, s=10, cmap="viridis")
plt.title("Colored by the number of connected components")
plt.colorbar()

plt.subplot(2, 2, 4)
plt.scatter(coords[:, 0], coords[:, 1], c=boundary_complexity, s=10, cmap="viridis")
plt.title("Colored by boundary complexity")
plt.colorbar()

plt.tight_layout()
plt.savefig("tsne_result.png", dpi=300, bbox_inches="tight")
plt.savefig("tsne_result.svg", bbox_inches="tight")
plt.show()
