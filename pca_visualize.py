import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -----------------------------
# 读取 CSV
# -----------------------------

data = np.loadtxt("puzzle_morphospace_v01.csv", delimiter=",", skiprows=1)

# data: shape (512, 6)
print("数据形状:", data.shape)

# -----------------------------
# PCA（二维）
# -----------------------------

pca = PCA(n_components=2)
coords = pca.fit_transform(data)

print("前两主成分累积解释率:", np.sum(pca.explained_variance_ratio_))

# -----------------------------
# 绘图
# -----------------------------

plt.figure(figsize=(8, 8))
plt.scatter(coords[:, 0], coords[:, 1], s=10, color="black")

plt.title("Puzzle Morphospace (PCA 2D)", fontsize=16)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)

plt.show()
