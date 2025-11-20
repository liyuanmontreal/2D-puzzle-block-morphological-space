import numpy as np
import csv
from collections import deque

# -----------------------------
# 形态特征函数
# -----------------------------

def black_ratio(board: np.ndarray) -> float:
    return np.sum(board == 1) / board.size

def mirror_similarity(a: np.ndarray, b: np.ndarray) -> float:
    same = np.sum(a == b)
    return same / a.size

def mirror_lr(board: np.ndarray) -> float:
    return mirror_similarity(board, np.fliplr(board))

def mirror_ud(board: np.ndarray) -> float:
    return mirror_similarity(board, np.flipud(board))

def rot_180(board: np.ndarray) -> float:
    return mirror_similarity(board, np.rot90(board, 2))

def count_components(board: np.ndarray) -> int:
    h, w = board.shape
    visited = np.zeros_like(board, dtype=bool)
    comps = 0

    def neighbors(r, c):
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < h and 0 <= nc < w:
                yield nr, nc

    for r in range(h):
        for c in range(w):
            if board[r, c] == 1 and not visited[r, c]:
                comps += 1
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in neighbors(cr, cc):
                        if board[nr, nc] == 1 and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
    return comps

def boundary_complexity(board: np.ndarray) -> int:
    h, w = board.shape
    b = 0
    # 水平
    for r in range(h):
        for c in range(w-1):
            if board[r, c] != board[r, c+1]:
                b += 1
    # 垂直
    for r in range(h-1):
        for c in range(w):
            if board[r, c] != board[r+1, c]:
                b += 1
    return b


def extract_features(board: np.ndarray) -> np.ndarray:
    return np.array([
        black_ratio(board),
        mirror_lr(board),
        mirror_ud(board),
        rot_180(board),
        float(count_components(board)),
        float(boundary_complexity(board)),
    ], dtype=float)


# -----------------------------
# 枚举所有 3×3 黑白图案
# -----------------------------

def int_to_board(x: int) -> np.ndarray:
    """
    将 0~511 映射到一个 3×3 的黑白图案。
    二进制的 9 位分别映射到 9 个格子。
    """
    bits = np.array([(x >> i) & 1 for i in range(9)], dtype=int)
    return bits.reshape((3,3))

# -----------------------------
# 主程序：枚举 + 输出 CSV
# -----------------------------

all_features = []
all_boards = []

for i in range(512):
    board = int_to_board(i)
    feats = extract_features(board)
    all_boards.append(board)
    all_features.append(feats)

# 保存为 CSV
header = ["black_ratio", "mirror_lr", "mirror_ud", "rot_180", "n_components", "boundary_complexity"]

with open("puzzle_morphospace_v01.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for feats in all_features:
        writer.writerow(feats.tolist())

print("已生成 puzzle_morphospace_v01.csv ，总共 512 条图案记录。")
