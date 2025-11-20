import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# 1. 准备数据
#    每行对应一个 tracker / 实验组
# -------------------------------------------------
data = {
    "tracker": ["ByteTrack", "OC-SORT", "Hybrid-SORT-ReID", "FineTrack", "StrongSORT++",  "DeconfuseTrack", "SparseTrack", "RBO-Track(ours)"],
    "HOTA":    [63.1, 63.2, 64.0, 64.3, 64.4, 64.9, 65.1, 65.2],
    "IDF1":    [77.3, 77.5, 78.7, 79.5, 79.5, 80.6, 80.1, 81.9],
    "AssA":    [62.0, 63.2, 63.5, 64.5, 64.4, 65.1, 65.1, 66.4],
}

# 如果想让半径“柔和”一点，可以开平方或线性放大
radius = (np.array(data["AssA"])-60)**3 * 20      # 5 是手动缩放因子，可改

# -------------------------------------------------
# 2. 画图
# -------------------------------------------------
plt.figure(figsize=(6, 5))
scatter = plt.scatter(
    data["HOTA"],
    data["IDF1"],
    s=radius,
    alpha=0.85,
    c=range(len(data["tracker"])),  # 给不同颜色区分
    cmap="tab10",
    linewidth=0,
)

# scatter.set_edgecolors(scatter.get_facecolors())
# 给每个点加 tracker 名称
# for x, y, name in zip(data["HOTA"], data["IDF1"], data["tracker"]):
#     plt.text(x, y, name, fontsize=9, ha="center", va="center")

# 坐标轴 & 标题
plt.xlabel("HOTA (%)")
plt.ylabel("IDF1 (%)")
plt.title("HOTA vs IDF1 (point size ∝ AssA)")
# plt.grid(True, ls="--", lw=0.8)

# 如果想让横纵坐标比例一致，可取消注释：
# plt.axis("equal")

plt.xlim(63, 65.5)   # 横轴范围
plt.ylim(77, 83)   # 纵轴范围

plt.tight_layout()
plt.show()