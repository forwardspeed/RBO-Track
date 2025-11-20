import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

plt.rcParams.update({
    'font.size': 14,               # 默认 10 → 14
    'axes.titlesize': 18,          # 子图标题
    'axes.labelsize': 18,          # 轴标签
    'xtick.labelsize': 18,         # x 刻度
    'ytick.labelsize': 18,         # y 刻度
    'legend.fontsize': 14,         # 图例
    'figure.titlesize': 18         # 总标题（suptitle）
})

# ---------- 0. 中文与负号 ----------
# 随便挑一个你系统里有的中文 TrueType 字体
zh_font = fm.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc")   # SimSun
rcParams['axes.unicode_minus'] = False   # 正常显示负号

# ---------- 1. 数据 ----------
video_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
                   'MOT17-10', 'MOT17-11', 'MOT17-13']

line_HOTA = [0.919, 0.014, 2.198, -0.435, 2.434, 0.562, 0.004]
line_MOTA = [-0.213, 0.008, 0.268, 0.555, 0.185, 0.623, -0.063]
line_IDF1 = [2.582, 0.023, 4.637, -2.535, 3.167, 1.111, 0.005]
line_ASSA = [1.936, 0.004, 4.213, -0.725, 5.116, 0.971, 0.042]

bar_values_5 = [0.116, 0.055, 0.092, 0.061, 0.210, 0.142, 0.133]
bar_values_7 = [0.412, 0.877, 0.714, 0.847, 0.486, 0.654, 0.683]

x_pos = np.arange(len(video_sequences))

# ---------- 2. 画图 ----------
fig, ax1 = plt.subplots(figsize=(12, 6))          # 1. 画布拉大
# --- 统一刻度字号（想改就改这里） ----------
tick_fontsize = 16          # ← 刻度字号
label_fontsize = 18         # ← 轴标签、视频序列名称字号

# --- 让 Times New Roman 作为默认西文字体 ----------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

ax1.set_xlim(-0.5, len(video_sequences)-0.5)

# 折线
ax1.plot(x_pos, line_HOTA, marker='o', linestyle='-',  label='HOTA')   # 实线
ax1.plot(x_pos, line_MOTA, marker='s', linestyle='--', label='MOTA')   # 虚线
ax1.plot(x_pos, line_IDF1, marker='^', linestyle='-.', label='IDF1')   # 点划线
ax1.plot(x_pos, line_ASSA, marker='d', linestyle=':',  label='AssA')   # 点线
ax1.set_ylim(-3, 6)
ax1.set_ylabel('Lines-Changes in various metrics by FDU',
               fontsize=label_fontsize, family='Times New Roman', labelpad=15)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(video_sequences, rotation=0, ha='center',
                    fontsize=label_fontsize, family='Times New Roman')

# 刻度字号
ax1.tick_params(axis='both', labelsize=tick_fontsize)

ax1.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.7)
ax1.set_axisbelow(True)

# 柱状
ax2 = ax1.twinx()
width = 0.4
ax2.bar(x_pos - width/2, bar_values_5, width=width,
        color='#FF69B4', alpha=0.5, label='0.5-0.75')
ax2.bar(x_pos + width/2, bar_values_7, width=width,
        color='#4682B4', alpha=0.5, label='0.75-1.0')
ax2.set_ylim(0, 0.9)
ax2.set_ylabel('Columns-Interval Detection/Total',
               fontsize=label_fontsize, family='Times New Roman', labelpad=15)
ax2.tick_params(axis='both', labelsize=tick_fontsize)

# ---------- 3. 图例放外边 ----------
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax1.legend(lines1 + lines2, labels1 + labels2,
                 loc='upper right', fontsize=10, ncol=2,
                 columnspacing=0.5, handletextpad=0.4,
                 frameon=True, fancybox=True, framealpha=0.7,
                 prop=zh_font)          # 图例里的中文仍用原来的 zh_font

# ---------- 4. 留白 + 标题 ----------
fig.subplots_adjust(bottom=0.2)
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()