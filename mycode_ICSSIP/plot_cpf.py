import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

def plot_cdf_focused(ax, data_list, colors, lines, labels, xlabel, title=None, rmse_text=None,
                     focus_percentile=95, x_padding=0.05):
    # 计算所有数据集中最小的95%分位数
    min_p95 = min(np.percentile(data, focus_percentile) for data in data_list)

    # 设置x轴范围：0到最小95%分位数 + 留白
    xmax = min_p95 * (1 + x_padding)
    ax.set_xlim(0, xmax)

    # 保持y轴完整显示
    ax.set_ylim(0, 1.0)

    # 绘制CDF曲线
    for data, color, label, line in zip(data_list, colors, labels, lines):
        # 计算CDF
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        # 绘制CDF曲线
        sns.lineplot(x=sorted_data, y=cdf,  linestyle=line, color=color, ax=ax, linewidth=3, legend=False)

    # 设置图表属性
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel('CPF', fontsize=20)
    #ax.legend(fontsize=16)
    # 同时设置X轴和Y轴刻度标签字体大小
    ax.tick_params(axis='both', labelsize=20)
    ax.grid(True, linestyle='--', alpha=0.7)

    if title:
        ax.set_title(f"{title} (Focused on {focus_percentile}% Point)")

    return min_p95  # 返回使用的x轴上限值



# ---------------- 读取数据 ----------------
Cam1_az_diff = np.load('Cam1_az_diff_list.npy', allow_pickle=True)
Cam2_az_diff = np.load('Cam2_az_diff_list.npy', allow_pickle=True)
CamAll_az_diff = np.load('CamAll_az_diff_list.npy', allow_pickle=True)
Echo_az_diff = np.load('Echo_az_diff_list.npy', allow_pickle=True)

# ---------------- 取绝对值 ----------------
Cam1_az_diff_abs = [np.abs(x) for x in Cam1_az_diff]
Cam2_az_diff_abs = [np.abs(x) for x in Cam2_az_diff]
CamAll_az_diff_abs = [np.abs(x) for x in CamAll_az_diff]
Echo_az_diff_abs = [np.abs(x) for x in Echo_az_diff]

all_batch_losses = []
for l in Echo_az_diff_abs:
    all_batch_losses.extend(l)
Echo_az_diff_abs = all_batch_losses[:2299]
# ---------------- 准备绘图数据 ----------------
az_data = [Cam1_az_diff_abs, Cam2_az_diff_abs, CamAll_az_diff_abs]

# ---------------- 绘图配置 ----------------
az_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 四组颜色
line_styles = ['-', '--', '-.']
labels = ['A', 'B', 'C']

current_dir = os.getcwd()

# ---------------- 绘制箱线图 ----------------
fig_az, ax_az = plt.subplots(figsize=(3, 4))
positions = [1, 2, 3]
box_width = 0.6
meanprops = dict(marker='o', markerfacecolor='black', markersize=5)

bplot_az = ax_az.boxplot(
    az_data,
    positions=positions,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    showmeans=True,
    medianprops={'color': 'white', 'linewidth': 1},
    meanprops=meanprops
)

for patch, color in zip(bplot_az['boxes'], az_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_az.tick_params(labelsize=14)
ax_az.set_xticks(positions)
ax_az.set_xticklabels(labels, fontsize=14)
ax_az.set_xlabel('Azimuth', fontsize=16)
ax_az.set_ylabel('Angle Error (rad)', fontsize=16)

# 图例
legend_elements = [Line2D([0], [0], color=c, label=l, linestyle=ls, alpha=0.7)
                   for c, l, ls in zip(az_colors, labels, line_styles)]
ax_az.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'fig_az_boxplot.png'), dpi=900, bbox_inches='tight')
plt.show()

# ---------------- 绘制CDF图 ----------------
fig, ax = plt.subplots(1, 1, figsize=(5, 4))

# 假设你有一个函数 plot_cdf_focused(ax, data, colors, line_styles, labels, xlabel)
# 这里调用
plot_cdf_focused(
    ax,
    az_data,
    az_colors,
    line_styles,
    labels,
    'Azimuth Error (rad)'
)

# 调整布局和图例
plt.subplots_adjust(left=0.4)  # 根据需要调整
plt.tight_layout(rect=[0, 0, 1, 0.96])
ax.legend(handles=legend_elements, loc='lower right', fontsize=18, bbox_to_anchor=(1, 0), frameon=True)

save_path = os.path.join(current_dir, 'fig_az_cpf.png')
plt.savefig(save_path, dpi=900, bbox_inches='tight')
print(f"综合分析图已保存至: {save_path}")
plt.show()


#==========================================================



# ---------------- 读取数据 ----------------
Cam1_el_diff = np.load('Cam1_el_diff_list.npy', allow_pickle=True)
Cam2_el_diff = np.load('Cam2_el_diff_list.npy', allow_pickle=True)
CamAll_el_diff = np.load('CamAll_el_diff_list.npy', allow_pickle=True)
#Echo_el_diff = np.load('Echo_az_diff_list.npy', allow_pickle=True)

# ---------------- 取绝对值 ----------------
Cam1_el_diff_abs = [np.abs(x) for x in Cam1_el_diff]
Cam2_el_diff_abs = [np.abs(x) for x in Cam2_el_diff]
CamAll_el_diff_abs = [np.abs(x) for x in CamAll_el_diff]

# ---------------- 准备绘图数据 ----------------
el_data = [Cam1_el_diff_abs, Cam2_el_diff_abs, CamAll_el_diff_abs]

# ---------------- 绘图配置 ----------------
el_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 四组颜色
line_styles = ['-', '--', '-.']
labels = ['A', 'B', 'C']

current_dir = os.getcwd()

# ---------------- 绘制箱线图 ----------------
fig_az, ax_az = plt.subplots(figsize=(3, 4))
positions = [1, 2, 3]
box_width = 0.6
meanprops = dict(marker='o', markerfacecolor='black', markersize=5)

bplot_az = ax_az.boxplot(
    el_data,
    positions=positions,
    widths=box_width,
    patch_artist=True,
    showfliers=False,
    showmeans=True,
    medianprops={'color': 'white', 'linewidth': 1},
    meanprops=meanprops
)

for patch, color in zip(bplot_az['boxes'], az_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_az.tick_params(labelsize=14)
ax_az.set_xticks(positions)
ax_az.set_xticklabels(labels, fontsize=14)
ax_az.set_xlabel('Elevation', fontsize=16)
ax_az.set_ylabel('Angle Error (rad)', fontsize=16)

# 图例
legend_elements = [Line2D([0], [0], color=c, label=l, linestyle=ls, alpha=0.7)
                   for c, l, ls in zip(az_colors, labels, line_styles)]
ax_az.legend(handles=legend_elements, loc='upper left', fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'fig_el_boxplot.png'), dpi=900, bbox_inches='tight')
plt.show()

# ---------------- 绘制CDF图 ----------------
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

# 假设你有一个函数 plot_cdf_focused(ax, data, colors, line_styles, labels, xlabel)
# 这里调用
plot_cdf_focused(
    ax,
    el_data,
    az_colors,
    line_styles,
    labels,
    'Elevation Error (rad)'
)

# 调整布局和图例
plt.subplots_adjust(left=0.4)  # 根据需要调整
plt.tight_layout(rect=[0, 0, 1, 0.96])
ax.legend(handles=legend_elements, loc='lower right', fontsize=18, bbox_to_anchor=(1, 0), frameon=True)

save_path = os.path.join(current_dir, 'fig_el_cpf.png')
plt.savefig(save_path, dpi=900, bbox_inches='tight')
print(f"综合分析图已保存至: {save_path}")
plt.show()


