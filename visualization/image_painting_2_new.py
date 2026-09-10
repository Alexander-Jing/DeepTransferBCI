import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ==========================================
# 1. 全局样式配置 (针对 Visio 编辑优化)
# ==========================================
def setup_plot_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 20
    plt.rcParams['mathtext.fontset'] = 'stix'
    # 核心：导出为文字对象而非路径，Visio中可直接双击编辑文字
    plt.rcParams['svg.fonttype'] = 'none'

def generate_combined_tau_plot(save_dir):
    """
    绘制三个数据集随 tau 变化的综合分析图。
    横轴点间距固定，纵轴范围 60-85，间隔 5。
    每个数据集使用不同的点标记、颜色和线型。
    使用用户指定的 plt.grid 参数。
    """
    # --- 数据提取 (来自表格 image_3.png) ---
    # 横轴标签
    tau_labels = ['0.10', '0.50', '1.00', '2.00', '3.00', '4.00']
    # 使用索引作为 X 轴坐标，以实现等间距排列
    x_indexes = np.arange(len(tau_labels))
    
    # 精度数据 (%)
    data_2a = [67.13, 69.05, 68.95, 69.08, 68.94, 68.88]
    data_2b = [79.16, 79.84, 79.85, 80.01, 80.03, 79.80]
    data_shu = [70.94, 72.34, 72.86, 73.01, 73.01, 72.92]

    # 创建画布 (调整为较窄的比例，适合单栏或跨栏排版)
    # 使用 plt.figure 和 add_axes，弃用 tight_layout，
    # 手动控制边距，增加 Visio 导入时的坐标系稳定性。
    fig = plt.figure(figsize=(9, 7))
    # 手动设置坐标轴位置：[左, 下, 宽, 高]，比例范围 0-1
    ax = fig.add_axes([0.15, 0.18, 0.80, 0.77])

    # --- 1. 绘制折线 (设置不同的点、颜色和线型，zorder 确保线在网格上方) ---
    
    # BCI-IV 2a: 蓝色 ('#377eb8'), 实线 ('-'), 圆点标记 ('o')
    ax.plot(x_indexes, data_2a, label='BCI-IV 2a', 
            color='#377eb8', linestyle='-', marker='o', 
            linewidth=2.5, markersize=10, zorder=3)
    
    # BCI-IV 2b: 红色 ('#e41a1c'), 虚线 ('--'), 方块标记 ('s')
    ax.plot(x_indexes, data_2b, label='BCI-IV 2b', 
            color='#e41a1c', linestyle='--', marker='s', 
            linewidth=2.5, markersize=10, zorder=3)
    
    # SHU-3C: 绿色 ('#4daf4a'), 点划线 ('-.'), 三角形标记 ('^')
    ax.plot(x_indexes, data_shu, label='SHU-3C', 
            color='#4daf4a', linestyle='-.', marker='^', 
            linewidth=2.5, markersize=10, zorder=3)

    # --- 2. 坐标轴与刻度配置 ---
    # 纵轴：范围 60-85，间隔 5
    ax.set_ylim(64, 86)
    # 显式设置纵轴刻度位置和标签（显示为整数）
    y_ticks = np.arange(65, 86, 2)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d')) # 显示整数刻度

    # 横轴：使用索引定位，但显示 tau 的数值标签，并留出首尾边距
    ax.set_xlim(-0.4, len(tau_labels) - 0.6)
    ax.set_xticks(x_indexes)
    ax.set_xticklabels(tau_labels)

    # --- 3. 标签与图例 ---
    # 使用 LaTeX 渲染希腊字母 \tau
    ax.set_xlabel(r'Value of parameter $\tau$', labelpad=20, fontsize=24)
    ax.set_ylabel('Accuracy (%)', labelpad=20, fontsize=24)

    # 图例放在右下角 (lower right)，避免遮挡上方的数据点，设置白色背景遮挡网格
    # zorder 设置较高，确保图例遮挡网格线
    legend = ax.legend(loc='upper right', frameon=True, edgecolor='black', 
                       fontsize=18, framealpha=1.0, facecolor='white')
    legend.set_zorder(10)

    # 调整刻度标签字号
    ax.tick_params(axis='both', labelsize=20)

    # --- 4. 绘制网格线 (使用用户指定的参数) ---
    # axis='y': 仅绘制水平网格线
    # zorder=0: 确保网格线在所有折线和标记点下方
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)

    # --- 5. 保存输出 ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    file_base = "Combined_Tau_Sensitivity_Different_Styles"
    
    # 保存 SVG (用于 Visio 编辑)
    plt.savefig(os.path.join(save_dir, f'{file_base}.svg'), format='svg')
    
    # 保存 PNG (用于 快速预览, dpi=300 保证高清，使用 bbox_inches='tight' 去白边)
    plt.savefig(os.path.join(save_dir, f'{file_base}.png'), format='png', dpi=300, bbox_inches='tight')
    
    # 保存 TIF (dpi=300, LZW 压缩)
    plt.savefig(os.path.join(save_dir, f'{file_base}.tif'), format='tiff', dpi=300, bbox_inches='tight',
                pil_kwargs={'compression': 'tiff_lzw'})

    
    plt.close()
    print(f"✅ 已生成综合分析图 (不同线型 & 标记) -> {save_dir}")

# ==========================================
# 2. 执行绘图
# ==========================================
if __name__ == "__main__":
    setup_plot_style()
    target_dir = './visualization/figures'
    generate_combined_tau_plot(target_dir)