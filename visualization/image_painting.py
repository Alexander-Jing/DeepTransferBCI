import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

def generate_bci_plot(dataset_name, x_values, ts_tta, wo_cta, y_range, x_label, save_dir):
    """
    通用绘图函数
    y_range: 元组 (ymin, ymax)，用于控制不同数据集的纵轴视野
    x_label: 横轴标签文字
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # --- 1. 绘制折线 ---
    ax.plot(x_values, ts_tta, marker='o', label='TS-TTA', 
            color='#377eb8', linewidth=2.5, markersize=10, zorder=3)
    ax.plot(x_values, wo_cta, marker='s', label='w/o CTA', 
            color='#e41a1c', linestyle='--', linewidth=2.5, markersize=10, zorder=3)

    # --- 2. 坐标轴与刻度配置 ---
    # 纵轴：动态范围，固定0.5间隔
    ax.set_ylim(y_range[0], y_range[1])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # 横轴：根据数据首尾自动留出0.03的空白边距
    ax.set_xlim(min(x_values) - 0.03, max(x_values) + 0.03)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    # --- 3. 标签与美化 ---
    ax.set_xlabel(x_label, labelpad=20)
    ax.set_ylabel('Accuracy (%)', labelpad=20)

    legend = ax.legend(loc='upper right', frameon=True, edgecolor='black', 
                       fontsize=20, framealpha=1.0, facecolor='white')
    legend.set_zorder(5)

    ax.tick_params(axis='both', labelsize=20)
    # 仅显示加深的水平网格
    ax.grid(True, axis='y', linestyle=':', alpha=0.8, color='#333333', linewidth=1.0, zorder=0)
    #ax.grid(True, axis='y', linestyle='-', alpha=1.0, color='#DDDDDD', linewidth=0.8, zorder=1)

    # --- 4. 保存输出 ---
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    fig.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
    file_path_svg = os.path.join(save_dir, f'plot_{dataset_name}.svg')
    plt.savefig(file_path_svg, format='svg')
    plt.savefig(os.path.join(save_dir, f'plot_{dataset_name}.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Dataset {dataset_name}: SVG and PNG files saved.")

# ==========================================
# 2. 数据定义与执行
# ==========================================
if __name__ == "__main__":
    setup_plot_style()
    target_dir = './visualization/figures'
    
    # 统一的横轴名称（根据你之前的要求）
    common_xlabel = r'Value of base threshold $c_{\tau_0}$'

    # --- 数据集 1: BCI-IV 2a ---
    generate_bci_plot(
        dataset_name = "BCI_IV_2a",
        x_values     = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        ts_tta       = [69.10, 69.21, 68.92, 69.00, 69.09, 69.00, 68.95],
        wo_cta       = [68.15, 68.13, 68.25, 68.27, 68.31, 68.21, 68.02],
        y_range      = (67.30, 69.70), # 2a 范围
        x_label      = common_xlabel,
        save_dir     = target_dir
    )

    # --- 数据集 2: BCI-IV 2b ---
    generate_bci_plot(
        dataset_name = "BCI_IV_2b",
        x_values     = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        ts_tta       = [80.07, 79.92, 79.82, 79.68, 79.85, 79.69, 79.78],
        wo_cta       = [79.04, 79.10, 79.12, 79.22, 79.22, 79.29, 79.20],
        y_range      = (78.30, 80.70), # 2b 精度较高，调整纵轴
        x_label      = common_xlabel,
        save_dir     = target_dir
    )

    # --- 数据集 3: SHU-3C ---
    generate_bci_plot(
        dataset_name = "SHU_3C",
        x_values     = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        ts_tta       = [73.04, 73.02, 73.16, 73.24, 73.11, 73.01, 73.00],
        wo_cta       = [72.55, 72.38, 72.54, 72.49, 72.55, 72.39, 72.31],
        y_range      = (71.30, 73.70), # SHU 范围
        x_label      = common_xlabel,
        save_dir     = target_dir
    )