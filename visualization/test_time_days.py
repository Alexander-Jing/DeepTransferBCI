import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tl.utils.utils import str2bool
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.ticker as ticker


def Test_time_visualizationClass(data_path, class_num, trial_num, current_dir, data_name, imgdata_save=False):
    # load the data
    data = pd.read_csv(data_path, header=0)

    # calculate the num of subjects
    n_subjects = len(data.columns) // (class_num+2)  
    all_accuracies = []  # for restoring accuracy values for each segment

    for subject_id in range(n_subjects):
        # obtain the prediction and true labels
        pred_col = subject_id * (class_num+2) + (class_num+2)-2
        true_col = subject_id * (class_num+2) + (class_num+2)-1
        subject_data = data.iloc[:, [pred_col, true_col]].dropna()
        subject_data.columns = ['pred', 'true']
        
        # calculate the accuracy in each segment
        num_segments = len(subject_data) // trial_num
        if subject_id == 9 and data_name == "WBCIC-SHU-3C":
            num_segments = num_segments+1
        for seg in range(num_segments):
            if subject_id == 9 and data_name == "WBCIC-SHU-3C" and seg==0:
                seg_data = subject_data.iloc[0 : 299]
            elif subject_id == 9 and data_name == "WBCIC-SHU-3C" and seg==1:
                seg_data = subject_data.iloc[299 : 599]
            elif subject_id == 9 and data_name == "WBCIC-SHU-3C" and seg==2:
                seg_data = subject_data.iloc[599 : ]
            else:
                seg_data = subject_data.iloc[seg*trial_num : (seg+1)*trial_num]
            
            pred = seg_data['pred'].astype(int)
            true = seg_data['true'].astype(int)

            # calculate accuracy for the segment
            accuracy = (pred == true).mean()
            all_accuracies.append(accuracy)
    
    # reshape the accuracies array (subjects x segments)
    all_accuracies = np.array(all_accuracies).reshape(n_subjects, -1)
    
    # calculate mean and std for each segment across subjects
    stats = {
        'mean': np.mean(all_accuracies, axis=0),
        'std': np.std(all_accuracies, axis=0),
        'segments': all_accuracies.shape[1]  # number of segments
    }

    if imgdata_save:
        # Prepare to save the plot and statistics to the specified directory
        # plot the results
        plt.figure(figsize=(10, 6))
        x = np.arange(stats['segments'])
        
        plt.plot(x, stats['mean'], 
                label='Accuracy', 
                marker='o')
        """plt.fill_between(x,
                        stats['mean'] - stats['std'],
                        stats['mean'] + stats['std'],
                        alpha=0.2)"""
        
        plt.xlabel('Sample Segment Start Index')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Change Over Segments (Window Size={trial_num})')
        plt.legend()
        plt.grid(True)

        output_filename = os.path.join(current_dir, "MI_accuracy_segments_days.png")
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # save as .png
        print(f"{output_filename} saved")

        # Export statistics to CSV
        stats_df = pd.DataFrame({
            'Segment': x,
            'Accuracy_mean': stats['mean'],
            'Accuracy_std': stats['std']
        })
        
        csv_path = os.path.join(current_dir, "MI_accuracy_stats_days.csv")
        stats_df.to_csv(csv_path, index=False)
        print(f"Statistics saved to {csv_path}")

    return stats

def Test_time_visualizationClass_seeds(class_num, trial_num, current_dir, data_name, args):
    # find the .csv files of different seeds
    _pattern = re.compile(r'_seed_\d+_pred\.csv$')
    csv_files = []
    for _file_name in os.listdir(args.log_path):
        if _file_name.endswith('.csv') and _pattern.search(_file_name):
            full_path = os.path.join(args.log_path, _file_name)
            csv_files.append(full_path)

    stats_ensamble = {
        'mean_seeds': [],
        'mean_ensamble': None,
        'std_ensamble': None
    }
    
    # Process each CSV file found
    for data_path in csv_files:
        stats = Test_time_visualizationClass(data_path, class_num, trial_num, current_dir, data_name, imgdata_save=args.data_save)
        stats_ensamble['mean_seeds'].append(stats['mean'])  # Append mean accuracy for each segment from current seed
    
    # Convert to numpy array for vectorized operations
    stats_ensamble['mean_seeds'] = np.array(stats_ensamble['mean_seeds'])
    
    # Calculate mean and std across seeds for each segment
    stats_ensamble['mean_ensamble'] = np.mean(stats_ensamble['mean_seeds'], axis=0)
    stats_ensamble['std_ensamble'] = np.std(stats_ensamble['mean_seeds'], axis=0)
    
    # Prepare to save the plot and statistics to the specified directory
    # plot the results
    plt.figure(figsize=(10, 6))
    x = np.arange(len(stats_ensamble['mean_ensamble']))
    
    plt.plot(x, stats_ensamble['mean_ensamble'], 
            label='Accuracy', 
            marker='o')
    plt.fill_between(x,
                    stats_ensamble['mean_ensamble'] - stats_ensamble['std_ensamble'],
                    stats_ensamble['mean_ensamble'] + stats_ensamble['std_ensamble'],
                    alpha=0.2)
    
    plt.xlabel('Sample Segment Start Index')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy Change Over Segments (Window Size={trial_num})')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.4, 1.0)

    output_filename = os.path.join(current_dir, "MI_accuracy_segments_seeds_days.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')  # save as .png
    print(f"{output_filename} saved")

    # Export statistics to CSV
    stats_df = pd.DataFrame({
        'Segment': x,
        'Accuracy_mean': stats_ensamble['mean_ensamble'],
        'Accuracy_std': stats_ensamble['std_ensamble']
    })
    
    csv_path = os.path.join(current_dir, "MI_accuracy_stats_seeds_days.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"Statistics saved to {csv_path}")


def Test_time_visualizationClass_seeds_multiple_methods(class_num, trial_num, current_dir, data_name, log_paths, args):
    """
    绘制多个方法的每一天精度对比图
    
    参数:
        class_num: 类别数量
        trial_num: 每个segment的样本数
        current_dir: 结果保存目录
        data_name: 数据集名称
        args: 包含log_paths(方法路径列表)和data_save(是否保存数据)的参数对象
    """
    # 确保log_paths是列表
    method_paths = log_paths if isinstance(log_paths, list) else [log_paths]
    
    # 存储所有方法的结果
    all_methods_results = {}
    
    # 处理每个方法
    for method_path in method_paths:
        method_name = os.path.basename(method_path.rstrip('/'))
        print(f"处理方法: {method_name}")
        
        # 查找该方法的seed文件
        _pattern = re.compile(r'_seed_\d+_pred\.csv$')
        csv_files = []
        for _file_name in os.listdir(method_path):
            if _file_name.endswith('.csv') and _pattern.search(_file_name):
                full_path = os.path.join(method_path, _file_name)
                csv_files.append(full_path)
        
        # 初始化存储结构
        stats_ensamble = {
            'mean_seeds': [],
            'mean_ensamble': None,
            'std_ensamble': None
        }
        
        # 处理每个seed文件
        for data_path in csv_files:
            stats = Test_time_visualizationClass(
                data_path, class_num, trial_num, current_dir, data_name, 
                imgdata_save=False  # 不保存中间结果
            )
            stats_ensamble['mean_seeds'].append(stats['mean'])
        
        # 计算跨seed的平均值和标准差
        if stats_ensamble['mean_seeds']:
            stats_ensamble['mean_seeds'] = np.array(stats_ensamble['mean_seeds'])
            stats_ensamble['mean_ensamble'] = np.mean(stats_ensamble['mean_seeds'], axis=0)
            stats_ensamble['std_ensamble'] = np.std(stats_ensamble['mean_seeds'], axis=0)
        
        # 存储该方法的结果
        all_methods_results[method_name] = stats_ensamble
    
    # 绘制所有方法的对比图
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_methods_results)))
    
    for i, (method_name, results) in enumerate(all_methods_results.items()):
        if results['mean_ensamble'] is None:
            continue
            
        x = np.arange(len(results['mean_ensamble']))
        
        # 绘制曲线和误差带
        plt.plot(
            x, results['mean_ensamble'], 
            label=method_name, 
            marker='o',
            color=colors[i],
            linewidth=2
        )
        plt.fill_between(
            x,
            results['mean_ensamble'] - results['std_ensamble'],
            results['mean_ensamble'] + results['std_ensamble'],
            alpha=0.2,
            color=colors[i]
        )
    
    # 图表装饰
    plt.xlabel('Day Index', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Accuracy Comparison Across Days (Window Size={trial_num})', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.4, 1.0)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # 保存图表
    output_filename = os.path.join(current_dir, f"MI_accuracy_multiple_methods_comparison_{args.dataset_name}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存至: {output_filename}")
    
    # 导出统计数据到CSV
    stats_dfs = []
    for method_name, results in all_methods_results.items():
        if results['mean_ensamble'] is None:
            continue
            
        method_df = pd.DataFrame({
            'Day': np.arange(len(results['mean_ensamble'])),
            f'{method_name}_mean': results['mean_ensamble'],
            f'{method_name}_std': results['std_ensamble']
        })
        stats_dfs.append(method_df)
    
    # 合并所有方法的数据
    if stats_dfs:
        combined_stats = stats_dfs[0]
        for df in stats_dfs[1:]:
            combined_stats = pd.merge(combined_stats, df, on='Day', how='outer')
        
        csv_path = os.path.join(current_dir, f"MI_accuracy_multiple_methods_stats_{args.dataset_name}.csv")
        combined_stats.to_csv(csv_path, index=False)
        print(f"统计数据已保存至: {csv_path}")


def Test_time_visualizationClass_seeds_multiple_methods_1(class_num, trial_num, current_dir, data_name, log_paths, args, font_size=18):
    # 确保log_paths是列表
    method_paths = log_paths if isinstance(log_paths, list) else [log_paths]
    
    # 存储所有方法的结果
    all_methods_results = {}
    
    # 处理每个方法（原函数逻辑保持不变）
    for method_path in method_paths:
        method_name = os.path.basename(method_path.rstrip('/'))
        print(f"处理方法: {method_name}")
        
        # 查找该方法的seed文件
        _pattern = re.compile(r'_seed_\d+_pred\.csv$')
        csv_files = []
        for _file_name in os.listdir(method_path):
            if _file_name.endswith('.csv') and _pattern.search(_file_name):
                full_path = os.path.join(method_path, _file_name)
                csv_files.append(full_path)
        
        # 初始化存储结构
        stats_ensamble = {
            'mean_seeds': [],
            'mean_ensamble': None,
            'std_ensamble': None
        }
        
        # 处理每个seed文件
        for data_path in csv_files:
            stats = Test_time_visualizationClass(
                data_path, class_num, trial_num, current_dir, data_name, 
                imgdata_save=False  # 不保存中间结果
            )
            stats_ensamble['mean_seeds'].append(stats['mean'])
        
        # 计算跨seed的平均值和标准差
        if stats_ensamble['mean_seeds']:
            stats_ensamble['mean_seeds'] = np.array(stats_ensamble['mean_seeds'])
            stats_ensamble['mean_ensamble'] = np.mean(stats_ensamble['mean_seeds'], axis=0)
            stats_ensamble['std_ensamble'] = np.std(stats_ensamble['mean_seeds'], axis=0)
        
        # 存储该方法的结果
        all_methods_results[method_name] = stats_ensamble
    
    #################### 修改的绘图部分 ####################
    # 设置全局字体（Times New Roman）和大小 [1,3](@ref)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'mathtext.fontset': 'stix'  # 确保数学符号也使用Times New Roman
    })
    
    # 定义颜色方案（按顺序循环使用）
    base_colors = ['red', 'blue', 'green', 'purple', 'orange', 
                   'cyan', 'magenta', 'yellow', 'brown', 'pink',
                   'lime', 'teal', 'navy', 'maroon', 'olive']
    
    # 创建图形（使用Seaborn样式）
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # 计算最大天数（用于设置x轴范围）
    max_days = 0
    for results in all_methods_results.values():
        if results['mean_ensamble'] is not None:
            max_days = max(max_days, len(results['mean_ensamble']))
    
    # 循环绘制每个方法的结果
    for i, (method_name, results) in enumerate(all_methods_results.items()):
        method_name = method_name_trans(method_name)
        if results['mean_ensamble'] is None:
            continue
            
        # 横坐标从1开始（原代码从0开始）
        x = np.arange(1, len(results['mean_ensamble']) + 1)
        
        # 选择颜色（循环使用base_colors）
        color = base_colors[i % len(base_colors)]
        
        # 绘制曲线和误差带
        plt.plot(
            x, results['mean_ensamble'], 
            label=method_name, 
            marker='o',
            color=color,
            linewidth=2
        )
        """plt.fill_between(
            x,
            results['mean_ensamble'] - results['std_ensamble'],
            results['mean_ensamble'] + results['std_ensamble'],
            alpha=0.2,
            color=color
        )"""
    
    # 设置坐标轴标签
    plt.xlabel('Day', fontname='Times New Roman')
    plt.ylabel('Accuracy (%)', fontname='Times New Roman')
    
    # 设置图例
    plt.legend(loc='best', prop={'family': 'Times New Roman'})
    
    # 设置横轴只显示整数
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # 设置纵轴为百分比格式
    plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # 设置坐标轴范围（修改为从0.6开始，到最大天数+0.4结束）
    plt.ylim(0.4, 1.0)
    plt.xlim(left=1-0.3, right=max_days + 0.3)  # 确保横轴从0.6开始，到最大天数+0.4结束
    
    # 设置刻度字体为Times New Roman [1,3](@ref)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Times New Roman')
    
    # 保存图表（原函数逻辑）
    output_filename = os.path.join(current_dir, f"MI_accuracy_multiple_methods_comparison_{args.dataset_name}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存至: {output_filename}")
    
    # 导出统计数据到CSV（原函数逻辑）
    stats_dfs = []
    for method_name, results in all_methods_results.items():
        if results['mean_ensamble'] is None:
            continue
            
        method_df = pd.DataFrame({
            'Day': np.arange(1, len(results['mean_ensamble']) + 1),  # 横坐标从1开始
            f'{method_name}_mean': results['mean_ensamble'],
            f'{method_name}_std': results['std_ensamble']
        })
        stats_dfs.append(method_df)
    
    # 合并所有方法的数据
    if stats_dfs:
        combined_stats = stats_dfs[0]
        for df in stats_dfs[1:]:
            combined_stats = pd.merge(combined_stats, df, on='Day', how='outer')
        
        csv_path = os.path.join(current_dir, f"MI_accuracy_multiple_methods_stats_{args.dataset_name}.csv")
        combined_stats.to_csv(csv_path, index=False)
        print(f"统计数据已保存至: {csv_path}")


def Test_time_visualizationClass_seeds_multiple_methods_2(class_num, trial_num, current_dir, data_name, log_paths, args, font_size=22):
    # 确保log_paths是列表
    method_paths = log_paths if isinstance(log_paths, list) else [log_paths]
    
    # 存储所有方法的结果
    all_methods_results = {}
    
    # 处理每个方法
    for method_path in method_paths:
        method_name = os.path.basename(method_path.rstrip('/'))
        print(f"处理方法: {method_name}")
        
        # 查找该方法的seed文件
        _pattern = re.compile(r'_seed_\d+_pred\.csv$')
        csv_files = []
        for _file_name in os.listdir(method_path):
            if _file_name.endswith('.csv') and _pattern.search(_file_name):
                full_path = os.path.join(method_path, _file_name)
                csv_files.append(full_path)
        
        # 初始化存储结构
        stats_ensamble = {
            'mean_seeds': [],
            'mean_ensamble': None,
            'std_ensamble': None
        }
        
        # 处理每个seed文件
        for data_path in csv_files:
            stats = Test_time_visualizationClass(
                data_path, class_num, trial_num, current_dir, data_name, 
                imgdata_save=False  # 不保存中间结果
            )
            stats_ensamble['mean_seeds'].append(stats['mean'])
        
        # 计算跨seed的平均值和标准差
        if stats_ensamble['mean_seeds']:
            stats_ensamble['mean_seeds'] = np.array(stats_ensamble['mean_seeds'])
            stats_ensamble['mean_ensamble'] = np.mean(stats_ensamble['mean_seeds'], axis=0)
            stats_ensamble['std_ensamble'] = np.std(stats_ensamble['mean_seeds'], axis=0)
        
        # 存储该方法的结果
        all_methods_results[method_name] = stats_ensamble
    
    #################### 修改的绘图部分 ####################
    # 设置全局字体（Times New Roman）和大小
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'mathtext.fontset': 'stix'  # 确保数学符号也使用Times New Roman
    })
    
    # 定义样式方案（颜色、标记、线型）
    base_colors = ['red', 'blue', 'green', 'purple', 'orange', 
                   'cyan', 'magenta', 'brown', 'pink', 'olive',
                   'navy', 'maroon', 'teal', 'lime', 'gray']
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd', 'P', '.', '1']
    
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), 
                 (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                 (0, (5, 5)), (0, (5, 10)), (0, (10, 5))]
    
    # 创建图形（使用Seaborn样式）
    # sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    # 计算最大天数（用于设置x轴范围）
    max_days = 0
    for results in all_methods_results.values():
        if results['mean_ensamble'] is not None:
            max_days = max(max_days, len(results['mean_ensamble']))
    
    # 循环绘制每个方法的结果
    for i, (method_name, results) in enumerate(all_methods_results.items()):
        method_name = method_name_trans(method_name)
        if results['mean_ensamble'] is None:
            continue
            
        # 横坐标从1开始
        x = np.arange(1, len(results['mean_ensamble']) + 1)
        
        # 选择样式（循环使用base_colors、markers和linestyles）
        color = base_colors[i % len(base_colors)]
        marker = markers[i % len(markers)]
        # linestyle = linestyles[i % len(linestyles)]
        
        # 绘制曲线
        plt.plot(
            x, results['mean_ensamble']*100, 
            label=method_name, 
            marker=marker,           # 设置不同标记
            # linestyle=linestyle,     # 设置不同线型
            color=color,
            linewidth=2,
            markersize=8,            # 适当增大标记大小
            markevery=max(1, len(x)//10)  # 控制标记密度，避免过于密集
        )
    
    # 设置坐标轴标签
    plt.xlabel('Day', fontname='Times New Roman')
    plt.ylabel('Accuracy (%)', fontname='Times New Roman')
    
    # 设置图例
    plt.legend(loc='best', prop={'family': 'Times New Roman'})
    
    # 设置横轴只显示整数
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # 设置纵轴为百分比格式
    # plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    
    # 设置坐标轴范围
    plt.ylim(40, 100)
    plt.xlim(left=1-0.3, right=max_days + 0.3)
    
    # 设置刻度字体为Times New Roman
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname('Times New Roman')
    
    plt.tick_params(axis='x', which='both', length=0)  # 设置X轴刻度线长度为0

    # 保存图表
    output_filename = os.path.join(current_dir, f"MI_accuracy_multiple_methods_comparison_{args.dataset_name}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"对比图表已保存至: {output_filename}")
    
    # 导出统计数据到CSV
    stats_dfs = []
    for method_name, results in all_methods_results.items():
        if results['mean_ensamble'] is None:
            continue
            
        method_df = pd.DataFrame({
            'Day': np.arange(1, len(results['mean_ensamble']) + 1),
            f'{method_name}_mean': results['mean_ensamble'],
            f'{method_name}_std': results['std_ensamble']
        })
        stats_dfs.append(method_df)
    
    # 合并所有方法的数据
    if stats_dfs:
        combined_stats = stats_dfs[0]
        for df in stats_dfs[1:]:
            combined_stats = pd.merge(combined_stats, df, on='Day', how='outer')
        
        csv_path = os.path.join(current_dir, f"MI_accuracy_multiple_methods_stats_{args.dataset_name}.csv")
        combined_stats.to_csv(csv_path, index=False)
        print(f"统计数据已保存至: {csv_path}")
    
    # 显示图表
    plt.show()


def Test_time_visualizationClass_seeds_multiple_methods_3(class_num, trial_num, current_dir, data_name, log_paths, args, font_size=22):
    # 确保log_paths是列表
    method_paths = log_paths if isinstance(log_paths, list) else [log_paths]
    
    # 存储所有方法的结果
    all_methods_results = {}
    
    # 处理每个方法
    for method_path in method_paths:
        method_name = os.path.basename(method_path.rstrip('/'))
        print(f"处理方法: {method_name}")
        
        # 查找该方法的seed文件
        _pattern = re.compile(r'_seed_\d+_pred\.csv$')
        csv_files = []
        for _file_name in os.listdir(method_path):
            if _file_name.endswith('.csv') and _pattern.search(_file_name):
                full_path = os.path.join(method_path, _file_name)
                csv_files.append(full_path)
        
        # 初始化存储结构
        stats_ensamble = {
            'mean_seeds': [],
            'mean_ensamble': None,
        }
        
        # 处理每个seed文件
        for data_path in csv_files:
            stats = Test_time_visualizationClass(
                data_path, class_num, trial_num, current_dir, data_name, 
                imgdata_save=False  # 不保存中间结果
            )
            stats_ensamble['mean_seeds'].append(stats['mean'])
        
        # 计算跨seed的平均值
        if stats_ensamble['mean_seeds']:
            stats_ensamble['mean_seeds'] = np.array(stats_ensamble['mean_seeds'])
            stats_ensamble['mean_ensamble'] = np.mean(stats_ensamble['mean_seeds'], axis=0)
        
        # 存储该方法的结果
        all_methods_results[method_name] = stats_ensamble
    
    # 打印每个方法每一天的平均准确率
    for method_name, results in all_methods_results.items():
        if results['mean_ensamble'] is None:
            continue
        print(f"\n方法: {method_name}")
        for day, acc in enumerate(results['mean_ensamble'], start=1):
            print(f"Day {day}: {acc*100:.2f}%")
        overall_mean = np.mean(results['mean_ensamble'])
        print(f"总体平均准确率: {overall_mean*100:.2f}%")



def Test_time_visualizationClass_seeds_multiple_methods_4(class_num, trial_num, current_dir, data_name, log_paths, args, font_size=22):
    # 确保log_paths是列表
    method_paths = log_paths if isinstance(log_paths, list) else [log_paths]
    
    # 存储所有方法的结果
    all_methods_results = {}
    
    # 处理每个方法
    for method_path in method_paths:
        if method_path in ["./logs/Baselines-WBCIC-SHU-3C-e300-b64/t3a-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/"]:
            _class_num = 0
        else:
            _class_num = class_num

        method_name = os.path.basename(method_path.rstrip('/'))
        print(f"处理方法: {method_name}")
        
        # 查找该方法的seed文件
        _pattern = re.compile(r'_seed_\d+_pred\.csv$')
        csv_files = []
        for _file_name in os.listdir(method_path):
            if _file_name.endswith('.csv') and _pattern.search(_file_name):
                full_path = os.path.join(method_path, _file_name)
                csv_files.append(full_path)
        
        # 初始化存储结构
        stats_ensamble = {
            'mean_seeds': [],
            'mean_ensamble': None,
        }
        
        # 处理每个seed文件
        for data_path in csv_files:
            stats = Test_time_visualizationClass(
                data_path, _class_num, trial_num, current_dir, data_name, 
                imgdata_save=False  # 不保存中间结果
            )
            stats_ensamble['mean_seeds'].append(stats['mean'])
        
        # 计算跨seed的平均值
        if stats_ensamble['mean_seeds']:
            stats_ensamble['mean_seeds'] = np.array(stats_ensamble['mean_seeds'])
            stats_ensamble['mean_ensamble'] = np.mean(stats_ensamble['mean_seeds'], axis=0)
        
        # 存储该方法的结果
        all_methods_results[method_name] = stats_ensamble
    
    # 统计最大天数
    max_days = 0
    for results in all_methods_results.values():
        if results['mean_ensamble'] is not None:
            days = len(results['mean_ensamble'])
            if days > max_days:
                max_days = days

    # 打印并收集结果
    csv_rows = []
    for method_name, results in all_methods_results.items():
        if results['mean_ensamble'] is None:
            continue
        row = [method_name]
        for acc in results['mean_ensamble']:
            row.append(f"{acc*100:.2f}")
        overall_mean = np.mean(results['mean_ensamble'])
        row.append(f"{overall_mean*100:.2f}")
        csv_rows.append(row)
        # 打印
        print(f"\n方法: {method_name}")
        for day, acc in enumerate(results['mean_ensamble'], start=1):
            print(f"Day {day}: {acc*100:.2f}%")
        print(f"总体平均准确率: {overall_mean*100:.2f}%")

    # 构建header
    header = ["method"] + [f"day{d+1}" for d in range(max_days)] + ["overall_mean"]

    # 对齐每行长度
    for row in csv_rows:
        while len(row) < len(header):
            row.insert(-1, "")  # 在overall_mean前补空

    # 写入csv
    csv_path = os.path.join(current_dir, f"{data_name}_method_days_accuracy.csv")
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    print(f"\n结果已保存到: {csv_path}")


def method_name_trans(method_name):
    # return the transferred method
    if method_name in ["source-WBCIC-SHU-3C-EEGNet-4,2-e300-b64","Source-BNCI2014001-4-all-EEGNet-4,2-e300-b64"]:
        return "Source"
    if method_name in ["proposed_50_BNoff_batch8stride8_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-5-scale10","proposed_50_BNoff_batch8stride8_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-11-scale10","proposed_50_BNoff_batch8stride1_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-11-scale10"]:
        return "ATS-TTA"


"""
if __name__ == '__main__':

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='BNCI2014001', help='the data set name, now support BNCI2014001, BNCI2014002, BNCI2015001 from moabb')
    parser.add_argument('--data_save', type=str2bool, default=True, help='whether save the data to file')
    parser.add_argument('--data_path', type=str, default='./data/', help='the path to save the data from mobba dataset')
    parser.add_argument('--data_path_MI', type=str, default='/home/jyt/workspace/transfer_models/datasets_MI/hand_elbow/derivatives', help='the path to save the data from other datasets')
    parser.add_argument('--log_path', type=str, default='./logs/', help='the path to save the logs')
    parser.add_argument('--gpu_idx', type=int, default=0, help='index of GPU')
    parser.add_argument('--use_pretrained_model', type=str2bool, default=False, help='whether to use the pretrained model parameters')
    parser.add_argument('--finetune', type=str2bool, default=False, help='whether to finetune the model with part of the target data')
    parser.add_argument('--ft_volume', type=int, default=7*40, help='the amount of data for finetuning in target domain')
    parser.add_argument('--momentum', type=str2bool, default=False, help='whether to use the momentum updating for model parameters')
    parser.add_argument('--momentum_param', type=float, default=0.5, help='the value for momentum updating')
    parser.add_argument('--align', type=str2bool, default=True, help='use EA alignment and IEA alignment')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in offline training')
    parser.add_argument('--batch_size_online', type=int, default=8, help='batch size in online adaptation')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate in offline and online training')
    parser.add_argument('--lr_online', type=float, default=0.001, help='learning rate in online adaptation')
    parser.add_argument('--epoch', type=int, default=100, help='epoches in offline and online training')
    parser.add_argument('--backbone', type=str, default='EEGNet-4,2', help='backbone of the model')
    parser.add_argument('--param_runs', type=str, default='./runs/', help='folder for saving the run paramters')

    # for visulization
    parser.add_argument('--visualfile_csv', type=str, default="MI-elbow_rest_T-TIME_seed_1_pred.csv", help='the name of .csv file for visualization')
    parser.add_argument('--visualfile_trial', type=int, default=40, help='the num of trials in each segment for visualization')
    parser.add_argument('--visual_acc', type=str2bool, default=False, help='whether to show the acc with segments in visualization')
    parser.add_argument('--visual_ensamble', type=str2bool, default=False, help='whether to ensamble all the results from differernt seeds')
    # parser.add_argument('--tta_method', type=str, default=None, help='the method for visualization')

    args = parser.parse_args()
    data_name = args.dataset_name
    data_save = args.data_save
    data_path = args.data_path
    data_path_MI = args.data_path_MI
    log_path = args.log_path
    gpu_idx = args.gpu_idx
    use_pretrained_model = args.use_pretrained_model
    finetune = args.finetune
    ft_volume = args.ft_volume
    momentum = args.momentum
    momentum_param = args.momentum_param
    align = args.align
    batch_size = args.batch_size
    batch_size_online = args.batch_size_online
    lr = args.lr
    epoch = args.epoch
    backbone = args.backbone
    param_runs = args.param_runs
    lr_online = args.lr_online
    
    visualfile_csv = args.visualfile_csv
    visualfile_trial = args.visualfile_trial
    visual_acc = args.visual_acc
    # tta_method = args.tta_method
    visual_ensamble = args.visual_ensamble

    if backbone == 'EEGNet':
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 496
        if data_name == 'BNCI2014001-4-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 496
        if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'BNCI2014001-4-all': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 496
        if data_name == 'BNCI2014001-4-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 496
        if data_name == 'BNCI2014_004-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 560
        if data_name == 'BNCI2014_004-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 560
        if data_name == 'WBCIC-SHU-3C': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 11, 58, 3, 1000, 250, 900, 496
    if backbone == 'EEGNet-4,2':
        if data_name == 'BNCI2014001-4-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        if data_name == 'MI-hand_elbow': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-elbow_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'MI-hand_rest': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 25, 62, 2, 800, 200, 600, 200
        if data_name == 'BNCI2014001-4-all': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 576, 248
        if data_name == 'BNCI2014001-4-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 4, 1001, 250, 288, 248
        if data_name == 'BNCI2014_004-train': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 280
        if data_name == 'BNCI2014_004-test': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 3, 2, 1126, 250, 400, 280
        if data_name == 'WBCIC-SHU-3C': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 11, 58, 3, 1000, 250, 900, 248
    
    #if not visual_ensamble:
    #    stats = Test_time_visualizationClass(os.path.join(log_path, visualfile_csv), class_num=class_num, trial_num=visualfile_trial, current_dir=log_path, data_name=data_name)
    #else:
    #    Test_time_visualizationClass_seeds(class_num=class_num, trial_num=visualfile_trial, current_dir=log_path, data_name=data_name, args=args)

    # log_paths = ["./logs/Baselines-001-all-e300-b64/Source-BNCI2014001-4-all-EEGNet-4,2-e300-b64", "./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_50_BNoff_batch8stride8_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-11"]
    # save_path = "./visualization/methods_days/"

    # log_paths = ["./logs/Baselines-WBCIC-SHU-3C-e300-b64/source-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/", "./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_50_BNoff_batch8stride8_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-5/"]
    # save_path = "./visualization/methods_days/"

    if args.dataset_name == "BNCI2014001-4-all":
        
        visualfile_trial = 288
        log_paths = ["./logs/Baselines-001-all-e300-b64/Source-BNCI2014001-4-all-EEGNet-4,2-e300-b64", "./logs/Baselines-001-test-e300-b64-debugs/ours_debug_m_cls_process_1-BNCI2014001-4-all-EEGNet-4,2-e300-b64/proposed_50_BNoff_batch8stride1_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-11-scale10"]
        save_path = "./visualization/methods_days/"
        Test_time_visualizationClass_seeds_multiple_methods_2(class_num=class_num, trial_num=visualfile_trial, current_dir=save_path, data_name=data_name, log_paths=log_paths, args=args, font_size=36)
    elif args.dataset_name == "WBCIC-SHU-3C":
        
        visualfile_trial = 300
        
        #log_paths = ["./logs/Baselines-WBCIC-SHU-3C-e300-b64/source-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/", "./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_50_BNoff_batch8stride8_CE_KL_lcs_ConsSamples_selection_twoStage_weighted_lr0.001-5-scale10"]
        #save_path = "./visualization/methods_days/"
        #Test_time_visualizationClass_seeds_multiple_methods_2(class_num=class_num, trial_num=visualfile_trial, current_dir=save_path, data_name=data_name, log_paths=log_paths, args=args, font_size=36)
        
        log_paths = [
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/source-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/", 
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/bn-adapt-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/tent-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/pl-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/t3a-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/cotta-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/eata-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/sar-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/rotta-WBCIC-SHU-3C-EEGNet-4,2-e300-b64-onlinelr0.001-1/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/ttime-WBCIC-SHU-3C-EEGNet-4,2-e300-b64-4/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/sotta-WBCIC-SHU-3C-EEGNet-4,2-e300-b64-onlinelr0.001-1/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/motta-WBCIC-SHU-3C-EEGNet-4,2-e300-b64-1/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/aea-WBCIC-SHU-3C-EEGNet-4,2-e300-b64-1/",
            "./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p21",
            ]
        save_path = "./visualization/methods_days/"
        Test_time_visualizationClass_seeds_multiple_methods_4(class_num=class_num, trial_num=visualfile_trial, current_dir=save_path, data_name=data_name, log_paths=log_paths, args=args, font_size=36)
"""

if __name__ == '__main__':
    # ... 省略 argparse 部分，手动构造 args ...
    import argparse
    args = argparse.Namespace(dataset_name='WBCIC-SHU-3C', data_save=False)

    data_name = 'WBCIC-SHU-3C'
    class_num = 3
    visualfile_trial = 300
    save_path = "./visualization/methods_days/"

    log_paths = [
        #"./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-1-new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p3"
        #"./logs/Baselines-WBCIC-SHU-3C-e300-b64/proposed/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-3-params-1-new/proposed_57_BNoff_batch8stride1_CE_KL_review_ConsSamples_selection_two_stage_weighted_4_1_double_lr0.001-p3"
        #"./logs/Baselines-WBCIC-SHU-3C-e300-b64/eeg_otta-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
        #"./logs/Baselines-WBCIC-SHU-3C-e300-b64/bft-d-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
        "./logs/Baselines-WBCIC-SHU-3C-e300-b64/bft-a-WBCIC-SHU-3C-EEGNet-4,2-e300-b64/",
    ]

    Test_time_visualizationClass_seeds_multiple_methods_3(
        class_num=class_num,
        trial_num=visualfile_trial,
        current_dir=save_path,
        data_name=data_name,
        log_paths=log_paths,
        args=args
    )