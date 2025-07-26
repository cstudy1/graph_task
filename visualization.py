#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图神经网络实验结果可视化脚本
用于分析和可视化不同图神经网络模型的实验结果
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def load_data():
    """加载CSV数据"""
    try:
        # 读取CSV文件 - 使用逗号作为分隔符
        df = pd.read_csv('result1.csv', sep=',')
        print("Dataset shape:", df.shape)
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
        print("\nBasic statistics:")
        print(df.describe())
        return df
    except Exception as e:
        print(f"Error reading data: {e}")
        return None

def analyze_model_performance(df):
    """分析模型性能"""
    print("\n=== Model Performance Analysis ===")
    print("Average accuracy by model:")
    model_acc = df.groupby('model')['acc'].mean().sort_values(ascending=False)
    print(model_acc)
    
    print("\nAverage F1-score by model:")
    model_f1 = df.groupby('model')['f1_micro'].mean().sort_values(ascending=False)
    print(model_f1)
    
    print("\nAverage AUROC by model:")
    model_auroc = df.groupby('model')['auroc'].mean().sort_values(ascending=False)
    print(model_auroc)

def plot_model_comparison(df):
    """绘制模型性能对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 准确率对比
    sns.boxplot(data=df, x='model', y='acc', ax=axes[0])
    axes[0].set_title('Model Accuracy Distribution', fontsize=14)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].tick_params(axis='x', rotation=45)
    
    # F1分数对比
    sns.boxplot(data=df, x='model', y='f1_micro', ax=axes[1])
    axes[1].set_title('Model F1-Score Distribution', fontsize=14)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('F1-Score')
    axes[1].tick_params(axis='x', rotation=45)
    
    # AUROC对比
    sns.boxplot(data=df, x='model', y='auroc', ax=axes[2])
    axes[2].set_title('Model AUROC Distribution', fontsize=14)
    axes[2].set_xlabel('Model')
    axes[2].set_ylabel('AUROC')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_layer_analysis(df):
    """分析不同层数对性能的影响"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 准确率 vs 层数
    sns.boxplot(data=df, x='num_layers', y='acc', ax=axes[0])
    axes[0].set_title('Impact of Number of Layers on Accuracy', fontsize=14)
    axes[0].set_xlabel('Number of Layers')
    axes[0].set_ylabel('Accuracy')
    
    # F1分数 vs 层数
    sns.boxplot(data=df, x='num_layers', y='f1_micro', ax=axes[1])
    axes[1].set_title('Impact of Number of Layers on F1-Score', fontsize=14)
    axes[1].set_xlabel('Number of Layers')
    axes[1].set_ylabel('F1-Score')
    
    # AUROC vs 层数
    sns.boxplot(data=df, x='num_layers', y='auroc', ax=axes[2])
    axes[2].set_title('Impact of Number of Layers on AUROC', fontsize=14)
    axes[2].set_xlabel('Number of Layers')
    axes[2].set_ylabel('AUROC')
    
    plt.tight_layout()
    plt.savefig('layer_analysis.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_lr_analysis(df):
    """分析学习率对性能的影响"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 准确率 vs 学习率
    sns.boxplot(data=df, x='lr', y='acc', ax=axes[0])
    axes[0].set_title('Impact of Learning Rate on Accuracy', fontsize=14)
    axes[0].set_xlabel('Learning Rate')
    axes[0].set_ylabel('Accuracy')
    
    # F1分数 vs 学习率
    sns.boxplot(data=df, x='lr', y='f1_micro', ax=axes[1])
    axes[1].set_title('Impact of Learning Rate on F1-Score', fontsize=14)
    axes[1].set_xlabel('Learning Rate')
    axes[1].set_ylabel('F1-Score')
    
    # AUROC vs 学习率
    sns.boxplot(data=df, x='lr', y='auroc', ax=axes[2])
    axes[2].set_title('Impact of Learning Rate on AUROC', fontsize=14)
    axes[2].set_xlabel('Learning Rate')
    axes[2].set_ylabel('AUROC')
    
    plt.tight_layout()
    plt.savefig('lr_analysis.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_heatmaps(df):
    """创建热力图显示模型、层数和学习率的组合效果"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 准确率热力图
    acc_pivot = df.pivot_table(values='acc', index='model', columns='num_layers', aggfunc='mean')
    sns.heatmap(acc_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0])
    axes[0].set_title('Model-Layers vs Accuracy', fontsize=14)
    
    # F1分数热力图
    f1_pivot = df.pivot_table(values='f1_micro', index='model', columns='num_layers', aggfunc='mean')
    sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[1])
    axes[1].set_title('Model-Layers vs F1-Score', fontsize=14)
    
    # AUROC热力图
    auroc_pivot = df.pivot_table(values='auroc', index='model', columns='num_layers', aggfunc='mean')
    sns.heatmap(auroc_pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[2])
    axes[2].set_title('Model-Layers vs AUROC', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('heatmaps.png', dpi=600, bbox_inches='tight')
    plt.show()

def find_best_configurations(df):
    """找出最佳配置"""
    print("\n=== Best Configuration Analysis ===")
    print("Best accuracy configuration:")
    best_acc = df.loc[df['acc'].idxmax()]
    print(best_acc)
    
    print("\nBest F1-score configuration:")
    best_f1 = df.loc[df['f1_micro'].idxmax()]
    print(best_f1)
    
    print("\nBest AUROC configuration:")
    best_auroc = df.loc[df['auroc'].idxmax()]
    print(best_auroc)

def plot_radar_chart(df):
    """创建综合性能雷达图（使用极坐标）"""
    # 计算每个模型的平均性能
    model_performance = df.groupby('model')[['acc', 'f1_micro', 'auroc']].mean()
    
    # 创建极坐标图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 设置角度
    angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (model, performance) in enumerate(model_performance.iterrows()):
        values = [performance['acc'], performance['f1_micro'], performance['auroc']]
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'F1-Score', 'AUROC'])
    ax.set_title('Model Performance Comparison', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=600, bbox_inches='tight')
    plt.show()

def plot_line_charts(df):
    """绘制学习曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 按模型分组的性能趋势
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        axes[0, 0].plot(model_data['lr'], model_data['acc'], 'o-', label=model, linewidth=2)
        axes[0, 1].plot(model_data['lr'], model_data['f1_micro'], 'o-', label=model, linewidth=2)
        axes[1, 0].plot(model_data['lr'], model_data['auroc'], 'o-', label=model, linewidth=2)
        axes[1, 1].plot(model_data['num_layers'], model_data['acc'], 'o-', label=model, linewidth=2)
    
    axes[0, 0].set_title('Learning Rate vs Accuracy', fontsize=14)
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    
    axes[0, 1].set_title('Learning Rate vs F1-Score', fontsize=14)
    axes[0, 1].set_xlabel('Learning Rate')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].legend()
    axes[0, 1].set_xscale('log')
    
    axes[1, 0].set_title('Learning Rate vs AUROC', fontsize=14)
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('AUROC')
    axes[1, 0].legend()
    axes[1, 0].set_xscale('log')
    
    axes[1, 1].set_title('Number of Layers vs Accuracy', fontsize=14)
    axes[1, 1].set_xlabel('Number of Layers')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('line_charts.png', dpi=600, bbox_inches='tight')
    plt.show()

def generate_summary_report(df):
    """生成总结报告"""
    print("\n" + "="*50)
    print("Graph Neural Network Experimental Results Summary")
    print("="*50)
    
    # 最佳模型
    best_model_acc = df.groupby('model')['acc'].mean().idxmax()
    best_model_f1 = df.groupby('model')['f1_micro'].mean().idxmax()
    best_model_auroc = df.groupby('model')['auroc'].mean().idxmax()
    
    print(f"\n1. Best Accuracy Model: {best_model_acc}")
    print(f"2. Best F1-Score Model: {best_model_f1}")
    print(f"3. Best AUROC Model: {best_model_auroc}")
    
    # 最佳配置
    best_config = df.loc[df['acc'].idxmax()]
    print(f"\n4. Global Best Configuration:")
    print(f"   Model: {best_config['model']}")
    print(f"   Number of Layers: {best_config['num_layers']}")
    print(f"   Learning Rate: {best_config['lr']}")
    print(f"   Accuracy: {best_config['acc']:.3f}")
    print(f"   F1-Score: {best_config['f1_micro']:.3f}")
    print(f"   AUROC: {best_config['auroc']:.3f}")
    
    # 性能统计
    print(f"\n5. Performance Statistics:")
    print(f"   Average Accuracy: {df['acc'].mean():.3f} ± {df['acc'].std():.3f}")
    print(f"   Average F1-Score: {df['f1_micro'].mean():.3f} ± {df['f1_micro'].std():.3f}")
    print(f"   Average AUROC: {df['auroc'].mean():.3f} ± {df['auroc'].std():.3f}")

def main():
    """主函数"""
    print("Loading data...")
    df = load_data()
    
    if df is None:
        print("Data loading failed, program exit")
        return
    
    print("\nStarting analysis...")
    
    # 执行各种分析
    analyze_model_performance(df)
    find_best_configurations(df)
    generate_summary_report(df)
    
    print("\nGenerating visualization charts...")
    
    # 生成各种图表
    plot_model_comparison(df)
    plot_layer_analysis(df)
    plot_lr_analysis(df)
    plot_heatmaps(df)
    plot_radar_chart(df)
    plot_line_charts(df)
    
    print("\nAll analysis completed! Charts saved as PNG files.")

if __name__ == "__main__":
    main() 