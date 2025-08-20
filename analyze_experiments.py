#!/usr/bin/env python3
"""
实验分析脚本

功能：
1. 加载和分析实验数据
2. 比较不同实验的结果
3. 生成分析报告和可视化
4. 实验日志管理
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from datetime import datetime

from experiment_tracker import ExperimentManager, ExperimentTracker


def analyze_single_experiment(experiment_id: str, manager: ExperimentManager):
    """分析单个实验"""
    print(f"\n=== 分析实验: {experiment_id} ===")
    
    try:
        tracker = manager.load_experiment(experiment_id)
        config = tracker.config
        metrics = tracker.metrics
        
        print(f"实验名称: {config.experiment_name}")
        print(f"描述: {config.description}")
        print(f"标签: {config.tags}")
        
        # 训练结果
        if metrics['train_loss']:
            final_train_loss = metrics['train_loss'][-1]
            min_train_loss = min(metrics['train_loss'])
            print(f"\n训练损失:")
            print(f"  最终: {final_train_loss:.4f}")
            print(f"  最小: {min_train_loss:.4f}")
        
        if metrics['val_loss']:
            final_val_loss = metrics['val_loss'][-1]
            min_val_loss = min(metrics['val_loss'])
            print(f"\n验证损失:")
            print(f"  最终: {final_val_loss:.4f}")
            print(f"  最小: {min_val_loss:.4f}")
        
        if metrics['val_perplexity']:
            final_perplexity = metrics['val_perplexity'][-1]
            min_perplexity = min(metrics['val_perplexity'])
            print(f"\n验证困惑度:")
            print(f"  最终: {final_perplexity:.2f}")
            print(f"  最小: {min_perplexity:.2f}")
        
        # 训练统计
        if metrics['iterations']:
            total_iterations = metrics['iterations'][-1]
            print(f"\n训练统计:")
            print(f"  总迭代次数: {total_iterations}")
            
            if metrics['wallclock_time']:
                total_time = metrics['wallclock_time'][-1]
                avg_time_per_iter = total_time / total_iterations
                print(f"  总训练时间: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
                print(f"  平均每迭代时间: {avg_time_per_iter:.4f} 秒")
        
        # 模型配置
        print(f"\n模型配置:")
        print(f"  词汇表大小: {config.vocab_size}")
        print(f"  上下文长度: {config.context_length}")
        print(f"  模型维度: {config.d_model}")
        print(f"  层数: {config.num_layers}")
        print(f"  注意力头数: {config.num_heads}")
        print(f"  批次大小: {config.batch_size}")
        print(f"  学习率: {config.learning_rate}")
        
        return {
            'experiment_id': experiment_id,
            'name': config.experiment_name,
            'final_train_loss': final_train_loss if metrics['train_loss'] else None,
            'final_val_loss': final_val_loss if metrics['val_loss'] else None,
            'final_perplexity': final_perplexity if metrics['val_perplexity'] else None,
            'total_iterations': total_iterations if metrics['iterations'] else None,
            'total_time': total_time if metrics['wallclock_time'] else None,
            'config': config
        }
        
    except Exception as e:
        print(f"分析实验 {experiment_id} 失败: {e}")
        return None


def compare_experiments(experiment_ids: List[str], manager: ExperimentManager):
    """比较多个实验"""
    print(f"\n=== 比较实验 ===")
    
    results = []
    for exp_id in experiment_ids:
        result = analyze_single_experiment(exp_id, manager)
        if result:
            results.append(result)
    
    if not results:
        print("没有可比较的实验")
        return
    
    # 创建比较表格
    df = pd.DataFrame(results)
    
    # 选择要显示的列
    display_columns = [
        'name', 'final_train_loss', 'final_val_loss', 'final_perplexity',
        'total_iterations', 'total_time'
    ]
    
    print("\n实验比较表:")
    print(df[display_columns].to_string(index=False))
    
    # 可视化比较
    plot_experiment_comparison(results)
    
    return df


def plot_experiment_comparison(results: List[Dict[str, Any]]):
    """绘制实验比较图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('实验比较', fontsize=16)
    
    # 准备数据
    names = [r['name'] for r in results]
    train_losses = [r['final_train_loss'] for r in results if r['final_train_loss'] is not None]
    val_losses = [r['final_val_loss'] for r in results if r['final_val_loss'] is not None]
    perplexities = [r['final_perplexity'] for r in results if r['final_perplexity'] is not None]
    times = [r['total_time'] for r in results if r['total_time'] is not None]
    
    # 训练损失比较
    if train_losses:
        axes[0, 0].bar(range(len(train_losses)), train_losses)
        axes[0, 0].set_title('最终训练损失')
        axes[0, 0].set_ylabel('训练损失')
        axes[0, 0].set_xticks(range(len(train_losses)))
        axes[0, 0].set_xticklabels(names, rotation=45)
    
    # 验证损失比较
    if val_losses:
        axes[0, 1].bar(range(len(val_losses)), val_losses, color='orange')
        axes[0, 1].set_title('最终验证损失')
        axes[0, 1].set_ylabel('验证损失')
        axes[0, 1].set_xticks(range(len(val_losses)))
        axes[0, 1].set_xticklabels(names, rotation=45)
    
    # 困惑度比较
    if perplexities:
        axes[1, 0].bar(range(len(perplexities)), perplexities, color='green')
        axes[1, 0].set_title('最终困惑度')
        axes[1, 0].set_ylabel('困惑度')
        axes[1, 0].set_xticks(range(len(perplexities)))
        axes[1, 0].set_xticklabels(names, rotation=45)
    
    # 训练时间比较
    if times:
        axes[1, 1].bar(range(len(times)), [t/3600 for t in times], color='red')
        axes[1, 1].set_title('训练时间')
        axes[1, 1].set_ylabel('时间 (小时)')
        axes[1, 1].set_xticks(range(len(times)))
        axes[1, 1].set_xticklabels(names, rotation=45)
    
    plt.tight_layout()
    plt.show()


def plot_training_curves(experiment_ids: List[str], manager: ExperimentManager):
    """绘制训练曲线比较"""
    plt.figure(figsize=(15, 10))
    
    for exp_id in experiment_ids:
        try:
            tracker = manager.load_experiment(exp_id)
            if tracker.metrics['train_loss'] and tracker.metrics['iterations']:
                plt.plot(tracker.metrics['iterations'], tracker.metrics['train_loss'], 
                        label=f"{tracker.config.experiment_name} (训练)", linewidth=2)
            if tracker.metrics['val_loss'] and tracker.metrics['iterations']:
                plt.plot(tracker.metrics['iterations'], tracker.metrics['val_loss'], 
                        label=f"{tracker.config.experiment_name} (验证)", linewidth=2, linestyle='--')
        except Exception as e:
            print(f"加载实验 {exp_id} 失败: {e}")
    
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练曲线比较')
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_analysis_report(experiment_ids: List[str], manager: ExperimentManager):
    """生成分析报告"""
    print(f"\n=== 生成分析报告 ===")
    
    report_content = f"""# 实验分析报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概述

本报告分析了 {len(experiment_ids)} 个实验的结果。

## 实验列表

"""
    
    # 添加每个实验的详细信息
    for i, exp_id in enumerate(experiment_ids, 1):
        try:
            tracker = manager.load_experiment(exp_id)
            config = tracker.config
            metrics = tracker.metrics
            
            report_content += f"""
### 实验 {i}: {config.experiment_name}

**实验ID**: {exp_id}
**描述**: {config.description}
**标签**: {', '.join(config.tags)}

**配置参数**:
- 词汇表大小: {config.vocab_size}
- 上下文长度: {config.context_length}
- 模型维度: {config.d_model}
- 层数: {config.num_layers}
- 注意力头数: {config.num_heads}
- 批次大小: {config.batch_size}
- 学习率: {config.learning_rate}

**训练结果**:
"""
            
            if metrics['train_loss']:
                final_train_loss = metrics['train_loss'][-1]
                report_content += f"- 最终训练损失: {final_train_loss:.4f}\n"
            
            if metrics['val_loss']:
                final_val_loss = metrics['val_loss'][-1]
                report_content += f"- 最终验证损失: {final_val_loss:.4f}\n"
            
            if metrics['val_perplexity']:
                final_perplexity = metrics['val_perplexity'][-1]
                report_content += f"- 最终困惑度: {final_perplexity:.2f}\n"
            
            if metrics['iterations'] and metrics['wallclock_time']:
                total_iterations = metrics['iterations'][-1]
                total_time = metrics['wallclock_time'][-1]
                report_content += f"- 总迭代次数: {total_iterations}\n"
                report_content += f"- 总训练时间: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)\n"
            
            report_content += "\n"
            
        except Exception as e:
            report_content += f"### 实验 {i}: {exp_id}\n\n加载失败: {e}\n\n"
    
    # 添加比较分析
    report_content += """
## 比较分析

### 最佳性能实验

"""
    
    # 找出最佳实验
    best_experiments = {}
    for exp_id in experiment_ids:
        try:
            tracker = manager.load_experiment(exp_id)
            metrics = tracker.metrics
            
            if metrics['val_loss']:
                best_experiments['val_loss'] = best_experiments.get('val_loss', []) + [
                    (exp_id, tracker.config.experiment_name, min(metrics['val_loss']))
                ]
            
            if metrics['val_perplexity']:
                best_experiments['perplexity'] = best_experiments.get('perplexity', []) + [
                    (exp_id, tracker.config.experiment_name, min(metrics['val_perplexity']))
                ]
                
        except Exception as e:
            continue
    
    if 'val_loss' in best_experiments:
        best_val_loss = min(best_experiments['val_loss'], key=lambda x: x[2])
        report_content += f"- **最佳验证损失**: {best_val_loss[1]} ({best_val_loss[2]:.4f})\n"
    
    if 'perplexity' in best_experiments:
        best_perplexity = min(best_experiments['perplexity'], key=lambda x: x[2])
        report_content += f"- **最佳困惑度**: {best_perplexity[1]} ({best_perplexity[2]:.2f})\n"
    
    report_content += """
### 建议

基于实验结果，建议：

1. 继续优化表现最好的实验配置
2. 尝试不同的超参数组合
3. 增加训练数据或模型规模
4. 实现更复杂的正则化技术

## 结论

通过系统性的实验分析，我们能够更好地理解不同配置对模型性能的影响，为后续的模型优化提供指导。

"""
    
    # 保存报告
    report_path = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"分析报告已保存到: {report_path}")
    return report_path


def main():
    """主函数"""
    print("实验分析工具")
    print("=" * 50)
    
    # 初始化实验管理器
    manager = ExperimentManager()
    
    # 列出所有实验
    experiments = manager.list_experiments()
    
    if not experiments:
        print("没有找到实验数据")
        return
    
    print(f"找到 {len(experiments)} 个实验:")
    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp['name']} ({exp['experiment_id']})")
    
    # 用户选择
    while True:
        print("\n请选择操作:")
        print("1. 分析单个实验")
        print("2. 比较多个实验")
        print("3. 绘制训练曲线")
        print("4. 生成分析报告")
        print("5. 退出")
        
        choice = input("请输入选择 (1-5): ").strip()
        
        if choice == '1':
            exp_id = input("请输入实验ID: ").strip()
            analyze_single_experiment(exp_id, manager)
            
        elif choice == '2':
            exp_ids = input("请输入实验ID (用逗号分隔): ").strip().split(',')
            exp_ids = [exp_id.strip() for exp_id in exp_ids]
            compare_experiments(exp_ids, manager)
            
        elif choice == '3':
            exp_ids = input("请输入实验ID (用逗号分隔): ").strip().split(',')
            exp_ids = [exp_id.strip() for exp_id in exp_ids]
            plot_training_curves(exp_ids, manager)
            
        elif choice == '4':
            exp_ids = input("请输入实验ID (用逗号分隔): ").strip().split(',')
            exp_ids = [exp_id.strip() for exp_id in exp_ids]
            generate_analysis_report(exp_ids, manager)
            
        elif choice == '5':
            print("退出分析工具")
            break
            
        else:
            print("无效选择，请重新输入")


if __name__ == '__main__':
    main()
