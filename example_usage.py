#!/usr/bin/env python3
"""
示例使用脚本 - 演示完整的数据准备和训练流程

这个脚本展示了如何：
1. 准备示例数据
2. 运行数据预处理
3. 配置训练参数
4. 开始训练
"""

import os
import json
import numpy as np
from pathlib import Path


def create_sample_data():
    """创建示例文本数据"""
    sample_texts = [
        "Hello world! This is a sample text for training.",
        "Machine learning is an exciting field of study.",
        "Natural language processing helps computers understand human language.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Training large language models requires significant computational resources.",
        "The quality of training data greatly affects model performance.",
        "Regularization techniques help prevent overfitting in neural networks.",
        "Gradient descent is a fundamental optimization algorithm for training neural networks.",
        "Backpropagation efficiently computes gradients for all parameters.",
        "Batch normalization helps stabilize training of deep networks.",
        "Dropout is a regularization technique that prevents co-adaptation of neurons.",
        "Learning rate scheduling can improve training convergence.",
        "Early stopping helps prevent overfitting by monitoring validation performance.",
        "Cross-validation provides a robust estimate of model performance.",
        "Hyperparameter tuning is crucial for achieving optimal model performance.",
        "Transfer learning allows models to leverage knowledge from pre-trained models.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Data augmentation increases the effective size of training datasets."
    ]
    
    # 创建示例数据目录
    os.makedirs('example_data', exist_ok=True)
    
    # 保存示例文本
    with open('example_data/sample_texts.txt', 'w', encoding='utf-8') as f:
        for text in sample_texts:
            f.write(text + '\n')
    
    print("示例数据已创建: example_data/sample_texts.txt")


def run_data_preparation():
    """运行数据准备"""
    print("\n=== 数据准备阶段 ===")
    
    # 检查示例数据是否存在
    if not os.path.exists('example_data/sample_texts.txt'):
        print("示例数据不存在，正在创建...")
        create_sample_data()
    
    # 运行数据准备脚本
    import subprocess
    
    cmd = [
        'python', 'prepare_data.py',
        '--input_dir', 'example_data',
        '--output_dir', 'data',
        '--vocab_size', '1000',
        '--min_freq', '1',
        '--test_split', '0.2'
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("数据准备成功完成！")
        print(result.stdout)
    else:
        print("数据准备失败:")
        print(result.stderr)
        return False
    
    return True


def create_training_config():
    """创建训练配置文件"""
    print("\n=== 创建训练配置 ===")
    
    # 读取数据统计信息
    with open('data/data_stats.json', 'r') as f:
        stats = json.load(f)
    
    # 创建适合小规模训练的配置
    config = {
        "vocab_size": stats['vocab_size'],
        "context_length": 64,  # 较小的上下文长度
        "d_model": 256,        # 较小的模型维度
        "num_layers": 4,       # 较少的层数
        "num_heads": 8,        # 较少的注意力头
        "d_ff": 1024,          # 较小的前馈网络
        "max_seq_len": 128,
        "rope_theta": 10000.0,
        "batch_size": 8,       # 较小的批次大小
        "max_iters": 1000,     # 较少的迭代次数
        "learning_rate": 1e-3,
        "min_learning_rate": 1e-5,
        "warmup_iters": 100,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
        "data_path": "data/train_data.bin",
        "split_ratio": 0.8,
        "checkpoint_dir": "checkpoints/example",
        "save_every": 200,
        "eval_every": 100,
        "log_every": 50,
        "device": "cpu",       # 使用CPU进行示例训练
        "use_wandb": False
    }
    
    # 保存配置
    os.makedirs('configs', exist_ok=True)
    config_path = 'configs/example_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"训练配置已保存到: {config_path}")
    return config_path


def run_training(config_path):
    """运行训练"""
    print("\n=== 开始训练 ===")
    
    import subprocess
    
    cmd = [
        'python', 'train.py',
        '--config', config_path
    ]
    
    print(f"运行命令: {' '.join(cmd)}")
    print("注意：这是一个示例训练，使用较小的模型和较少的迭代次数")
    print("训练可能需要几分钟时间...")
    
    # 运行训练（不捕获输出，让用户看到实时进度）
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n训练成功完成！")
        print("检查点文件保存在: checkpoints/example/")
    else:
        print("\n训练过程中出现错误")
        return False
    
    return True


def show_results():
    """显示训练结果"""
    print("\n=== 训练结果 ===")
    
    checkpoint_dir = Path('checkpoints/example')
    if checkpoint_dir.exists():
        print("生成的检查点文件:")
        for file in checkpoint_dir.glob('*.pt'):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.1f} MB)")
    
    log_file = Path('training.log')
    if log_file.exists():
        print(f"\n训练日志文件: {log_file}")
        print("最后几行日志:")
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"  {line.strip()}")


def main():
    """主函数"""
    print("=== 语言模型训练示例 ===")
    print("这个脚本将演示完整的数据准备和训练流程")
    
    # 步骤1: 数据准备
    if not run_data_preparation():
        print("数据准备失败，退出")
        return
    
    # 步骤2: 创建训练配置
    config_path = create_training_config()
    
    # 步骤3: 运行训练
    if not run_training(config_path):
        print("训练失败，退出")
        return
    
    # 步骤4: 显示结果
    show_results()
    
    print("\n=== 示例完成 ===")
    print("您现在已经有了一个完整的训练流程！")
    print("可以修改配置文件来调整模型参数和训练设置")


if __name__ == '__main__':
    main()
