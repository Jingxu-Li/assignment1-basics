#!/usr/bin/env python3
"""
训练脚本 - 实现语言模型的训练循环

功能特性：
1. 超参数配置和控制
2. 内存高效的数据加载（使用np.memmap）
3. 检查点序列化
4. 定期日志记录（控制台和wandb）
"""

import argparse
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm

# 导入自定义模块
from cs336_basics.MyLMBlock import MyLMBlock
from cs336_basics.MyAdam import AdamW
from cs336_basics.utils import (
    cross_entropy, 
    get_lr_cosine_schedule, 
    get_gradient_clipping_fn,
    get_batch,
    save_checkpoint,
    load_checkpoint
)

# 导入实验跟踪模块
from experiment_tracker import ExperimentTracker, ExperimentConfig, experiment_context


class MemmapDataset(Dataset):
    """使用np.memmap的内存高效数据集"""
    
    def __init__(self, data_path: str, context_length: int, split_ratio: float = 0.9):
        """
        Args:
            data_path: 数据文件路径
            context_length: 上下文长度
            split_ratio: 训练集比例
        """
        self.context_length = context_length
        
        # 使用memmap加载数据
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # 计算训练集和验证集的分割点
        split_idx = int(len(self.data) * split_ratio)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]
        
        # 确保数据长度足够
        if len(self.train_data) < context_length + 1:
            raise ValueError(f"训练数据长度 {len(self.train_data)} 小于 context_length + 1 = {context_length + 1}")
        if len(self.val_data) < context_length + 1:
            raise ValueError(f"验证数据长度 {len(self.val_data)} 小于 context_length + 1 = {context_length + 1}")
    
    def get_train_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """获取训练批次"""
        return get_batch(self.train_data, batch_size, self.context_length, device)
    
    def get_val_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """获取验证批次"""
        return get_batch(self.val_data, batch_size, self.context_length, device)


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self, **kwargs):
        # 实验信息
        self.experiment_name = kwargs.get('experiment_name', 'default_experiment')
        self.experiment_description = kwargs.get('experiment_description', 'Default training experiment')
        self.experiment_tags = kwargs.get('experiment_tags', ['training'])
        
        # 模型参数
        self.vocab_size = kwargs.get('vocab_size', 50257)
        self.context_length = kwargs.get('context_length', 1024)
        self.d_model = kwargs.get('d_model', 768)
        self.num_layers = kwargs.get('num_layers', 12)
        self.num_heads = kwargs.get('num_heads', 12)
        self.d_ff = kwargs.get('d_ff', 3072)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        
        # 训练参数
        self.batch_size = kwargs.get('batch_size', 32)
        self.max_iters = kwargs.get('max_iters', 10000)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)
        self.min_learning_rate = kwargs.get('min_learning_rate', 1e-5)
        self.warmup_iters = kwargs.get('warmup_iters', 1000)
        self.weight_decay = kwargs.get('weight_decay', 0.1)
        self.grad_clip = kwargs.get('grad_clip', 1.0)
        
        # 数据参数
        self.data_path = kwargs.get('data_path', 'data/tokenized_data.bin')
        self.split_ratio = kwargs.get('split_ratio', 0.9)
        
        # 检查点和日志参数
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 'checkpoints')
        self.save_every = kwargs.get('save_every', 1000)
        self.eval_every = kwargs.get('eval_every', 500)
        self.log_every = kwargs.get('log_every', 100)
        
        # 设备参数
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = kwargs.get('seed', 42)
        
        # wandb参数
        self.use_wandb = kwargs.get('use_wandb', False)
        self.wandb_project = kwargs.get('wandb_project', 'cs336-training')
        self.wandb_name = kwargs.get('wandb_name', None)
    
    def to_experiment_config(self) -> ExperimentConfig:
        """转换为实验配置"""
        return ExperimentConfig(
            experiment_name=self.experiment_name,
            experiment_id="",  # 将由ExperimentTracker生成
            description=self.experiment_description,
            tags=self.experiment_tags,
            vocab_size=self.vocab_size,
            context_length=self.context_length,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            d_ff=self.d_ff,
            rope_theta=self.rope_theta,
            batch_size=self.batch_size,
            max_iters=self.max_iters,
            learning_rate=self.learning_rate,
            min_learning_rate=self.min_learning_rate,
            warmup_iters=self.warmup_iters,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            data_path=self.data_path,
            split_ratio=self.split_ratio,
            device=self.device,
            seed=self.seed
        )
    
    def save(self, path: str):
        """保存配置到文件"""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """从文件加载配置"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """设置日志记录"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )
    return logging.getLogger(__name__)


def evaluate_model(model: nn.Module, dataset: MemmapDataset, batch_size: int, 
                  device: str, num_eval_batches: int = 10) -> Dict[str, float]:
    """评估模型性能"""
    try:
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for _ in range(num_eval_batches):
                try:
                    x, y = dataset.get_val_batch(batch_size, device)
                    logits = model(x)
                    loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                    
                    total_loss += loss.item() * x.numel()
                    total_tokens += x.numel()
                except Exception as e:
                    print(f"评估批次发生错误: {e}")
                    continue
        
        if total_tokens == 0:
            # 如果所有批次都失败了，返回默认值
            return {
                'val_loss': float('inf'),
                'val_perplexity': float('inf')
            }
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity
        }
    except Exception as e:
        print(f"评估模型时发生错误: {e}")
        # 返回默认值
        return {
            'val_loss': float('inf'),
            'val_perplexity': float('inf')
        }


def train_model(config: TrainingConfig):
    """主训练函数"""
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 创建实验配置
    exp_config = config.to_experiment_config()
    
    # 使用实验上下文管理器
    with experiment_context(
        experiment_name=config.experiment_name,
        config=exp_config,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project
    ) as tracker:
        
        logger = tracker.logger
        
        # 创建检查点目录
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # 设置设备
        device = torch.device(config.device)
        logger.info(f"使用设备: {device}")
        
        # 加载数据集
        logger.info(f"加载数据集: {config.data_path}")
        dataset = MemmapDataset(config.data_path, config.context_length, config.split_ratio)
        logger.info(f"训练数据大小: {len(dataset.train_data)}, 验证数据大小: {len(dataset.val_data)}")
        
        # 初始化模型
        logger.info("初始化模型...")
        model = MyLMBlock(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
            in_indices=None  # 这个参数在forward中不会使用
        ).to(device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"模型参数总数: {total_params:,}, 可训练参数: {trainable_params:,}")
        
        # 记录模型信息
        tracker.log_experiment_note(f"模型参数总数: {total_params:,}")
        tracker.log_experiment_note(f"可训练参数: {trainable_params:,}")
        
        # 初始化优化器
        optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 训练状态
        start_iter = 0
        best_val_loss = float('inf')
        
        # 加载检查点（如果存在）
        checkpoint_path = os.path.join(config.checkpoint_dir, 'latest.pt')
        if os.path.exists(checkpoint_path):
            logger.info(f"加载检查点: {checkpoint_path}")
            start_iter = load_checkpoint(checkpoint_path, model, optimizer)
            logger.info(f"从迭代 {start_iter} 开始训练")
            tracker.log_experiment_note(f"从检查点恢复训练，起始迭代: {start_iter}")
        
        # 训练循环
        logger.info("开始训练...")
        model.train()
        
        # 初始化变量，避免UnboundLocalError
        loss = None
        grad_norm = None
        lr = config.learning_rate
        
        for iter_num in range(start_iter, config.max_iters):
            try:
                # 获取批次数据
                x, y = dataset.get_train_batch(config.batch_size, device)
                
                # 前向传播
                logits = model(x)
                loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 计算裁剪前梯度范数并使用自定义梯度裁剪函数
                grads = [p.grad for p in model.parameters() if getattr(p, "grad", None) is not None]
                if len(grads) > 0:
                    total_sq = torch.zeros((), device=device)
                    for g in grads:
                        total_sq = total_sq + g.detach().float().pow(2).sum()
                    grad_norm = torch.sqrt(total_sq)
                else:
                    grad_norm = torch.tensor(0.0, device=device)

                get_gradient_clipping_fn(model.parameters(), config.grad_clip)
                
                # 更新学习率
                lr = get_lr_cosine_schedule(
                    iter_num,
                    config.learning_rate,
                    config.min_learning_rate,
                    config.warmup_iters,
                    config.max_iters
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # 优化器步进
                optimizer.step()
                
            except Exception as e:
                logger.error(f"训练迭代 {iter_num} 发生错误: {e}")
                # 如果发生错误，跳过这次迭代，但确保变量有默认值
                if loss is None:
                    loss = torch.tensor(float('inf'))
                if grad_norm is None:
                    grad_norm = torch.tensor(0.0)
                continue
            
            # 记录指标
            val_loss = None
            val_perplexity = None
            
            # 评估
            if iter_num % config.eval_every == 0:
                try:
                    eval_metrics = evaluate_model(model, dataset, config.batch_size, device)
                    val_loss = eval_metrics['val_loss']
                    val_perplexity = eval_metrics['val_perplexity']
                    logger.info(f"验证损失: {val_loss:.4f}, 困惑度: {val_perplexity:.2f}")
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
                        save_checkpoint(model, optimizer, iter_num, best_model_path)
                        logger.info(f"保存最佳模型到: {best_model_path}")
                        tracker.log_experiment_note(f"新的最佳验证损失: {val_loss:.4f}")
                except Exception as e:
                    logger.error(f"评估迭代 {iter_num} 发生错误: {e}")
                    val_loss = None
                    val_perplexity = None
            
            # 记录指标到实验跟踪器
            try:
                tracker.log_metrics(
                    iteration=iter_num,
                    train_loss=loss.item() if loss is not None else float('inf'),
                    val_loss=val_loss,
                    val_perplexity=val_perplexity,
                    learning_rate=lr,
                    gradient_norm=grad_norm.item() if grad_norm is not None else 0.0
                )
            except Exception as e:
                logger.error(f"记录指标时发生错误: {e}")
            
            # 日志记录
            if iter_num % config.log_every == 0:
                train_loss = loss.item() if loss is not None else float('inf')
                gn = (grad_norm.item() if grad_norm is not None else 0.0)
                logger.info(f"迭代 {iter_num}/{config.max_iters} - 训练损失: {train_loss:.4f} - 学习率: {lr:.6f} - 梯度范数: {gn:.4f}")
            
            # 保存检查点
            if iter_num % config.save_every == 0:
                try:
                    checkpoint_path = os.path.join(config.checkpoint_dir, f'checkpoint_{iter_num}.pt')
                    save_checkpoint(model, optimizer, iter_num, checkpoint_path)
                    logger.info(f"保存检查点到: {checkpoint_path}")
                    
                    # 更新最新检查点
                    latest_path = os.path.join(config.checkpoint_dir, 'latest.pt')
                    save_checkpoint(model, optimizer, iter_num, latest_path)
                except Exception as e:
                    logger.error(f"保存检查点时发生错误: {e}")
        
        # 保存最终模型
        try:
            final_path = os.path.join(config.checkpoint_dir, 'final_model.pt')
            save_checkpoint(model, optimizer, config.max_iters, final_path)
            logger.info(f"训练完成！最终模型保存到: {final_path}")
        except Exception as e:
            logger.error(f"保存最终模型时发生错误: {e}")
        
        # 记录最终结果
        try:
            final_loss = loss.item() if loss is not None else float('inf')
            tracker.log_experiment_note(f"训练完成，最终训练损失: {final_loss:.4f}")
            if val_loss is not None:
                tracker.log_experiment_note(f"最终验证损失: {val_loss:.4f}")
            if val_perplexity is not None:
                tracker.log_experiment_note(f"最终困惑度: {val_perplexity:.2f}")
        except Exception as e:
            logger.error(f"记录最终结果时发生错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='语言模型训练脚本')
    
    # 配置参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='检查点目录')
    
    # 模型参数
    parser.add_argument('--vocab_size', type=int, help='词汇表大小')
    parser.add_argument('--context_length', type=int, help='上下文长度')
    parser.add_argument('--d_model', type=int, help='模型维度')
    parser.add_argument('--num_layers', type=int, help='层数')
    parser.add_argument('--num_heads', type=int, help='注意力头数')
    parser.add_argument('--d_ff', type=int, help='前馈网络维度')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--max_iters', type=int, help='最大迭代次数')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--warmup_iters', type=int, help='预热迭代次数')
    
    # 其他参数
    parser.add_argument('--device', type=str, help='设备类型 (cpu/cuda)')
    parser.add_argument('--use_wandb', action='store_true', help='启用wandb日志记录')
    parser.add_argument('--wandb_project', type=str, help='wandb项目名称')
    parser.add_argument('--wandb_name', type=str, help='wandb运行名称')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()
    
    # 用命令行参数覆盖配置
    for arg, value in vars(args).items():
        if value is not None and hasattr(config, arg):
            setattr(config, arg, value)
    
    # 开始训练
    train_model(config)


if __name__ == '__main__':
    main()
