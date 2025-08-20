#!/usr/bin/env python3
"""
语言模型解码/生成函数

功能特性：
1. 根据提示生成文本补全
2. 控制最大生成token数
3. 温度缩放控制随机性
4. Top-p采样（核采样）
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Union
import json
import argparse
from pathlib import Path

# 导入自定义模块
from cs336_basics.MyLMBlock import MyLMBlock
from cs336_basics.utils import load_checkpoint


class TextGenerator:
    """文本生成器类"""
    
    def __init__(self, model: MyLMBlock, vocab: dict, device: str = 'cpu'):
        """
        初始化文本生成器
        
        Args:
            model: 训练好的语言模型
            vocab: 词汇表字典 {token: id}
            device: 设备类型 ('cpu' 或 'cuda')
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # 创建反向词汇表 {id: token}
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.vocab = vocab
        
        # 特殊token
        self.endoftext_token = '<|endoftext|>'
        self.endoftext_id = vocab.get(self.endoftext_token, None)
        
        if self.endoftext_id is None:
            print(f"警告: 词汇表中未找到 {self.endoftext_token} token")
    
    def tokenize(self, text: str) -> List[int]:
        """
        简单的tokenization（基于空格分割）
        在实际应用中，您可能需要使用更复杂的tokenizer
        
        Args:
            text: 输入文本
            
        Returns:
            token IDs列表
        """
        words = text.lower().split()
        tokens = []
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # 对于未知词，使用<unk> token
                unk_id = self.vocab.get('<unk>', 1)
                tokens.append(unk_id)
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """
        将token IDs转换回文本
        
        Args:
            token_ids: token ID列表
            
        Returns:
            生成的文本
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                # 跳过特殊token（除了endoftext）
                if token not in ['<pad>', '<unk>', '<sos>', '<eos>']:
                    tokens.append(token)
        
        return ' '.join(tokens)
    
    def apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        应用温度缩放
        
        Args:
            logits: 模型输出的logits
            temperature: 温度值 (0.1-2.0)
            
        Returns:
            缩放后的logits
        """
        if temperature <= 0:
            raise ValueError("温度值必须大于0")
        
        # 应用温度缩放
        scaled_logits = logits / temperature
        return scaled_logits
    
    def top_p_sampling(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Top-p采样（核采样）
        
        Args:
            logits: 模型输出的logits
            p: 累积概率阈值 (0.0-1.0)
            
        Returns:
            过滤后的logits
        """
        if p <= 0 or p > 1:
            raise ValueError("p值必须在(0, 1]范围内")
        
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 按概率降序排序
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过p的位置
        sorted_indices_to_remove = cumulative_probs > p
        
        # 将第一个超过p的位置设为True（保留该token）
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 创建mask
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        
        # 将不需要的logits设为负无穷
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float('-inf')
        
        return filtered_logits
    
    def sample_next_token(self, logits: torch.Tensor, temperature: float = 1.0, 
                         top_p: float = 1.0) -> int:
        """
        采样下一个token
        
        Args:
            logits: 模型输出的logits
            temperature: 温度值
            top_p: top-p采样阈值
            
        Returns:
            采样的token ID
        """
        # 应用温度缩放
        if temperature != 1.0:
            logits = self.apply_temperature(logits, temperature)
        
        # 应用top-p采样
        if top_p < 1.0:
            logits = self.top_p_sampling(logits, top_p)
        
        # 计算概率分布
        probs = F.softmax(logits, dim=-1)
        
        # 采样
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token.item()
    
    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0,
                top_p: float = 1.0, stop_token: Optional[str] = None) -> str:
        """
        生成文本补全
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度值
            top_p: top-p采样阈值
            stop_token: 停止token（如果为None，使用endoftext）
            
        Returns:
            生成的文本
        """
        # 设置停止token
        if stop_token is None:
            stop_token = self.endoftext_token
        
        stop_token_id = self.vocab.get(stop_token, None)
        
        # Tokenize提示
        input_tokens = self.tokenize(prompt)
        if not input_tokens:
            raise ValueError("提示文本无法tokenize")
        
        # 转换为tensor
        input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_tokens):
                # 前向传播
                logits = self.model(input_tensor)
                
                # 获取最后一个位置的logits
                next_token_logits = logits[0, -1, :]
                
                # 采样下一个token
                next_token = self.sample_next_token(
                    next_token_logits, temperature, top_p
                )
                
                # 检查是否遇到停止token
                if next_token == stop_token_id:
                    break
                
                generated_tokens.append(next_token)
                
                # 更新输入tensor
                input_tensor = torch.cat([
                    input_tensor, 
                    torch.tensor([[next_token]], dtype=torch.long, device=self.device)
                ], dim=1)
        
        # 将生成的tokens转换为文本
        generated_text = self.detokenize(generated_tokens)
        
        return generated_text
    
    def generate_with_context(self, prompt: str, max_tokens: int = 100, 
                            temperature: float = 1.0, top_p: float = 1.0) -> str:
        """
        生成文本（包含原始提示）
        
        Args:
            prompt: 输入提示
            max_tokens: 最大生成token数
            temperature: 温度值
            top_p: top-p采样阈值
            
        Returns:
            完整的生成文本（提示+补全）
        """
        completion = self.generate(prompt, max_tokens, temperature, top_p)
        return prompt + " " + completion


def load_model_and_vocab(model_path: str, vocab_path: str, 
                        config: dict, device: str = 'cpu') -> tuple[MyLMBlock, dict]:
    """
    加载模型和词汇表
    
    Args:
        model_path: 模型检查点路径
        vocab_path: 词汇表路径
        config: 模型配置
        
    Returns:
        (model, vocab) 元组
    """
    # 加载词汇表
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    # 创建模型
    model = MyLMBlock(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        rope_theta=config['rope_theta'],
        in_indices=None
    )
    
    # 加载模型权重
    if model_path:
        # 创建虚拟优化器（用于加载检查点）
        optimizer = torch.optim.Adam(model.parameters())
        load_checkpoint(model_path, model, optimizer)
        print(f"模型已从 {model_path} 加载")
    
    return model, vocab


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='语言模型文本生成')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, help='模型检查点路径')
    parser.add_argument('--vocab_path', type=str, default='data/vocabulary.json', 
                       help='词汇表路径')
    parser.add_argument('--config_path', type=str, help='模型配置文件路径')
    
    # 生成参数
    parser.add_argument('--prompt', type=str, required=True, help='输入提示')
    parser.add_argument('--max_tokens', type=int, default=100, help='最大生成token数')
    parser.add_argument('--temperature', type=float, default=1.0, help='温度值')
    parser.add_argument('--top_p', type=float, default=1.0, help='top-p采样阈值')
    parser.add_argument('--device', type=str, default='cpu', help='设备类型')
    
    # 输出参数
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--include_prompt', action='store_true', 
                       help='输出包含原始提示')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        # 默认配置
        config = {
            'vocab_size': 50257,
            'context_length': 1024,
            'd_model': 768,
            'num_layers': 12,
            'num_heads': 12,
            'd_ff': 3072,
            'max_seq_len': 2048,
            'rope_theta': 10000.0
        }
    
    # 加载模型和词汇表
    try:
        model, vocab = load_model_and_vocab(
            args.model_path, args.vocab_path, config, args.device
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 创建生成器
    generator = TextGenerator(model, vocab, args.device)
    
    # 生成文本
    try:
        if args.include_prompt:
            generated_text = generator.generate_with_context(
                args.prompt, args.max_tokens, args.temperature, args.top_p
            )
        else:
            generated_text = generator.generate(
                args.prompt, args.max_tokens, args.temperature, args.top_p
            )
        
        # 输出结果
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            print(f"生成结果已保存到: {args.output_file}")
        else:
            print("生成结果:")
            print("-" * 50)
            print(generated_text)
            print("-" * 50)
    
    except Exception as e:
        print(f"生成失败: {e}")


if __name__ == '__main__':
    main()
