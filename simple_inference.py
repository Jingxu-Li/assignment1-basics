#!/usr/bin/env python3
"""
简化的推理脚本 - 使用训练好的模型进行文本生成
"""

import json
import torch
import os
import numpy as np
from cs336_basics.utils import load_checkpoint, cross_entropy
from cs336_basics.MyLMBlock import MyLMBlock
from cs336_basics.bpe_tokenizer import MyBpeTokenizer


def load_model_and_vocab():
    """加载模型和词汇表"""
    model_path = "checkpoints/best_model.pt"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    # 模型配置（与您修改后的配置一致）
    config = {
        'vocab_size': 10000,
        'context_length': 256,
        'd_model': 512,
        'num_layers': 4,
        'num_heads': 16,
        'd_ff': 1344,
        'max_seq_len': 2048,
        'rope_theta': 10000.0
    }
    
    # 加载词汇表和merges文件
    vocab_path = "vocab.json"
    merges_path = "merges.txt"
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件 {vocab_path} 不存在")
    
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"merges文件 {merges_path} 不存在")
    
    # 使用 MyBpeTokenizer 加载词汇表和merges
    tokenizer = MyBpeTokenizer.from_files(
        vocab_path, 
        merges_path, 
        special_tokens=["<|endoftext|>"]
    )
    
    print(f"✓ 成功加载BPE tokenizer，包含 {len(tokenizer.vocab)} 个token")
    
    # 创建模型
    model = MyLMBlock(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta'],
        in_indices=None
    )
    
    # 加载模型权重
    try:
        optimizer = torch.optim.Adam(model.parameters())
        start_iter = load_checkpoint(model_path, model, optimizer)
        print(f"✓ 成功加载模型，训练迭代次数: {start_iter}")
    except Exception as e:
        print(f"加载优化器状态失败，但模型权重已加载: {e}")
        # 只加载模型权重
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_iter = checkpoint.get('iteration', 0)
        print(f"✓ 成功加载模型权重，训练迭代次数: {start_iter}")
    
    return model, tokenizer, config


def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.8, device='cpu'):
    """生成文本"""
    model.eval()
    model.to(device)
    
    # 使用 MyBpeTokenizer 进行 tokenization
    input_tokens = tokenizer.encode(prompt)
    if len(input_tokens) == 0:
        # 如果编码结果为空，使用空格token
        input_tokens = [tokenizer.token_to_id.get(b' ', 0)]
    
    # 转换为tensor
    input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)
    
    generated_tokens = input_tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 确保输入长度不超过context_length
            if len(input_tensor[0]) > model.context_length:
                input_tensor = input_tensor[:, -model.context_length:]
            
            # 前向传播
            logits = model(input_tensor)
            
            # 获取最后一个token的logits
            next_token_logits = logits[0, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # 添加到生成序列
            generated_tokens.append(next_token)
            
            # 更新输入tensor
            input_tensor = torch.tensor([generated_tokens], dtype=torch.long, device=device)
            
            # 检查是否生成了结束token
            endoftext_id = tokenizer.token_to_id.get("<|endoftext|>".encode('utf-8'), -1)
            if next_token == endoftext_id:
                break
    
    # 使用 MyBpeTokenizer 进行 detokenization
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def main():
    """主函数"""
    print("=== 语言模型推理 ===")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 加载模型和tokenizer
        model, tokenizer, config = load_model_and_vocab()
        
        # 示例提示
        prompts = [
            "Once upon a time, there was a boy named Ding. "
        ]
        
        print("\n开始生成文本...")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. 提示: '{prompt}'")
            print("-" * 40)
            
            try:
                generated = generate_text(
                    model, tokenizer, prompt,
                    max_tokens=256,
                    temperature=0.1,
                    device=device
                )
                print(f"生成结果: {generated}")
            except Exception as e:
                print(f"生成失败: {e}")
        
        print("\n" + "=" * 60)
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()
