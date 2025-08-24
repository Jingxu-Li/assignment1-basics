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
    
    # 加载词汇表
    vocab_path = "vocab.json"
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"词汇表文件 {vocab_path} 不存在")
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # 转换词汇表格式
    vocab = {token: int(token_id) for token_id, token in vocab_data.items()}
    id_to_token = {int(token_id): token for token_id, token in vocab_data.items()}
    
    print(f"✓ 成功加载词汇表，包含 {len(vocab)} 个token")
    
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
    
    return model, vocab, id_to_token, config


def simple_tokenize(text, vocab):
    """简单的tokenization"""
    # 这里使用简单的字符级tokenization
    # 在实际应用中，您可能需要使用更复杂的tokenizer
    tokens = []
    for char in text:
        if char in vocab:
            tokens.append(vocab[char])
        else:
            # 对于未知字符，使用空格token
            tokens.append(vocab.get(' ', vocab.get('0', 0)))
    return tokens


def simple_detokenize(token_ids, id_to_token):
    """简单的detokenization"""
    text = ""
    for token_id in token_ids:
        if token_id in id_to_token:
            token = id_to_token[token_id]
            # 跳过控制字符
            if ord(token) >= 32 or token in ['\n', '\t']:
                text += token
    return text


def generate_text(model, vocab, id_to_token, prompt, max_tokens=50, temperature=0.8, device='cpu'):
    """生成文本"""
    model.eval()
    model.to(device)
    
    # Tokenize prompt
    input_tokens = simple_tokenize(prompt, vocab)
    if len(input_tokens) == 0:
        input_tokens = [vocab.get(' ', 0)]
    
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
            if next_token in [vocab.get('<|endoftext|>', -1), vocab.get('<eos>', -1)]:
                break
    
    # Detokenize
    generated_text = simple_detokenize(generated_tokens, id_to_token)
    return generated_text


def main():
    """主函数"""
    print("=== 语言模型推理 ===")
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 加载模型和词汇表
        model, vocab, id_to_token, config = load_model_and_vocab()
        
        # 示例提示
        prompts = [
            "hello",
            "machine learning",
            "artificial intelligence",
            "deep learning",
            "neural network"
        ]
        
        print("\n开始生成文本...")
        print("=" * 60)
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{i}. 提示: '{prompt}'")
            print("-" * 40)
            
            try:
                generated = generate_text(
                    model, vocab, id_to_token, prompt,
                    max_tokens=30,
                    temperature=0.8,
                    device=device
                )
                print(f"生成结果: {generated}")
            except Exception as e:
                print(f"生成失败: {e}")
        
        print("\n" + "=" * 60)
        
        # 交互式生成
        print("\n=== 交互式生成 ===")
        print("输入提示文本，按Ctrl+C退出")
        
        while True:
            try:
                prompt = input("\n请输入提示: ").strip()
                if not prompt:
                    continue
                
                print("生成中...")
                generated = generate_text(
                    model, vocab, id_to_token, prompt,
                    max_tokens=50,
                    temperature=0.8,
                    device=device
                )
                
                print(f"生成结果: {generated}")
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n退出交互模式")
                break
            except Exception as e:
                print(f"生成失败: {e}")
        
    except Exception as e:
        print(f"错误: {e}")


if __name__ == '__main__':
    main()
