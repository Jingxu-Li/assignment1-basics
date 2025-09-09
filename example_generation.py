#!/usr/bin/env python3
"""
文本生成示例脚本

演示如何使用训练好的语言模型进行文本生成
"""

import json
import torch
import os
from decode import TextGenerator, load_model_and_vocab
from cs336_basics.utils import load_checkpoint
from cs336_basics.MyLMBlock import MyLMBlock
from cs336_basics.bpe_tokenizer import MyBpeTokenizer


def load_trained_model():
    """加载训练好的模型"""
    print("=== 加载训练好的模型 ===")
    
    # 模型路径
    model_path = "checkpoints/best_model.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在")
        return None, None, None
    
    # 模型配置（与训练时使用的配置一致）
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
    
    # 加载BPE tokenizer
    vocab_path = "vocab.json"
    merges_path = "merges.txt"
    
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        try:
            tokenizer = MyBpeTokenizer.from_files(
                vocab_path, 
                merges_path, 
                special_tokens=["<|endoftext|>"]
            )
            print(f"✓ 成功加载BPE tokenizer，包含 {len(tokenizer.vocab)} 个token")
        except Exception as e:
            print(f"加载BPE tokenizer失败: {e}")
            return None, None, None
    else:
        print(f"警告：tokenizer文件不存在，vocab.json: {os.path.exists(vocab_path)}, merges.txt: {os.path.exists(merges_path)}")
        return None, None, None
    
    try:
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
        optimizer = torch.optim.Adam(model.parameters())  # 虚拟优化器
        start_iter = load_checkpoint(model_path, model, optimizer)
        print(f"✓ 成功加载模型，训练迭代次数: {start_iter}")
        
        return model, tokenizer, config
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None, None, None


def generate_with_trained_model():
    """使用训练好的模型进行文本生成"""
    print("=== 使用训练好的模型生成文本 ===")
    
    # 加载模型
    model, tokenizer, config = load_trained_model()
    if model is None:
        print("无法加载模型，退出")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建生成器
    generator = TextGenerator(model, tokenizer, device)
    
    # 示例提示
    prompts = [
        "hello world",
        "machine learning",
        "artificial intelligence",
        "deep learning",
        "neural network",
        "transformer model",
        "attention mechanism",
        "natural language processing"
    ]
    
    print("\n开始生成文本...")
    print("=" * 60)
    
    # 生成参数
    max_tokens = 50
    temperature = 0.8
    top_p = 0.9
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. 提示: '{prompt}'")
        print("-" * 40)
        
        try:
            generated = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            print(f"生成结果: {generated}")
        except Exception as e:
            print(f"生成失败: {e}")
    
    print("\n" + "=" * 60)


def interactive_generation_with_trained_model():
    """使用训练好的模型进行交互式文本生成"""
    print("=== 交互式文本生成（使用训练好的模型）===")
    
    # 加载模型
    model, tokenizer, config = load_trained_model()
    if model is None:
        print("无法加载模型，退出")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建生成器
    generator = TextGenerator(model, tokenizer, device)
    
    print("\n开始交互式生成...")
    print("输入提示文本，按Ctrl+C退出")
    print("生成参数：")
    print("- 最大token数: 50")
    print("- 温度: 0.8")
    print("- Top-p: 0.9")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\n请输入提示: ").strip()
            if not prompt:
                continue
            
            print("生成中...")
            generated = generator.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=0.8,
                top_p=0.9
            )
            
            print(f"生成结果: {generated}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"生成失败: {e}")


def load_pretrained_model():
    """加载预训练模型（示例）"""
    print("=== 加载预训练模型 ===")
    
    # 模型配置（GPT-2 小模型配置）
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
    
    # 加载BPE tokenizer
    vocab_path = "vocab.json"
    merges_path = "merges.txt"
    
    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        tokenizer = MyBpeTokenizer.from_files(
            vocab_path, 
            merges_path, 
            special_tokens=["<|endoftext|>"]
        )
        print(f"✓ 成功加载BPE tokenizer，包含 {len(tokenizer.vocab)} 个token")
    else:
        print("警告：使用默认tokenizer配置")
        # 创建默认词汇表
        vocab = {}
        for i in range(config['vocab_size']):
            vocab[f'token_{i}'] = i
        
        # 添加特殊token
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>', '<|endoftext|>']
        for i, token in enumerate(special_tokens):
            vocab[token] = config['vocab_size'] - len(special_tokens) + i
        
        # 创建简单的tokenizer（这里需要适配，暂时跳过）
        print("无法创建默认tokenizer，跳过预训练模型示例")
        return None, None, None
    
    return model, tokenizer, config


def generate_with_pretrained_model():
    """使用预训练模型进行文本生成"""
    print("=== 使用预训练模型生成文本 ===")
    
    # 加载模型
    model, tokenizer, config = load_pretrained_model()
    if model is None:
        print("无法加载预训练模型，跳过此示例")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建生成器
    generator = TextGenerator(model, tokenizer, device)
    
    # 示例提示
    prompts = [
        "The future of artificial intelligence",
        "Machine learning applications",
        "Deep learning breakthroughs",
        "Natural language processing advances"
    ]
    
    print("\n开始生成文本...")
    print("=" * 60)
    
    # 生成参数
    max_tokens = 100
    temperature = 0.7
    top_p = 0.9
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. 提示: '{prompt}'")
        print("-" * 40)
        
        try:
            generated = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            print(f"生成结果: {generated}")
        except Exception as e:
            print(f"生成失败: {e}")
    
    print("\n" + "=" * 60)


def interactive_generation_with_pretrained_model():
    """使用预训练模型进行交互式文本生成"""
    print("=== 交互式文本生成（使用预训练模型）===")
    
    # 加载模型
    model, tokenizer, config = load_pretrained_model()
    if model is None:
        print("无法加载预训练模型，跳过此示例")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建生成器
    generator = TextGenerator(model, tokenizer, device)
    
    print("\n开始交互式生成...")
    print("输入提示文本，按Ctrl+C退出")
    print("生成参数：")
    print("- 最大token数: 100")
    print("- 温度: 0.7")
    print("- Top-p: 0.9")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\n请输入提示: ").strip()
            if not prompt:
                continue
            
            print("生成中...")
            generated = generator.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.7,
                top_p=0.9
            )
            
            print(f"生成结果: {generated}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"生成失败: {e}")


def compare_generation_methods():
    """比较不同的生成方法"""
    print("=== 比较不同的生成方法 ===")
    
    # 加载模型
    model, tokenizer, config = load_trained_model()
    if model is None:
        print("无法加载模型，跳过此示例")
        return
    
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建生成器
    generator = TextGenerator(model, tokenizer, device)
    
    # 测试提示
    prompt = "The future of AI"
    
    # 不同的生成参数
    generation_configs = [
        {"name": "确定性生成", "temperature": 0.1, "top_p": 1.0},
        {"name": "创造性生成", "temperature": 1.0, "top_p": 1.0},
        {"name": "平衡生成", "temperature": 0.7, "top_p": 0.9},
        {"name": "保守生成", "temperature": 0.5, "top_p": 0.8},
    ]
    
    print(f"\n测试提示: '{prompt}'")
    print("=" * 60)
    
    for config in generation_configs:
        print(f"\n{config['name']}:")
        print(f"温度: {config['temperature']}, Top-p: {config['top_p']}")
        print("-" * 40)
        
        try:
            generated = generator.generate(
                prompt=prompt,
                max_tokens=50,
                temperature=config['temperature'],
                top_p=config['top_p']
            )
            print(f"生成结果: {generated}")
        except Exception as e:
            print(f"生成失败: {e}")
    
    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("语言模型文本生成示例")
    print("=" * 60)
    
    # 检查是否有训练好的模型
    if os.path.exists("checkpoints/best_model.pt"):
        print("发现训练好的模型，运行相关示例...")
        
        # 使用训练好的模型生成文本
        generate_with_trained_model()
        
        # 交互式生成
        try:
            interactive_generation_with_trained_model()
        except KeyboardInterrupt:
            print("跳过交互式生成")
        
        # 比较生成方法
        compare_generation_methods()
    
    else:
        print("未发现训练好的模型，运行预训练模型示例...")
        
        # 使用预训练模型生成文本
        generate_with_pretrained_model()
        
        # 交互式生成
        try:
            interactive_generation_with_pretrained_model()
        except KeyboardInterrupt:
            print("跳过交互式生成")


if __name__ == '__main__':
    main()
