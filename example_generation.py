#!/usr/bin/env python3
"""
文本生成示例脚本

演示如何使用训练好的语言模型进行文本生成
"""

import json
import torch
from decode import TextGenerator, load_model_and_vocab


def demo_generation():
    """演示文本生成功能"""
    
    # 示例配置（使用小模型进行演示）
    config = {
        'vocab_size': 1000,  # 小词汇表
        'context_length': 64,
        'd_model': 128,
        'num_layers': 2,
        'num_heads': 4,
        'd_ff': 512,
        'max_seq_len': 128,
        'rope_theta': 10000.0
    }
    
    # 创建示例词汇表
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<sos>': 2,
        '<eos>': 3,
        '<|endoftext|>': 4,
        'hello': 5,
        'world': 6,
        'this': 7,
        'is': 8,
        'a': 9,
        'test': 10,
        'of': 11,
        'the': 12,
        'language': 13,
        'model': 14,
        'generation': 15,
        'capability': 16,
        'artificial': 17,
        'intelligence': 18,
        'machine': 19,
        'learning': 20,
        'deep': 21,
        'neural': 22,
        'network': 23,
        'transformer': 24,
        'attention': 25,
        'mechanism': 26,
        'natural': 27,
        'processing': 28,
        'text': 29,
        'completion': 30,
        'prediction': 31,
        'sequence': 32,
        'token': 33,
        'vocabulary': 34,
        'embedding': 35,
        'layer': 36,
        'activation': 37,
        'function': 38,
        'optimization': 39,
        'gradient': 40,
        'descent': 41,
        'backpropagation': 42,
        'loss': 43,
        'accuracy': 44,
        'training': 45,
        'validation': 46,
        'testing': 47,
        'dataset': 48,
        'batch': 49,
        'epoch': 50,
        'iteration': 51,
        'convergence': 52,
        'overfitting': 53,
        'regularization': 54,
        'dropout': 55,
        'normalization': 56,
        'initialization': 57,
        'hyperparameter': 58,
        'tuning': 59,
        'cross': 60,
        'validation': 61,
        'early': 62,
        'stopping': 63,
        'learning': 64,
        'rate': 65,
        'scheduling': 66,
        'momentum': 67,
        'weight': 68,
        'decay': 69,
        'clipping': 70,
        'sampling': 71,
        'temperature': 72,
        'nucleus': 73,
        'top': 74,
        'beam': 75,
        'search': 76,
        'greedy': 77,
        'random': 78,
        'distribution': 79,
        'probability': 80,
        'logits': 81,
        'softmax': 82,
        'entropy': 83,
        'perplexity': 84,
        'bleu': 85,
        'score': 86,
        'evaluation': 87,
        'metric': 88,
        'performance': 89,
        'benchmark': 90,
        'comparison': 91,
        'analysis': 92,
        'visualization': 93,
        'plot': 94,
        'graph': 95,
        'chart': 96,
        'figure': 97,
        'table': 98,
        'statistics': 99
    }
    
    # 创建模型（未训练的随机权重）
    from cs336_basics.MyLMBlock import MyLMBlock
    
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
    
    # 创建生成器
    generator = TextGenerator(model, vocab, device='cpu')
    
    # 示例提示
    prompts = [
        "hello world",
        "this is a test",
        "machine learning",
        "artificial intelligence",
        "deep neural network"
    ]
    
    print("=== 文本生成演示 ===")
    print("注意：这是使用随机权重模型的演示，生成结果可能没有意义")
    print("在实际使用中，请使用训练好的模型\n")
    
    # 演示不同温度值的效果
    temperatures = [0.5, 1.0, 1.5]
    
    for prompt in prompts:
        print(f"提示: '{prompt}'")
        print("-" * 40)
        
        for temp in temperatures:
            try:
                generated = generator.generate(
                    prompt=prompt,
                    max_tokens=20,
                    temperature=temp,
                    top_p=0.9
                )
                print(f"温度 {temp}: {generated}")
            except Exception as e:
                print(f"温度 {temp}: 生成失败 - {e}")
        
        print()
    
    # 演示不同top-p值的效果
    print("=== Top-p采样演示 ===")
    prompt = "machine learning"
    top_p_values = [0.5, 0.7, 0.9, 1.0]
    
    print(f"提示: '{prompt}'")
    print("-" * 40)
    
    for p in top_p_values:
        try:
            generated = generator.generate(
                prompt=prompt,
                max_tokens=15,
                temperature=1.0,
                top_p=p
            )
            print(f"Top-p {p}: {generated}")
        except Exception as e:
            print(f"Top-p {p}: 生成失败 - {e}")
    
    print()


def demo_with_real_model():
    """使用真实训练模型的演示"""
    print("=== 使用真实模型生成文本 ===")
    
    # 检查是否存在训练好的模型
    model_path = "checkpoints/example/best_model.pt"
    vocab_path = "data/vocabulary.json"
    config_path = "configs/example_config.json"
    
    try:
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载模型和词汇表
        model, vocab = load_model_and_vocab(model_path, vocab_path, config, 'cpu')
        
        # 创建生成器
        generator = TextGenerator(model, vocab, 'cpu')
        
        # 生成文本
        prompts = [
            "hello",
            "machine learning",
            "artificial intelligence",
            "deep learning"
        ]
        
        for prompt in prompts:
            try:
                generated = generator.generate(
                    prompt=prompt,
                    max_tokens=30,
                    temperature=0.8,
                    top_p=0.9
                )
                print(f"提示: '{prompt}'")
                print(f"生成: '{generated}'")
                print("-" * 50)
            except Exception as e:
                print(f"生成失败: {e}")
    
    except FileNotFoundError as e:
        print(f"模型文件未找到: {e}")
        print("请先运行训练脚本生成模型")
    except Exception as e:
        print(f"加载模型失败: {e}")


def interactive_generation():
    """交互式文本生成"""
    print("=== 交互式文本生成 ===")
    print("输入提示文本，按Ctrl+C退出")
    print()
    
    try:
        # 尝试加载真实模型
        model_path = "checkpoints/example/best_model.pt"
        vocab_path = "data/vocabulary.json"
        config_path = "configs/example_config.json"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        model, vocab = load_model_and_vocab(model_path, vocab_path, config, 'cpu')
        generator = TextGenerator(model, vocab, 'cpu')
        print("✓ 已加载训练好的模型")
        
    except:
        # 使用随机模型
        config = {
            'vocab_size': 1000,
            'context_length': 64,
            'd_model': 128,
            'num_layers': 2,
            'num_heads': 4,
            'd_ff': 512,
            'max_seq_len': 128,
            'rope_theta': 10000.0
        }
        
        vocab = {'<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3, '<|endoftext|>': 4}
        from cs336_basics.MyLMBlock import MyLMBlock
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
        generator = TextGenerator(model, vocab, 'cpu')
        print("⚠ 使用随机权重模型（生成结果可能无意义）")
    
    print()
    
    while True:
        try:
            prompt = input("请输入提示: ").strip()
            if not prompt:
                continue
            
            # 获取生成参数
            try:
                max_tokens = int(input("最大生成token数 (默认20): ") or "20")
                temperature = float(input("温度值 (默认1.0): ") or "1.0")
                top_p = float(input("Top-p值 (默认0.9): ") or "0.9")
            except ValueError:
                print("使用默认参数")
                max_tokens, temperature, top_p = 20, 1.0, 0.9
            
            # 生成文本
            generated = generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            
            print(f"生成结果: {generated}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"生成失败: {e}")


def main():
    """主函数"""
    print("语言模型文本生成演示")
    print("=" * 50)
    
    # 演示基本功能
    demo_generation()
    
    # 尝试使用真实模型
    demo_with_real_model()
    
    # 交互式生成
    try:
        interactive_generation()
    except KeyboardInterrupt:
        print("\n程序结束")


if __name__ == '__main__':
    main()
