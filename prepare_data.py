#!/usr/bin/env python3
"""
数据准备脚本 - 将文本数据转换为训练所需的格式

功能：
1. 文本预处理
2. 简单的tokenization（基于空格分割）
3. 构建词汇表
4. 保存为二进制格式
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Set
import re


def preprocess_text(text: str) -> str:
    """简单的文本预处理"""
    # 转换为小写
    text = text.lower()
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留基本的标点符号）
    text = re.sub(r'[^\w\s.,!?;:()"\'-]', '', text)
    
    return text.strip()


def build_vocabulary(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """构建词汇表"""
    word_freq = {}
    
    # 统计词频
    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 过滤低频词
    filtered_words = {word: freq for word, freq in word_freq.items() if freq >= min_freq}
    
    # 构建词汇表（添加特殊token）
    vocab = {
        '<pad>': 0,      # 填充token
        '<unk>': 1,      # 未知token
        '<sos>': 2,      # 句子开始
        '<eos>': 3,      # 句子结束
    }
    
    # 按频率排序添加词汇
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    for word, _ in sorted_words:
        if word not in vocab:
            vocab[word] = len(vocab)
    
    return vocab


def tokenize_text(text: str, vocab: Dict[str, int]) -> List[int]:
    """将文本转换为token IDs"""
    words = text.split()
    tokens = [vocab['<sos>']]  # 句子开始
    
    for word in words:
        if word in vocab:
            tokens.append(vocab[word])
        else:
            tokens.append(vocab['<unk>'])  # 未知词
    
    tokens.append(vocab['<eos>'])  # 句子结束
    return tokens


def save_vocabulary(vocab: Dict[str, int], output_path: str):
    """保存词汇表"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # 创建反向映射
    id_to_word = {v: k for k, v in vocab.items()}
    reverse_path = output_path.replace('.json', '_reverse.json')
    with open(reverse_path, 'w', encoding='utf-8') as f:
        json.dump(id_to_word, f, ensure_ascii=False, indent=2)


def process_file(input_path: str, vocab: Dict[str, int]) -> List[int]:
    """处理单个文件"""
    all_tokens = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # 预处理文本
                processed_text = preprocess_text(line)
                if processed_text:
                    # tokenize
                    tokens = tokenize_text(processed_text, vocab)
                    all_tokens.extend(tokens)
    
    return all_tokens


def main():
    parser = argparse.ArgumentParser(description='准备训练数据')
    parser.add_argument('--input_dir', type=str, required=True, help='输入文本文件目录')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')
    parser.add_argument('--vocab_size', type=int, default=10000, help='词汇表大小限制')
    parser.add_argument('--min_freq', type=int, default=2, help='最小词频')
    parser.add_argument('--test_split', type=float, default=0.1, help='测试集比例')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 收集所有文本文件
    input_files = []
    for ext in ['*.txt', '*.json', '*.jsonl']:
        input_files.extend(Path(args.input_dir).glob(ext))
    
    if not input_files:
        print(f"在 {args.input_dir} 中没有找到文本文件")
        return
    
    print(f"找到 {len(input_files)} 个文本文件")
    
    # 读取和预处理所有文本
    all_texts = []
    for file_path in input_files:
        print(f"处理文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    processed_text = preprocess_text(line)
                    if processed_text:
                        all_texts.append(processed_text)
    
    print(f"总共处理了 {len(all_texts)} 行文本")
    
    # 构建词汇表
    print("构建词汇表...")
    vocab = build_vocabulary(all_texts, args.min_freq)
    
    # 限制词汇表大小
    if len(vocab) > args.vocab_size:
        # 保留特殊token和最高频的词
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        word_freq = {}
        
        for text in all_texts:
            words = text.split()
            for word in words:
                if word not in special_tokens:
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # 按频率排序
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # 重建词汇表
        vocab = {token: i for i, token in enumerate(special_tokens)}
        for word, _ in sorted_words[:args.vocab_size - len(special_tokens)]:
            vocab[word] = len(vocab)
    
    print(f"词汇表大小: {len(vocab)}")
    
    # 保存词汇表
    vocab_path = os.path.join(args.output_dir, 'vocabulary.json')
    save_vocabulary(vocab, vocab_path)
    print(f"词汇表已保存到: {vocab_path}")
    
    # 转换为token IDs
    print("转换为token IDs...")
    all_tokens = []
    for text in all_texts:
        tokens = tokenize_text(text, vocab)
        all_tokens.extend(tokens)
    
    print(f"总共生成了 {len(all_tokens)} 个tokens")
    
    # 分割训练集和测试集
    split_idx = int(len(all_tokens) * (1 - args.test_split))
    train_tokens = all_tokens[:split_idx]
    test_tokens = all_tokens[split_idx:]
    
    print(f"训练集: {len(train_tokens)} tokens")
    print(f"测试集: {len(test_tokens)} tokens")
    
    # 保存为二进制文件
    train_path = os.path.join(args.output_dir, 'train_data.bin')
    test_path = os.path.join(args.output_dir, 'test_data.bin')
    
    train_array = np.array(train_tokens, dtype=np.int32)
    test_array = np.array(test_tokens, dtype=np.int32)
    
    train_array.tofile(train_path)
    test_array.tofile(test_path)
    
    print(f"训练数据已保存到: {train_path}")
    print(f"测试数据已保存到: {test_path}")
    
    # 保存数据统计信息
    stats = {
        'vocab_size': len(vocab),
        'train_tokens': len(train_tokens),
        'test_tokens': len(test_tokens),
        'total_tokens': len(all_tokens),
        'special_tokens': ['<pad>', '<unk>', '<sos>', '<eos>']
    }
    
    stats_path = os.path.join(args.output_dir, 'data_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"数据统计信息已保存到: {stats_path}")
    print("数据准备完成！")


if __name__ == '__main__':
    main()
