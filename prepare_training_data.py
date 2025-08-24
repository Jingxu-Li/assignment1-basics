#!/usr/bin/env python3
"""
准备训练数据脚本 - 将data文件夹下的txt文件转换为train.py需要的格式

这个脚本会：
1. 读取data文件夹下的txt文件
2. 使用现有的BPE tokenizer进行tokenization
3. 生成train.py需要的二进制数据文件
"""

import argparse
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import re

# 导入自定义BPE tokenizer
from cs336_basics.bpe_tokenizer import MyBpeTokenizer


def load_bpe_tokenizer(vocab_path: str, merges_path: str, special_tokens: List[str] = None) -> MyBpeTokenizer:
    """加载BPE tokenizer"""
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    print(f"加载BPE tokenizer...")
    print(f"词汇表文件: {vocab_path}")
    print(f"合并规则文件: {merges_path}")

    tokenizer = MyBpeTokenizer.from_files(
        vocab_path, merges_path, special_tokens)
    print(f"BPE tokenizer加载完成，词汇表大小: {len(tokenizer.vocab)}")

    return tokenizer


def process_txt_file(file_path: str, tokenizer: MyBpeTokenizer) -> List[int]:
    """处理单个txt文件"""
    all_tokens = []

    print(f"处理文件: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            if line:
                try:
                    # 使用BPE tokenizer进行tokenization
                    tokens = tokenizer.encode(line)
                    all_tokens.extend(tokens)

                except Exception as e:
                    print(f"警告: 处理第{line_num}行时出错: {e}")
                    continue

    print(f"文件 {file_path} 处理完成，生成了 {len(all_tokens)} 个tokens")
    return all_tokens


def main():
    parser = argparse.ArgumentParser(description='准备训练数据')
    parser.add_argument('--input_dir', type=str,
                        default='data', help='输入txt文件目录')
    parser.add_argument('--input_file', type=str,
                        default='data/valid.txt', help='指定单个输入txt文件路径')
    parser.add_argument('--output_file', type=str,
                        default='data/tokenized_data.bin', help='输出二进制文件路径')
    parser.add_argument('--vocab_path', type=str,
                        default='vocab.json', help='BPE词汇表文件路径')
    parser.add_argument('--merges_path', type=str,
                        default='merges.txt', help='BPE合并规则文件路径')
    parser.add_argument('--special_tokens', type=str, nargs='*',
                        default=['<|endoftext|>'], help='特殊token列表')
    parser.add_argument('--max_files', type=int,
                        default=None, help='最大处理文件数（用于调试）')

    args = parser.parse_args()

    # 检查必要文件是否存在
    if not os.path.exists(args.vocab_path):
        print(f"错误: 词汇表文件不存在: {args.vocab_path}")
        return

    if not os.path.exists(args.merges_path):
        print(f"错误: 合并规则文件不存在: {args.merges_path}")
        return

    # 创建输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 加载BPE tokenizer
    tokenizer = load_bpe_tokenizer(
        args.vocab_path, args.merges_path, args.special_tokens)

    # 收集输入文件
    input_files = []

    if args.input_file:
        # 如果指定了单个文件
        input_file_path = Path(args.input_file)
        if not input_file_path.exists():
            print(f"错误: 指定的文件不存在: {args.input_file}")
            return
        if not input_file_path.suffix.lower() == '.txt':
            print(f"警告: 文件不是txt格式: {args.input_file}")
        input_files = [input_file_path]
        print(f"使用指定的文件: {args.input_file}")
    else:
        # 从目录中收集所有txt文件
        for ext in ['*.txt']:
            input_files.extend(Path(args.input_dir).glob(ext))

        if not input_files:
            print(f"在 {args.input_dir} 中没有找到txt文件")
            return

        # 限制文件数量（用于调试）
        if args.max_files:
            input_files = input_files[:args.max_files]

    print(f"找到 {len(input_files)} 个txt文件")

    # 处理所有文件
    all_tokens = []
    processed_files = 0

    for file_path in input_files:
        try:
            tokens = process_txt_file(str(file_path), tokenizer)
            all_tokens.extend(tokens)
            processed_files += 1

            # 显示进度
            if processed_files % 5 == 0:
                print(
                    f"已处理 {processed_files}/{len(input_files)} 个文件，当前总tokens: {len(all_tokens)}")

        except Exception as e:
            print(f"错误: 处理文件 {file_path} 时出错: {e}")
            continue

    print(f"总共处理了 {processed_files} 个文件")
    print(f"总共生成了 {len(all_tokens)} 个tokens")

    if len(all_tokens) == 0:
        print("错误: 没有生成任何tokens")
        return

    # 保存为二进制文件
    print(f"保存数据到: {args.output_file}")
    data_array = np.array(all_tokens, dtype=np.uint16)
    data_array.tofile(args.output_file)

    # 保存数据统计信息
    stats = {
        'vocab_size': len(tokenizer.vocab),
        'total_tokens': len(all_tokens),
        'processed_files': processed_files,
        'special_tokens': args.special_tokens,
        'vocab_path': args.vocab_path,
        'merges_path': args.merges_path,
        'output_file': args.output_file
    }

    stats_path = args.output_file.replace('.bin', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"数据统计信息已保存到: {stats_path}")
    print("数据准备完成！")

    # 显示一些示例
    print("\n示例tokenization:")
    if len(all_tokens) > 0:
        sample_tokens = all_tokens[:50]
        sample_text = tokenizer.decode(sample_tokens)
        print(f"前50个tokens: {sample_tokens}")
        print(f"解码结果: {sample_text[:200]}...")


if __name__ == '__main__':
    main()
