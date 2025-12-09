#!/usr/bin/env python3
"""
测试 train_bpe 函数，使用 TinyStoriesV2-GPT4-train.txt 数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cs336_basics.train_bpe import train_bpe
import time


def main():
    # 数据文件路径
    data_path = project_root / "data" / "TinyStoriesV2-GPT4-train.txt"
    
    if not data_path.exists():
        print(f"错误: 数据文件不存在: {data_path}")
        sys.exit(1)
    
    # 设置参数
    vocab_size = 10000  # 可以根据需要调整
    special_tokens = ["<|endoftext|>"]
    
    print(f"[INFO] 开始训练 BPE")
    print(f"[INFO] 数据文件: {data_path}")
    print(f"[INFO] 词汇表大小: {vocab_size}")
    print(f"[INFO] 特殊标记: {special_tokens}")
    print("-" * 60)
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练 BPE
    vocab, merges = train_bpe(
        input_path=str(data_path),
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 打印结果统计
    print("-" * 60)
    print(f"[INFO] 训练完成!")
    print(f"[INFO] 耗时: {elapsed_time:.2f} 秒")
    print(f"[INFO] 词汇表大小: {len(vocab)}")
    print(f"[INFO] 合并操作数: {len(merges)}")
    print(f"[INFO] 前 10 个合并操作:")
    for i, (a, b) in enumerate(merges[:10], 1):
        try:
            a_str = a.decode("utf-8", errors="replace")
            b_str = b.decode("utf-8", errors="replace")
            print(f"  {i}. {repr(a_str)} + {repr(b_str)}")
        except:
            print(f"  {i}. {a} + {b}")
    
    # 验证结果
    assert len(vocab) == vocab_size, f"词汇表大小不匹配: 期望 {vocab_size}, 实际 {len(vocab)}"
    assert len(merges) == vocab_size - 256 - len(special_tokens), \
        f"合并操作数不匹配: 期望 {vocab_size - 256 - len(special_tokens)}, 实际 {len(merges)}"
    
    print("\n[INFO] 所有验证通过!")
    return vocab, merges


if __name__ == "__main__":
    main()

