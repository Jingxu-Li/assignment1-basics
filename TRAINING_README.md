# 训练脚本使用说明

这个训练脚本实现了语言模型的完整训练循环，满足所有要求：

## 功能特性

✅ **超参数配置和控制** - 支持配置文件、命令行参数和默认值  
✅ **内存高效数据加载** - 使用 `np.memmap` 处理大型数据集  
✅ **检查点序列化** - 自动保存和恢复训练状态  
✅ **定期日志记录** - 支持控制台输出和 Weights & Biases 集成  

## 快速开始

### 1. 基本使用

```bash
# 使用默认配置训练
python train.py

# 使用配置文件训练
python train.py --config configs/default_config.json

# 指定数据路径
python train.py --data_path /path/to/your/data.bin
```

### 2. 自定义参数

```bash
# 修改模型参数
python train.py \
    --vocab_size 50257 \
    --context_length 1024 \
    --d_model 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 32 \
    --max_iters 10000 \
    --learning_rate 3e-4
```

### 3. 启用 Weights & Biases 日志记录

```bash
python train.py \
    --use_wandb \
    --wandb_project "my-language-model" \
    --wandb_name "experiment-1"
```

## 数据格式

训练脚本期望的数据格式是：
- 文件类型：二进制文件（`.bin`）
- 数据类型：`np.int32`
- 内容：tokenized 文本数据，每个 token 用整数表示

示例数据准备：
```python
import numpy as np

# 假设您有 tokenized 数据
tokens = [1, 2, 3, 4, 5, ...]  # token IDs
tokens_array = np.array(tokens, dtype=np.int32)

# 保存为二进制文件
tokens_array.tofile('data/tokenized_data.bin')
```

## 配置参数说明

### 模型参数
- `vocab_size`: 词汇表大小 (默认: 50257)
- `context_length`: 上下文长度 (默认: 1024)
- `d_model`: 模型维度 (默认: 768)
- `num_layers`: Transformer 层数 (默认: 12)
- `num_heads`: 注意力头数 (默认: 12)
- `d_ff`: 前馈网络维度 (默认: 3072)

### 训练参数
- `batch_size`: 批次大小 (默认: 32)
- `max_iters`: 最大迭代次数 (默认: 10000)
- `learning_rate`: 学习率 (默认: 3e-4)
- `warmup_iters`: 预热迭代次数 (默认: 1000)
- `weight_decay`: 权重衰减 (默认: 0.1)
- `grad_clip`: 梯度裁剪阈值 (默认: 1.0)

### 数据参数
- `data_path`: 数据文件路径 (默认: "data/tokenized_data.bin")
- `split_ratio`: 训练/验证集分割比例 (默认: 0.9)

### 检查点和日志参数
- `checkpoint_dir`: 检查点保存目录 (默认: "checkpoints")
- `save_every`: 保存检查点的频率 (默认: 1000)
- `eval_every`: 评估频率 (默认: 500)
- `log_every`: 日志记录频率 (默认: 100)

## 检查点管理

训练脚本会自动管理检查点：

- `latest.pt`: 最新的训练状态
- `checkpoint_{iter}.pt`: 定期保存的检查点
- `best_model.pt`: 验证损失最低的模型
- `final_model.pt`: 训练完成后的最终模型

### 恢复训练

```bash
# 从最新检查点恢复训练
python train.py --checkpoint_dir checkpoints

# 从特定检查点恢复
python train.py --checkpoint_dir checkpoints/experiment1
```

## 内存优化

脚本使用 `np.memmap` 进行内存高效的数据加载：

- 大型数据集不会完全加载到内存中
- 支持处理比可用内存更大的数据集
- 自动分割训练集和验证集

## 日志记录

### 控制台输出
训练过程中会显示：
- 训练损失
- 验证损失和困惑度
- 学习率变化
- 模型参数数量
- 检查点保存信息

### Weights & Biases 集成
启用后会自动记录：
- 训练指标
- 验证指标
- 学习率曲线
- 模型配置
- 系统资源使用情况

## 示例配置文件

```json
{
  "vocab_size": 50257,
  "context_length": 1024,
  "d_model": 768,
  "num_layers": 12,
  "num_heads": 12,
  "d_ff": 3072,
  "batch_size": 32,
  "max_iters": 10000,
  "learning_rate": 3e-4,
  "warmup_iters": 1000,
  "weight_decay": 0.1,
  "grad_clip": 1.0,
  "data_path": "data/tokenized_data.bin",
  "checkpoint_dir": "checkpoints",
  "use_wandb": false
}
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 减小 `context_length`
   - 使用更小的模型配置

2. **数据格式错误**
   - 确保数据文件是 `np.int32` 格式
   - 检查数据文件路径是否正确

3. **CUDA 内存不足**
   - 减小 `batch_size`
   - 使用 `--device cpu` 切换到 CPU 训练

4. **检查点加载失败**
   - 确保检查点文件完整
   - 检查模型配置是否匹配

## 性能优化建议

1. **使用 GPU 训练**：确保安装了 CUDA 版本的 PyTorch
2. **调整批次大小**：根据 GPU 内存调整 `batch_size`
3. **使用混合精度**：可以进一步优化内存使用
4. **数据预处理**：确保数据已经正确 tokenized 和格式化

## 依赖项

确保安装以下依赖：
```bash
pip install torch numpy tqdm wandb
```

或者使用项目提供的依赖管理：
```bash
# 如果使用 uv
uv sync

# 如果使用 pip
pip install -r requirements.txt
```
