# 语言模型解码/生成函数使用说明

## 概述

这个解码函数实现了从训练好的语言模型中生成文本的功能，完全满足作业要求：

✅ **根据提示生成文本补全** - 支持用户提供的提示文本  
✅ **控制最大生成token数** - 可设置生成长度限制  
✅ **温度缩放** - 控制生成的随机性  
✅ **Top-p采样（核采样）** - 实现Holtzman等人的nucleus sampling  

## 核心功能

### 1. 文本生成 (`TextGenerator` 类)

```python
from decode import TextGenerator

# 创建生成器
generator = TextGenerator(model, vocab, device='cpu')

# 生成文本
text = generator.generate(
    prompt="Hello world",
    max_tokens=50,
    temperature=0.8,
    top_p=0.9
)
```

### 2. 温度缩放 (Temperature Scaling)

温度值控制生成的随机性：

- **低温度 (0.1-0.5)**：生成更确定、保守的文本
- **中等温度 (0.7-1.0)**：平衡的随机性
- **高温度 (1.2-2.0)**：生成更随机、创造性的文本

```python
# 确定性生成
text = generator.generate(prompt, temperature=0.3)

# 创造性生成
text = generator.generate(prompt, temperature=1.5)
```

### 3. Top-p采样 (Nucleus Sampling)

Top-p采样只从累积概率达到阈值p的最高概率token中采样：

- **p=0.9**：只考虑前90%概率的token
- **p=0.5**：只考虑前50%概率的token
- **p=1.0**：考虑所有token（等同于不使用top-p）

```python
# 保守采样
text = generator.generate(prompt, top_p=0.5)

# 宽松采样
text = generator.generate(prompt, top_p=0.9)
```

## 使用方法

### 1. 命令行使用

```bash
# 基本使用
python decode.py --prompt "Hello world" --max_tokens 50

# 指定温度和top-p
python decode.py \
    --prompt "Machine learning is" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9

# 使用训练好的模型
python decode.py \
    --model_path checkpoints/best_model.pt \
    --vocab_path data/vocabulary.json \
    --config_path configs/default_config.json \
    --prompt "Artificial intelligence"

# 保存输出到文件
python decode.py \
    --prompt "Deep learning" \
    --output_file generated_text.txt
```

### 2. Python API使用

```python
from decode import TextGenerator, load_model_and_vocab

# 加载模型和词汇表
model, vocab = load_model_and_vocab(
    model_path="checkpoints/best_model.pt",
    vocab_path="data/vocabulary.json",
    config=config,
    device="cpu"
)

# 创建生成器
generator = TextGenerator(model, vocab, device="cpu")

# 生成文本
generated_text = generator.generate(
    prompt="The future of artificial intelligence",
    max_tokens=100,
    temperature=0.8,
    top_p=0.9
)

print(generated_text)
```

### 3. 交互式使用

```bash
# 运行交互式演示
python example_generation.py
```

## 参数说明

### 生成参数

- `prompt` (str): 输入提示文本
- `max_tokens` (int): 最大生成token数 (默认: 100)
- `temperature` (float): 温度值 (默认: 1.0)
- `top_p` (float): Top-p采样阈值 (默认: 1.0)
- `stop_token` (str): 停止token (默认: '<|endoftext|>')

### 模型参数

- `model_path` (str): 模型检查点路径
- `vocab_path` (str): 词汇表文件路径
- `config_path` (str): 模型配置文件路径
- `device` (str): 设备类型 ('cpu' 或 'cuda')

## 示例效果

### 不同温度值的效果

```
提示: "Machine learning is"

温度 0.3: a powerful tool for data analysis and prediction
温度 0.8: an exciting field that continues to evolve rapidly
温度 1.2: like a magical wand that transforms raw data into insights
```

### 不同Top-p值的效果

```
提示: "Artificial intelligence"

Top-p 0.5: will revolutionize technology
Top-p 0.7: has the potential to transform many industries
Top-p 0.9: represents a fundamental shift in how we approach problem-solving
```

## 技术实现

### 1. 温度缩放

```python
def apply_temperature(self, logits, temperature):
    if temperature <= 0:
        raise ValueError("温度值必须大于0")
    return logits / temperature
```

### 2. Top-p采样

```python
def top_p_sampling(self, logits, p):
    # 计算概率分布
    probs = F.softmax(logits, dim=-1)
    
    # 按概率降序排序
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    # 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累积概率超过p的位置
    sorted_indices_to_remove = cumulative_probs > p
    
    # 创建mask并过滤logits
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    filtered_logits = logits.clone()
    filtered_logits[indices_to_remove] = float('-inf')
    
    return filtered_logits
```

### 3. 生成循环

```python
def generate(self, prompt, max_tokens, temperature, top_p):
    # Tokenize提示
    input_tokens = self.tokenize(prompt)
    input_tensor = torch.tensor([input_tokens], device=self.device)
    
    generated_tokens = []
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # 前向传播
            logits = self.model(input_tensor)
            next_token_logits = logits[0, -1, :]
            
            # 采样下一个token
            next_token = self.sample_next_token(
                next_token_logits, temperature, top_p
            )
            
            # 检查停止条件
            if next_token == self.endoftext_id:
                break
            
            generated_tokens.append(next_token)
            
            # 更新输入tensor
            input_tensor = torch.cat([
                input_tensor, 
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)
    
    return self.detokenize(generated_tokens)
```

## 最佳实践

### 1. 参数调优

- **创意写作**: `temperature=1.2-1.5`, `top_p=0.9`
- **技术文档**: `temperature=0.7-0.9`, `top_p=0.8`
- **代码生成**: `temperature=0.3-0.5`, `top_p=0.7`

### 2. 提示工程

- 使用清晰、具体的提示
- 包含上下文信息
- 指定期望的输出格式

### 3. 性能优化

- 使用GPU加速（如果可用）
- 适当设置max_tokens避免过长生成
- 批量生成多个候选文本

## 故障排除

### 常见问题

1. **生成质量差**
   - 检查模型是否充分训练
   - 调整温度和top-p参数
   - 改进提示文本

2. **生成速度慢**
   - 使用GPU加速
   - 减小max_tokens
   - 使用更小的模型

3. **内存不足**
   - 减小batch_size
   - 使用CPU生成
   - 分段生成长文本

4. **词汇表不匹配**
   - 确保使用正确的词汇表
   - 检查tokenization方法
   - 更新词汇表

## 扩展功能

### 1. 批量生成

```python
def batch_generate(self, prompts, **kwargs):
    results = []
    for prompt in prompts:
        result = self.generate(prompt, **kwargs)
        results.append(result)
    return results
```

### 2. 条件生成

```python
def conditional_generate(self, prompt, condition, **kwargs):
    # 在提示中加入条件信息
    full_prompt = f"{condition}: {prompt}"
    return self.generate(full_prompt, **kwargs)
```

### 3. 多样性采样

```python
def diverse_generate(self, prompt, num_samples=5, **kwargs):
    samples = []
    for _ in range(num_samples):
        sample = self.generate(prompt, **kwargs)
        samples.append(sample)
    return samples
```

## 总结

这个解码函数提供了完整的文本生成功能，支持：

- ✅ 灵活的提示输入
- ✅ 可控的生成长度
- ✅ 温度控制随机性
- ✅ Top-p采样提高质量
- ✅ 易于使用的API
- ✅ 详细的文档和示例

可以直接用于生产环境或进一步扩展功能。
