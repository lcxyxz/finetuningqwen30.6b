# Qwen3-0.6B 微调项目 - 甄嬛风格对话模型

本项目使用 LoRA (Low-Rank Adaptation) 技术对 Qwen3-0.6B 模型进行微调，让模型学会模仿《甄嬛传》中甄嬛的说话风格，包括其典雅、含蓄、有深度的语言特点，以及善用典故的表达方式。

我自己训练出来的小模型参数同步上传，只需要将 system prompt = "床前明月光" 即可在推理代码中直观看到模型的变化。很多参数其实还能更优化 比如lora的秩感觉可以设置为8。

## 项目结构

```
.
├── datasets/                    # 数据集目录
│   ├── huanhuan.json           # 原始数据集（JSON格式）
│   └── huanhuan_train.jsonl    # 转换后的训练数据（JSONL格式）
├── model/                       # 模型输出目录（训练后生成）
├── src/                         # 源代码目录
│   └── finetuningqwen30_6b/
│       ├── __init__.py          # 包初始化
│       ├── preprocess.py        # 数据预处理脚本
│       └── train.py             # 模型训练脚本
├── util.py/                     # 工具函数目录
│   └── loadmodel.py             # 模型加载示例
├── inference.py                 # 模型推理与交互脚本
├── pyproject.toml               # 项目配置文件
├── requirements.txt             # 依赖列表
└── README.md                    # 项目说明文件
```

## 技术栈

- **基础模型**: Qwen3-0.6B (阿里巴巴通义千问)
- **微调方法**: LoRA (Low-Rank Adaptation)
- **训练框架**: Hugging Face Transformers + PEFT + TRL
- **数据格式**: JSONL (对话格式)
- **Python版本**: 3.14+

## 安装

### 1. 克隆项目
```bash
git clone <项目地址>
cd finetuningqwen30.6b
```

### 2. 创建虚拟环境（推荐）
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. 安装依赖
```bash
# 使用pip安装
pip install -r requirements.txt

# 或者使用uv（推荐）
uv sync
```

## 使用方法

### 数据预处理

将原始的 `huanhuan.json` 数据转换为 Qwen3 支持的对话训练格式：

```bash
python -m src.finetuningqwen30_6b.preprocess
```

执行后会生成 `datasets/huanhuan_train.jsonl` 文件。

### 模型训练

#### 基础训练
```bash
python -m src.finetuningqwen30_6b.train
```

#### 自定义训练参数
```bash
python -m src.finetuningqwen30_6b.train \
    --output_dir ./model/huanhuanmodel \
    --num_train_epochs 5.0 \
```

#### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `model/qwen3-0.6b-huanhuan-lora` | 模型输出目录 |
| `--num_train_epochs` | `3.0` | 训练轮数 |
| `--max_steps` | `-1` | 最大训练步数（>0 时覆盖 num_train_epochs，用于快速试跑） |

### 训练监控

训练过程中可以使用 TensorBoard 监控训练进度：

```bash
tensorboard --logdir ./model/huanhuanmodel
```

### 模型推理

#### 交互式对话模式

训练完成后，可以使用交互式模式与微调后的模型对话：

```bash
python inference.py
```

交互式模式支持以下命令：
- `quit` 或 `exit`：退出程序
- `clear`：清空对话历史
- `thinking on/off`：开启/关闭思考模式
- `compare`：对比微调模型与基础模型的输出

#### 对比模式

在交互式模式中输入 `compare`，可以对比微调模型与基础模型对同一问题的回答差异。

## 训练配置说明

本项目使用 LoRA 进行参数高效微调，主要配置如下：

### LoRA 配置
- **秩 (r)**: 16 - 控制可训练参数量
- **缩放系数 (lora_alpha)**: 32 - 通常为秩的两倍
- **Dropout**: 0.05
- **目标模块**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### 训练超参数
- **批大小**: 8 (每设备)
- **梯度累积步数**: 2 (等效批大小 = 16)
- **学习率**: 2e-4
- **学习率调度器**: cosine
- **预热步数**: 20
- **最大序列长度**: 512
- **精度**: bf16

## 数据集说明

本项目使用《甄嬛传》相关对话数据进行训练，数据格式如下：

```json
{
  "instruction": "用户问题",
  "input": "可选的补充输入",
  "output": "甄嬛风格的回答"
}
```

经过预处理后，数据会被转换为 Qwen3 支持的对话格式：

```json
{
  "messages": [
    {"role": "system", "content": "床前明月光"},
    {"role": "user", "content": "用户问题"},
    {"role": "assistant", "content": "甄嬛风格的回答"}
  ]
}
```

## 常见问题

### 1. 显存不足
可以尝试以下方法：
- 减小 `per_device_train_batch_size`
- 减小 `max_length`
- 使用更小的 LoRA 秩 (r)
- 启用梯度检查点 (gradient checkpointing)

### 2. 训练速度慢
- 确保使用 GPU 进行训练
- 检查是否启用了 bf16 混合精度训练
- 增加批大小（如果显存允许）

### 3. 模型效果不好
- 增加训练轮数
- 调整学习率
- 检查数据质量
- 增加训练数据量

## 许可证

本项目基于 MIT 许可证开源。

## 联系方式

- 作者: XianYuSenior
- 邮箱: 1148656615@qq.com

ps: 数据集是从原来一个git仓库找到的，忘记在什么仓库了。如果有侵权请联系我删除，谢谢