# SmolLM2 SFT / PEFT Lab

用真实开源模型 `HuggingFaceTB/SmolLM2-135M-Instruct` 跑通**指令微调（SFT）与参数高效微调（LoRA/PEFT）**的完整工程链路，并在 CPU 上完成可复现的留出集评测。

本项目的目标是**验证工程链路正确**：数据 → Chat Template → assistant-only labels → 动态 padding → LoRA 注入 → 短程训练 → Adapter-only 保存 → 新进程重载比对 → 留出集评测。
它**不宣称**模型能力提升，也不构成任何泛化结论。

## 当前状态（实测）

| 项目 | 值 |
|---|---|
| 测试 | `python -m pytest tests/ -q` → **45 passed**（5.11s） |
| 环境 | Windows 10 / CPU / Python 3.10.9 / PyTorch 2.6.0+cpu / transformers 4.57.6 / peft 0.18.1 |
| 基础模型 | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| 证据产物 | `outputs/m03d09/metrics.json`、`outputs/m03d10/evaluation.json` |

### m03d09：LoRA 训练链路

`seed=42`、`steps=2`、`lr=5e-4`、`rank=4`、`alpha=8.0`、`max_length=160`，训练 12 条 / 验证 3 条：

| 指标 | 值 |
|---|---:|
| 可训练参数 | 230,400 |
| 训练 Loss（step 1 → 2） | 2.4203 → 2.1358 |
| 验证 Loss（训练前 → 训练后） | 3.817586 → 3.265532 |

Adapter 只保存增量权重，不含基础模型；`verify_adapter.py` 在**新进程**中重新加载基础模型 + Adapter 并比对 logits，用于确认保存/加载没有失真。

### m03d10：留出集分类评测

12 条独立测试样本，三类各 4 条，与训练/验证集**无重叠**（`overlap: []`）。评测采用严格单标签解析：输出必须恰好是 `billing` / `technical` / `account` 之一（忽略首尾空白与单个句末标点，大小写不敏感），否则记为 `INVALID`。

| 指标 | Base | LoRA（2 步） |
|---|---:|---:|
| 正确数 / 12 | 0 | 0 |
| Accuracy | 0.0 | 0.0 |
| Macro-F1 | 0.0 | 0.0 |
| INVALID 数 | 12 | 12 |
| 评测耗时 | 5.94s | 6.30s |

**如何解读这个 0.0**：这是**解析严格导致的负结果，不是「LoRA 无效」的结论**。135M 模型在贪心解码 8 个 token 内输出的是自然语言句子（例如 `The ticket is categorized as a billing ticket`、`You can classify the support ticket as follows`），严格解析一律判为 `INVALID`；同一批样本里 LoRA 侧已出现 `The support ticket is categorized as billing.` 这类更接近目标格式的输出，但仍是句子而非单词。

尚未解决的问题与下一步（属于后续课程日范围）：

- 用格式约束（few-shot 提示、`max_new_tokens` 收紧、或受限解码）让输出落到单标签；
- 增加训练步数与数据量，2 步更新不足以改变输出分布；
- 在解析放宽与解析严格两种口径下分别报告，避免用宽松匹配掩盖问题。

## 目录结构

```text
SmolLM2_SFT_PEFT/
├─ sft_lab/
│  ├─ dataset.py       # JSONL 读取、Chat Template、assistant-only labels、动态 padding
│  ├─ engine.py        # PEFT LoRA 注入与单个训练 Step
│  └─ evaluation.py    # 严格单标签解析、准确率 / Macro-F1 / 混淆矩阵
├─ scripts/
│  ├─ inspect_data.py  # 只下载 Tokenizer，检查真实监督范围
│  ├─ train_smoke.py   # 加载真实 135M 模型、短程训练、保存 Adapter
│  ├─ verify_adapter.py# 新进程重载并比对 logits
│  ├─ evaluate.py      # 留出集评测（Base vs LoRA）
│  └─ check_splits.py  # 训练/验证/测试划分重叠检查
├─ data/
│  ├─ train.jsonl      # 12 条英文工单分类样本
│  ├─ valid.jsonl      # 3 条
│  └─ test_m03d10.jsonl# 12 条留出样本
├─ tests/              # 离线单元测试，不下载模型、不联网
├─ outputs/            # 训练与评测产物（权重不入库，仅保留证据 JSON）
├─ README_学习指南.md   # 课程原始指南与逐文件 TODO 说明
└─ requirements.txt
```

## 运行

```bash
python -m pip install -r requirements.txt

python -m pytest tests/ -q                              # 离线单测，45 passed
python -m scripts.inspect_data                          # 只下载 Tokenizer 并检查监督范围
python -m scripts.train_smoke --steps 2 --device cpu    # 真实模型短程训练 + 保存 Adapter
python -m scripts.verify_adapter                        # 新进程重载比对
python -m scripts.evaluate                              # Base vs LoRA 留出集评测
```

首次运行需下载 Tokenizer 与真实权重；网络不可达时离线单测仍可完成，但此时不能声称已通过真实模型训练。

## 已知限制

- 数据是**手工编写的极小规模英文工单样本**（12/3/12），不代表真实业务分布。
- 只做 **2 步** LoRA 训练，不足以改变模型输出分布；`rank=4` 的选择未做消融。
- 本项目**不做 4-bit QLoRA**（不引入 bitsandbytes），只在 CPU 上验证通用 PEFT 链路。
- 可训练参数少 **不等于** 显存按同比例下降；本项目未测量显存峰值、吞吐与延迟。
- 超长样本直接报错而非静默截断，正式训练需要过滤 / 分桶 / 截断策略。
- 评测的 0.0 受严格解析影响，**不能**作为「SFT 无效」或「模型能力差」的证据。

## 与 MiniGPT 的关系

`MiniGPT`（本目录的姊妹项目）用 PyTorch 从零实现了 Decoder-only Transformer、KV Cache 与 GQA，并验证了自研 LoRA 注入的数值正确性；本项目改用 Hugging Face `transformers` + `peft` 生态，在**真实预训练权重**上验证 SFT/PEFT 工程链路。两者的实验都不能外推为真实语言任务效果。
