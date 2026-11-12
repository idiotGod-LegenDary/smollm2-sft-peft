# month 03 day 07：真实模型 SFT / PEFT 中文工程指南

## 这是新项目，不覆盖 MiniGPT

在 `MiniGPT/` **旁边**解压本目录，运行时先 `cd SmolLM2_SFT_PEFT_M03D07`。
所有 `outputs/` 均由真实运行后生成，不是课程预置的结果。

## 文件职责与 TODO

| 文件 | 作用 | 学生需实现 |
|---|---|---|
| `sft_lab/dataset.py` | 读取 JSONL、Chat Template、assistant-only labels、动态 Padding | TODO 1/2/3 |
| `sft_lab/engine.py` | PEFT LoRA 注入、一个训练 Step | TODO 4/5 |
| `data/train.jsonl` | 12 条手工编写英文工单训练样本 | 已提供 |
| `data/valid.jsonl` | 3 条独立英文验证样本 | 已提供 |
| `scripts/inspect_data.py` | 只下载 Tokenizer、检查真实监督范围 | 已提供 |
| `scripts/train_smoke.py` | 加载真实 135M 模型、短程训练、保存 Adapter、记录 Loss | 已提供 |
| `scripts/verify_adapter.py` | **新进程**重新加载基础模型+Adapter，并比较 logits | 已提供 |
| `tests/` | 离线单元测试（不下载模型） | 已提供 |

## 本项目数据形式

每行一个 JSON 对象：

```json
{"instruction": "Classify ... Ticket: ...", "response": "billing"}
```

本日样本用于验证真实语言训练管线，但数据规模极小，验证 Loss 不代表统计意义上的泛化。下一日更换为有正式训练/验证划分的数据集，并增加分类正确率和对比生成样例。

## Windows CMD 运行（Python 3.10+）

前提：已安装适合本机 CPU/CUDA 的 PyTorch。你可以先运行：

```bat
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

然后：

```bat
python -m pip install -r requirements.txt
python -m pytest tests/ -q
python -m scripts.inspect_data
python -m scripts.train_smoke --steps 2 --device cpu
python -m scripts.verify_adapter
```

首次运行会下载模型 Tokenizer 与真实开源权重；如果网络不可达，离线测试仍然可以先完成，但不能声称已通过真实模型训练。

如果 CPU 较慢，可先设 `--steps 1`，待流水线运行正常再做两步。任务重点是工程正确性，不承诺几步训练即可改善模型质量。

## 观察并记录

- `outputs/m03d09/metrics.json`：模型标识、训练/验证样本数、真实训练 Loss、验证 Loss、可训练参数量。
- `outputs/m03d09/adapter/`：PEFT Adapter 文件；不包含完整基础权重。
- `outputs/m03d09/reload_reference.pt`：用于比较的输入与最后位置 logits。
- `python -m scripts.verify_adapter` 的数值差异结果。

## 特别重要

1. Prompt 位置用 `labels=-100`，Assistant 内容才参与监督，**不手工 shift**。
2. 用 `apply_chat_template(tokenize=True)` 处理 Token，不能再重复添加 special tokens。
3. 本项目对超长样本直接报错，不静默截断答案；正式训练需要过滤/分桶/截断策略。
4. 模型虽已指令对齐，本日 SFT 仅是小数据进一步训练，不代表从零预训练或从零对齐。
5. LoRA 训练参数少，不等于总显存按相同比例降低。
6. 本日不做 4-bit QLoRA，不安装 bitsandbytes；Win/CPU 上先验证通用 PEFT 链路。
7. 训练输出不会自动发布到 Hub；避免上传私人数据/模型权重。
