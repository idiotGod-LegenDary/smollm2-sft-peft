"""month 03 day 08 / A线：工单分类的无泄漏评测组件。

只新增本文件，不覆盖上一日 sft_lab/dataset.py 与 sft_lab/engine.py。
请独立完成 TODO 1~4；单元测试不会调用真实模型，也不会访问网络。
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Any
import torch
import string

LABELS: tuple[str, ...] = ("billing", "technical", "account")


def normalize_prediction(raw: str) -> str | None:
    """TODO A1：把生成文本转换成严格的类别预测。

    输入：模型生成部分 decode 后的原始字符串 raw。
    输出：只可能是 'billing'/'technical'/'account'/None。
    规则：非 str 抛 TypeError；忽略首尾空白和单个末尾标点 .!?；大小写不敏感。
         例如 ' BILLING. ' -> 'billing'；'billing because...' -> None；'' -> None。
         不允许通过 `if 'billing' in raw` 这种包含关系猜类别。
    """
    if not isinstance(raw, str):
        raise TypeError("raw 类型必须是 str")
    raw = raw.strip()
    if raw and raw[-1] in string.punctuation:
        raw = raw[:-1]
    raw = raw.strip().lower()

    return raw if raw in LABELS else None

def validate_splits(
    train: Sequence[Mapping[str, str]],
    valid: Sequence[Mapping[str, str]],
    test: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """TODO A2：检查三份数据并统计类别分布。

    输入：三个非空序列，元素形如 {'instruction':str, 'response':str}。
    输出：{'sizes':{'train':int,'valid':int,'test':int},
           'class_counts':{'train':{label:int,...},...}, 'overlap':[]}。
    契约：
      - 任何字段缺失、空白指令、不属于 LABELS 的答案、任意切分内重复或跨切分重复，抛 ValueError。
      - 去重比较键为 instruction.strip().casefold()，但不改变原始文本。
      - response.strip().casefold() 必须是允许的类别；三份数据都要求三类出现至少 1 次。
      - 不在此函数中读取文件，不向模型传入正确答案，不自动修改输入。
    """
    insset = set()
    splits = (("train", train), ("valid", valid), ("test", test))

    for name, data in splits:
        for s in data:
            if 'instruction' not in s or 'response' not in s:
                raise ValueError(f"{name}: 字段缺失")
            ins, res = s['instruction'], s['response']
            if not ins.strip():
                raise ValueError(f"{name}: 存在空白指令")
            res_norm = res.strip().casefold()
            if res_norm not in LABELS:
                raise ValueError(f"{name}: 答案不属于 LABELS")
            k = ins.strip().casefold()
            if k in insset:
                raise ValueError(f"{name}: 重复 {k!r}")
            insset.add(k)
    
    from collections import Counter
    class_counts = {}
    for name, data in splits:
        counter = Counter()
        for s in data:
            counter[s['response'].strip().casefold()] += 1
        class_counts[name] = {label: counter[label] for label in LABELS}

        for label in LABELS:
            if class_counts[name][label] < 1:
                raise ValueError(f"{name} 缺少类别 {label}")

    return {
        "sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "class_counts": class_counts,
        "overlap": [],
    }



def classification_metrics(
    gold: Sequence[str], predictions: Sequence[str | None]
) -> dict[str, Any]:
    """TODO A3：根据真实预测计算准确率、Macro-F1 和混淆矩阵。

    输入：gold 为 LABELS 里的标准标签；predictions 为 normalize_prediction 后的标签或 None。
    输出必须包含：
      samples:int, correct:int, accuracy:float, macro_f1:float,
      invalid_count:int, per_class:{label:{precision,recall,f1,support}},
      confusion:{真标签:{billing:int,technical:int,account:int,INVALID:int}}
    契约：
      - 空数据、长度不一致、非法真值/预测标签抛 ValueError；不能用概率或 Loss 代替预测。
      - None 记为 INVALID 并计作错误，同时属于真实类别的 False Negative。
      - 每类 P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R)，分母0时值0。
      - Macro-F1 是三个类的 F1 算术平均；accuracy=correct/samples。
      - 数值返回普通 Python float，不使用 sklearn，不更改传入数据。
    """
    if len(gold) != len(predictions):
        raise ValueError("长度不一致")
    if not predictions:
        raise ValueError("无数据")
    for g in gold:
        if g not in LABELS:
            raise ValueError(f"非法真值标签: {g!r}")
    for p in predictions:
        if p is not None and p not in LABELS:
            raise ValueError(f"非法预测标签: {p!r}")
        
    per_class = {}
    confusion: dict[str, dict[str, int]] = {
        "billing":{"billing":0, "technical":0, "account":0,"INVALID":0}, 
        "technical":{"billing":0, "technical":0, "account":0,"INVALID":0}, 
        "account":{"billing":0, "technical":0, "account":0,"INVALID":0}
        }
    samples = len(gold)
    correct = invalid_count = 0
    for g, p in zip(gold, predictions):
        if p is None:
            invalid_count += 1
            col = "INVALID"
        else:
            col = p
            if p == g:
                correct += 1
        confusion[g][col] += 1

    for c in LABELS:
        tp = confusion[c][c]
        support = sum(confusion[c].values())          # 真实为 c 的总数 = TP + FN
        pred_c = sum(confusion[g][c] for g in LABELS) # 预测为 c 的总数 = TP + FP

        precision = tp / pred_c if pred_c else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)

        per_class[c] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }


    macro_f1 = sum(per_class[c]["f1"] for c in LABELS) / len(LABELS)

    return {
        "samples": samples,
        "correct": correct,
        "accuracy": correct / samples,
        "macro_f1": float(macro_f1),
        "invalid_count": invalid_count,
        "per_class": per_class,
        "confusion": confusion,
    }


def predict_one(
    model: Any, tokenizer: Any, instruction: str, *,
    device: str = "cpu", max_new_tokens: int = 8,
) -> dict[str, str | None]:
    """TODO A4：不泄漏标签地对单条指令进行贪心生成，并只解码新增 Token。

    输入：已加载的基础模型或 PEFT 模型、配套 tokenizer、用户指令、device、最大生成 Token 数。
    输出：{'raw_output':str, 'prediction': LABELS 中的字符串或 None}。
    步骤：
      1. 检查 instruction 非空，max_new_tokens 是正整数（bool 非法）。
      2. tokenizer.apply_chat_template([{'role':'user','content':instruction}],
           tokenize=True, add_generation_prompt=True, return_dict=False) -> Prompt ID 列表。
      3. 转换为 [1,T] long Tensor，构造全1 attention_mask，放至 device。
      4. model.eval() + torch.inference_mode()；调用 model.generate(
           input_ids=..., attention_mask=..., do_sample=False,
           max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)。
      5. 只能切出输出中 input_ids.shape[1]: 的新 Token；tokenizer.decode(新Token,skip_special_tokens=True)。
      6. normalize_prediction(raw_output)，不可将整段 Prompt 当模型回答，也不可把 gold 输入 Prompt。
      7. 如果输出长度短于 Prompt、没有新 Token 或返回形状不对，抛 ValueError。
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction 不能为空")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens 应该是正整数")

    prompt_ids = tokenizer.apply_chat_template([{'role':'user','content':instruction}], tokenize=True, add_generation_prompt=True, return_dict=False)
    prompt = torch.tensor(prompt_ids,dtype=torch.long, device=device).unsqueeze(0) 
    attn_mask = torch.ones_like(prompt, device=device)
    model.eval()
    with torch.inference_mode():
        output = model.generate(
            input_ids = prompt,
            attention_mask=attn_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens, 
            pad_token_id=tokenizer.pad_token_id
        )
    if not isinstance(output, torch.Tensor) or output.dim() != 2 or output.shape[0] != 1:
        raise ValueError("generate 返回形状异常")

    prompt_len = prompt.shape[1]          # 原始输入长度
    if output.shape[1] < prompt_len:
        raise ValueError("输出长度短于 Prompt")
    if output.shape[1] == prompt_len:
        raise ValueError("没有生成新 Token")

    # 步骤 5：只切新 token
    new_tokens = output[0, prompt_len:]   # [new_len]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # 步骤 6：归一化
    prediction = normalize_prediction(raw_output)

    return {"raw_output": raw_output, "prediction": prediction}

