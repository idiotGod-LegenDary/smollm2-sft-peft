"""指令数据与 assistant-only labels 构造；本日有三个 TODO。"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import torch

REQUIRED_KEYS = {"instruction", "response"}

def read_records(path: str | Path) -> list[dict[str, str]]:
    """
    TODO 1：读取并严格验证 UTF-8 JSONL 指令数据。

    输入：存在的 JSONL 文件；每个非空行是一个 JSON 对象。
    返回：非空 list[{'instruction': str, 'response': str}]，保留文件顺序。
    契约：
      - 跳过纯空白行，行号按原文件行号计数；
      - 对象必须恰好含 instruction/response 两个键，值均为非空字符串（去除首尾空格后非空）；
      - 返回原字符串，不偷偷修改训练文本；	
      - 格式错误、无效样本、空文件抛 ValueError，并尽可能指出行号；
      - 找不到路径允许 FileNotFoundError 传播；不执行任何下载。
    """
    # TODO 1    
    if not isinstance(path, (str, Path)):
        raise ValueError("path type must be str or Path")
    p = Path(path)

    with open(p, "r", encoding="utf-8") as f:
        records = []
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"[第 {i} 行] 不是合法 JSON：{e}") from e
            
            if not isinstance(obj, dict):
                raise ValueError(f"[第 {i} 行] 必须是 JSON object，实际是 {type(obj).__name__}")
            
            if set(obj.keys()) != REQUIRED_KEYS:
                raise ValueError(f"[第 {i} 行] 键必须恰好为 {REQUIRED_KEYS}，实际为 {set(obj.keys())}")
            
            for key in REQUIRED_KEYS:
                value = obj[key]
                if not isinstance(value, str):
                    raise ValueError(f"[第 {i} 行] {key} 必须是 str，实际是 {type(value).__name__}")
                if not value.strip():
                    raise ValueError(f"[第 {i} 行] {key} 不能为空白字符串")
            records.append({"instruction": obj["instruction"], "response": obj["response"]})

    if not records:
        raise ValueError("文件为空或没有有效样本")
    return records


def encode_instruction_sample(
    tokenizer: Any,
    example: Mapping[str, str],
    *,
    max_length: int = 160,
) -> dict[str, list[int]]:
    """TODO 2：使用目标模型 chat template 构造 assistant-only 监督标签。

    输入：有 instruction/response 的映射、支持 apply_chat_template 的 tokenizer、最大长度正整数。
    返回：恰好三个字段：input_ids / attention_mask / labels，均为等长整数列表。
    契约：
      - prompt_msgs=[{'role':'user','content':instruction}]；
        full_msgs=prompt_msgs + [{'role':'assistant','content':response}]。
      - prompt_ids: apply_chat_template(prompt_msgs, tokenize=True,
                        add_generation_prompt=True, return_dict=False)。
      - full_ids: apply_chat_template(full_msgs, tokenize=True,
                        add_generation_prompt=False, return_dict=False)。
      - 保证 prompt_ids 是 full_ids 的严格前缀，否则 ValueError；
      - labels = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]；
        input_ids=full_ids，attention_mask 全 1。
      - 至少有一个真实监督 Token；不手动 shift labels，不重复添加 special tokens。
      - 为避免悄悄截断正确答案：full_ids 超过 max_length 时抛 ValueError，
        今天直接修改样本/增加上限，不自动丢弃回答结尾。
      - 非法 max_length、空字符串、非法 token ID 均抛 ValueError/TypeError。
    """
    # TODO 2
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length 必须是整数")
    if max_length <= 0:
        raise ValueError("max_length 必须是正整数")
    
    prompt_msgs = []
    full_msgs = []
    instruction = example["instruction"]
    response = example["response"]
    if isinstance(instruction, str) and instruction.strip() == "":
        raise ValueError("instruction不能是空字符串")
    if isinstance(response, str) and response.strip() == "":
            raise ValueError("response 不能是空字符串")

    prompt_msgs.append({'role':'user','content':instruction})
    full_msgs = prompt_msgs + [{"role": "assistant", "content": response}]
    prompt_ids = tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=False)
    full_ids = tokenizer.apply_chat_template(full_msgs, tokenize=True, add_generation_prompt=False, return_dict=False)
    prompt_ids = list(prompt_ids)
    full_ids = list(full_ids)

    prompt_len = len(prompt_ids)
    full_len = len(full_ids)
    if full_len <= prompt_len:
        raise ValueError("没有可监督的 assistant token")
    if full_len > max_length :
        raise ValueError("超长")
    if prompt_ids != full_ids[:prompt_len]:
        raise ValueError("prompt_ids 不是 full_ids 的前缀")

    if not all(isinstance(t, int) and t >= 0 for t in full_ids):
        raise ValueError("存在非法 token ID")

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    attention_mask = [1] * full_len
    return {
        'input_ids':full_ids,
        'attention_mask':attention_mask,
        'labels':labels
    }

def collate_sft_batch(
    examples: Sequence[Mapping[str, Sequence[int]]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """TODO 3：动态右侧 Padding，分别处理输入、注意力与标签。

    输入：非空样本列表，各样本已含 input_ids/attention_mask/labels 三个等长整数序列；
          pad_token_id：非 bool、非负整数。
    返回：三组 torch.long Tensor，形状均为 [B, max_T]。
    契约：
      - input_ids 用 pad_token_id 补齐；attention_mask 用 0 补齐；labels 用 -100 补齐；
      - 不修改任何输入样本；空 batch/无监督 Token/错位长度需抛 ValueError；
      - 检查非 padding 区域 attention_mask 都为 1；不允许返回 Tensor float dtype。
    """
    # TODO 3
    if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
        raise TypeError("pad_token_id 必须是整数")
    if pad_token_id < 0:
        raise ValueError("pad_token_id 必须是非负整数")

    if not examples:
        raise ValueError("batch 不能为空")

    seqs = []
    for example in examples:      
        try:
            ids = example["input_ids"]
            mask = example["attention_mask"]
            lbls = example["labels"]
        except KeyError as e:
            raise ValueError(f"样本缺少字段: {e}") from e

        if not(len(ids) == len(mask) == len(lbls)):
            raise ValueError("长度不一致")
        if all(label == -100 for label in lbls):
            raise ValueError("无监督token")
        if any(m != 1 for m in mask):
            raise ValueError("非 padding 区域的 attention_mask 必须全为 1")
        
        seqs.append((ids, mask, lbls))

    max_len = max(len(ids) for ids, _, _ in seqs)

    input_ids = []
    attention_mask = []
    labels = []
    for ids, mask, lbls in seqs:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_token_id] * pad_len)
        attention_mask.append(mask + [0] * pad_len)
        labels.append(lbls + [-100] * pad_len)

    return {
        'input_ids':torch.tensor(input_ids,dtype = torch.long),
        'attention_mask':torch.tensor(attention_mask,dtype = torch.long),
        'labels':torch.tensor(labels,dtype = torch.long)
    }

def ensure_disjoint(train: list[dict[str,str]], valid: list[dict[str,str]]) -> None:
    """已提供：避免相同指令同时出现在训练与验证集合。"""
    t = {x['instruction'].strip().lower() for x in train}
    v = {x['instruction'].strip().lower() for x in valid}
    overlap = sorted(t & v)
    if overlap:
        raise ValueError(f"train/valid 指令重复：{overlap[:3]}")
