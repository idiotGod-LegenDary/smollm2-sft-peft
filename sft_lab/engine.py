"""PEFT LoRA 注入与最小训练 Step；本日有两个 TODO。"""
from __future__ import annotations
import math
from collections.abc import Mapping
from typing import Any
import torch
from torch import nn

def build_lora_model(
    base_model: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
) -> nn.Module:
    """TODO 4：用 PEFT 官方接口对真实 HF 因果语言模型注入 LoRA。

    输入：已从 AutoModelForCausalLM.from_pretrained() 加载的 base_model。
    返回：PEFT 包装模型，不是新创建的随机基础模型。
    契约：
      - base_model 必须是 nn.Module；rank 为非 bool 正整数；alpha 为有限正实数；
        dropout 为有限实数且 0<=p<1；target_modules 必须非空、无重复、全为合法字符串。
      - 注入前验证 model.named_modules() 中确实包含目标叶子模块名；
        未找到目标时清晰抛 ValueError，不能静默注入 0 层。
      - 用 from peft import LoraConfig, TaskType, get_peft_model，构造：
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
          lora_dropout=dropout, target_modules=list(target_modules), bias='none')。
      - return get_peft_model(base_model, lora_config)。
      - 注入后至少一项参数可训练，且可训练参数名仅属于 LoRA（含 lora_），
        否则抛 ValueError；不得重新初始化整个基础模型。
    """
    # TODO 4
    if not isinstance(base_model, nn.Module):
        raise ValueError("base_model 必须是 nn.Module")

    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank 应该是正整数")

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha 应该是有限正实数")

    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not (0 <= dropout < 1):
            raise ValueError("dropout 必须是 [0, 1) 内的有限实数")

    if not target_modules:
        raise ValueError("target_modules 必须非空")
    if len(set(target_modules)) != len(target_modules):
        raise ValueError("target_modules 必须无重复")
    if not all(isinstance(m, str) for m in target_modules):
        raise ValueError("target_modules 元素必须是字符串")

    found = False
    for name, module in base_model.named_modules():
        short = name.rsplit('.', 1)[-1] if '.' in name else name
        if short in target_modules and next(module.children(), None) is None:
            found = True
            break
    if not found:
        raise ValueError(f"未找到目标叶子模块: {target_modules}")

    
    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            lora_dropout=dropout, target_modules=list(target_modules), bias='none')
    peft_model = get_peft_model(base_model, lora_config)



    trainable_names = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise ValueError("注入后没有任何可训练参数")
    if not all("lora_" in n for n in trainable_names):
        raise ValueError("存在不属于 LoRA 的可训练参数")

    return peft_model


def train_one_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, torch.Tensor],
    *,
    device: str = "cpu",
) -> float:
    """TODO 5：使用 HF CausalLM 内置 labels Loss 做一次真实优化。

    输入：可训练的 model、已绑定可训练参数的 optimizer、
          包含 input_ids/attention_mask/labels 的 Long Tensor batch；device='cpu'/'cuda'。
    返回：更新前真实 loss.item() 对应的 Python float。
    契约：
      - model.train()；optimizer.zero_grad(set_to_none=True)；
      - 移动三个 Tensor 到 device，调用 model(**moved_batch) 得到 outputs.loss；
      - 检查 loss 是有限标量，至少有一个有效 labels（!= -100）；
      - loss.backward() 后 optimizer.step()；返回 float(loss.detach().item())；
      - 不调用 tokenizer，不手动再 shift labels，不修改 batch 字典或 Tensor 内容；
      - 不对整个模型再次加载权重。
    """
    # TODO 5
    model.train()
    optimizer.zero_grad(set_to_none=True)
    moved_batch = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "labels": batch["labels"].to(device),
    }
    if not (moved_batch["labels"] != -100).any():
        raise ValueError("至少要有一个有效 labels（!= -100）")
    outputs = model(**moved_batch)
    loss = outputs.loss
    if loss is None or not torch.isfinite(loss).all():
        raise ValueError(f"loss 不是有限值: {loss}")
    if loss.dim() != 0:
        raise ValueError(f"loss 应该是标量，实际形状: {loss.shape}")
    loss_res = float(loss.detach().item())
    loss.backward()
    optimizer.step()
    return loss_res


@torch.no_grad()
def evaluate_mean_loss(model: nn.Module, encoded: list[dict], pad_token_id: int, device: str) -> float:
    """已提供：每条验证样本等权的平均 Loss，用于烟雾检查，不是严谨的基准评测。"""
    from .dataset import collate_sft_batch
    if not encoded:
        raise ValueError("验证样本为空")
    previous_mode = model.training
    model.eval()
    losses: list[float] = []
    try:
        for ex in encoded:
            batch = collate_sft_batch([ex], pad_token_id=pad_token_id)
            inputs = {k: v.to(device) for k,v in batch.items()}
            loss = float(model(**inputs).loss.detach().item())
            if not math.isfinite(loss):
                raise ValueError("验证出现非有限 Loss")
            losses.append(loss)
    finally:
        model.train(previous_mode)
    return sum(losses) / len(losses)
