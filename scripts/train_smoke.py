"""真实 SmolLM2-135M-Instruct + PEFT LoRA 冒烟训练：不会使用随机模型代替。"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sft_lab.dataset import read_records, ensure_disjoint, encode_instruction_sample, collate_sft_batch
from sft_lab.engine import build_lora_model, train_one_step, evaluate_mean_loss

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

@dataclass(frozen=True)
class RunConfig:
    model_id: str = MODEL_ID
    seed: int = 42
    max_length: int = 160
    steps: int = 2
    learning_rate: float = 5e-4
    rank: int = 4
    alpha: float = 8.0
    device: str = "cpu"


def run(cfg: RunConfig, output: Path) -> dict:
    if cfg.device not in ("cpu", "cuda"):
        raise ValueError("device 只允许 cpu/cuda")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；改用 --device cpu")
    if type(cfg.steps) is not int or not 1 <= cfg.steps <= 20:
        raise ValueError("本课 steps 必须位于 [1,20]")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    train = read_records("data/train.jsonl")
    valid = read_records("data/valid.jsonl")
    ensure_disjoint(train, valid)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer 无 PAD/EOS Token")
        tokenizer.pad_token = tokenizer.eos_token
    encoded_train = [encode_instruction_sample(tokenizer, x, max_length=cfg.max_length) for x in train]
    encoded_valid = [encode_instruction_sample(tokenizer, x, max_length=cfg.max_length) for x in valid]
    # 本课使用 CPU FP32，GPU 同样用 FP32 以便进行基本比较；不使用 4-bit 量化。
    base = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype=torch.float32)
    base.config.use_cache = False
    model = build_lora_model(base, rank=cfg.rank, alpha=cfg.alpha)
    model.to(cfg.device)
    model.print_trainable_parameters()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable <= 0:
        raise AssertionError("LoRA 没有任何可训练参数")
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    baseline = evaluate_mean_loss(model, encoded_valid, tokenizer.pad_token_id, cfg.device)
    history = []
    for step in range(cfg.steps):
        ex = encoded_train[step % len(encoded_train)]
        batch = collate_sft_batch([ex], pad_token_id=tokenizer.pad_token_id)
        loss = train_one_step(model, opt, batch, device=cfg.device)
        history.append({"step": step + 1, "train_loss": loss})
        print(f"Step {step+1}/{cfg.steps}, train_loss={loss:.5f}")
    final = evaluate_mean_loss(model, encoded_valid, tokenizer.pad_token_id, cfg.device)
    output.mkdir(parents=True, exist_ok=True)
    adapter_dir = output / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    # 单独留一份用于重载数值验证的输入和最后位置 logits。
    probe_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": valid[0]["instruction"]}],
        tokenize=True, add_generation_prompt=True, return_dict=False,
    )
    probe_ids = torch.tensor([probe_ids], dtype=torch.long, device=cfg.device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=probe_ids, attention_mask=torch.ones_like(probe_ids)).logits[:, -1].float().cpu()
    torch.save({"input_ids": probe_ids.cpu(), "last_logits": logits}, output / "reload_reference.pt")
    record = {
        "config": asdict(cfg),
        "train_samples": len(train), "valid_samples": len(valid),
        "trainable_parameters": trainable,
        "baseline_valid_mean_loss": baseline,
        "final_valid_mean_loss": final,
        "history": history,
        "note": "手工编写的极小规模英文工单数据，仅验证真实预训练模型的 SFT/PEFT 工程链路，不得宣称泛化提升。",
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("基础验证 Loss：", baseline)
    print("结束验证 Loss：", final)
    print("Adapter 保存目录：", adapter_dir)
    return record


def main() -> None:
    parser=argparse.ArgumentParser(description="SmolLM2 真实开源模型 LoRA/SFT 冒烟训练")
    parser.add_argument("--steps",type=int,default=2)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--output",default="outputs/m03d09")
    args=parser.parse_args()
    run(RunConfig(steps=args.steps,device=args.device),Path(args.output))

if __name__=="__main__":
    main()
