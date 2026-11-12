"""重新加载基础模型+磁盘 Adapter，以独立进程核对最后一个位置 logits。"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

def main() -> None:
    parser = argparse.ArgumentParser(description="重新读取已保存 PEFT Adapter 并检查 logits")
    parser.add_argument("--output", default="outputs/m03d09")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    directory = Path(args.output)
    saved = torch.load(directory / "reload_reference.pt", map_location="cpu", weights_only=True)
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model = PeftModel.from_pretrained(base, directory / "adapter").to(args.device)
    model.eval()
    ids = saved["input_ids"].to(args.device)
    with torch.no_grad():
        current = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, -1].float().cpu()
    prior = saved["last_logits"]
    diff = float((current - prior).abs().max().item())
    print(f"Adapter 重载最大 logits 绝对差：{diff:.8g}")
    if not torch.allclose(current, prior, atol=1e-4, rtol=1e-4):
        raise AssertionError("重载输出不一致，检查基础模型版本、配置与精度")
    print("重载一致性检查通过")

if __name__ == "__main__":
    main()
