"""只读取 Tokenizer 与数据，不下载模型权重；完成前三个 TODO 后执行。"""
from __future__ import annotations
from transformers import AutoTokenizer
from sft_lab.dataset import read_records, encode_instruction_sample, collate_sft_batch
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    record = read_records("data/train.jsonl")[0]
    encoded = encode_instruction_sample(tok, record, max_length=160)
    batch = collate_sft_batch([encoded], pad_token_id=tok.pad_token_id)
    print("源指令：", record["instruction"])
    print("目标回复：", record["response"])
    print("Token 数：", len(encoded["input_ids"]))
    print("参与监督的 Token 数：", sum(x != -100 for x in encoded["labels"]))
    print("监督片段解码：", tok.decode([x for x in encoded["labels"] if x != -100]))
    print("Batch shape：", tuple(batch["input_ids"].shape))

if __name__ == "__main__":
    main()
