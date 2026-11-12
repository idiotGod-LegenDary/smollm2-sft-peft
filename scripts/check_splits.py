"""先单独检查三份数据，零模型下载，完成 TODO A2 后即可运行。"""
from __future__ import annotations
import json
from sft_lab.dataset import read_records
from sft_lab.evaluation import validate_splits


def main() -> None:
    train = read_records('data/train.jsonl')
    valid = read_records('data/valid.jsonl')
    test = read_records('data/test_m03d10.jsonl')
    info = validate_splits(train, valid, test)
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
