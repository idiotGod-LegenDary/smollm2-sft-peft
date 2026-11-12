"""实际生成式三分类对照：同一测试集，基础权重 vs 基础权重+已经保存的 LoRA。

重要：测试集标准答案只用于最后打分，不会加入 Prompt。
此脚本只做推理与统计，不重新训练、不覆盖旧 metrics.json。
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import time
import torch

from sft_lab.dataset import read_records
from sft_lab.evaluation import validate_splits, predict_one, classification_metrics

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M-Instruct'


def infer_all(model, tokenizer, test, device: str, max_new_tokens: int):
    rows=[]
    for i, ex in enumerate(test, start=1):
        # 绝对不能用 ex['response'] 构造 Prompt；它仅用于计算指标。
        answer = predict_one(model, tokenizer, ex['instruction'], device=device,
                             max_new_tokens=max_new_tokens)
        rows.append(answer)
        print(f'  生成 {i}/{len(test)}：预测={answer["prediction"]}, 原文={answer["raw_output"]!r}', flush=True)
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description='m03d08 SmolLM2 微调前后独立测试集生成式对照')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--adapter',default='outputs/m03d09/adapter')
    parser.add_argument('--output',default='outputs/m03d10/evaluation.json')
    parser.add_argument('--max-new-tokens',type=int,default=8)
    args=parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，使用 --device cpu')
    if type(args.max_new_tokens) is not int or args.max_new_tokens <= 0:
        raise ValueError('--max-new-tokens 必须是正整数')
    adapter_dir=Path(args.adapter)
    if not (adapter_dir/'adapter_config.json').is_file():
        raise FileNotFoundError(f'Adapter 不存在：{adapter_dir}')

    train=read_records('data/train.jsonl')
    valid=read_records('data/valid.jsonl')
    test=read_records('data/test_m03d10.jsonl')
    splits=validate_splits(train, valid, test)
    gold=[r['response'].strip().casefold() for r in test]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    if args.device == 'cpu':
        torch.set_num_threads(min(4, torch.get_num_threads()))

    print('第一阶段：基础模型，逐条生成')
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(args.device)
    base.eval()
    t0=time.perf_counter()
    base_outputs=infer_all(base,tokenizer,test,args.device,args.max_new_tokens)
    base_seconds=time.perf_counter()-t0
    del base
    gc.collect()
    if args.device=='cuda':torch.cuda.empty_cache()

    print('第二阶段：重新加载相同基础模型 + 磁盘 Adapter')
    same_base=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32)
    tuned=PeftModel.from_pretrained(same_base,adapter_dir).to(args.device)
    tuned.eval()
    t0=time.perf_counter()
    tuned_outputs=infer_all(tuned,tokenizer,test,args.device,args.max_new_tokens)
    tuned_seconds=time.perf_counter()-t0

    base_metrics=classification_metrics(gold,[x['prediction'] for x in base_outputs])
    tuned_metrics=classification_metrics(gold,[x['prediction'] for x in tuned_outputs])
    cases=[]
    for example,b,t in zip(test,base_outputs,tuned_outputs):
        cases.append({
            'instruction':example['instruction'], 'gold':example['response'],
            'base_raw':b['raw_output'],'base_prediction':b['prediction'],
            'lora_raw':t['raw_output'],'lora_prediction':t['prediction'],
            'base_correct':b['prediction']==example['response'],
            'lora_correct':t['prediction']==example['response'],
        })
    report={
        'model_id':MODEL_ID,'adapter_dir':str(adapter_dir),'device':args.device,
        'generation':{'do_sample':False,'max_new_tokens':args.max_new_tokens},
        'data':splits,'base':base_metrics,'lora':tuned_metrics,
        'base_seconds':base_seconds,'lora_seconds':tuned_seconds,
        'cases':cases,
        'limitations': '人工构造的12条独立测试样本；只进行两步LoRA训练；结果不能代表真实业务泛化能力。',
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('基础模型 Accuracy/MacroF1：',base_metrics['accuracy'],base_metrics['macro_f1'])
    print('LoRA 模型 Accuracy/MacroF1：',tuned_metrics['accuracy'],tuned_metrics['macro_f1'])
    print('实验结果：',out)


if __name__=='__main__':
    main()
