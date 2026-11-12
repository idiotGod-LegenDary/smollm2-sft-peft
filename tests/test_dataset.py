import json
import pytest
import torch
from sft_lab.dataset import read_records, encode_instruction_sample, collate_sft_batch, ensure_disjoint


def test_read_valid_and_blank_lines(tmp_path):
    p=tmp_path/'records.jsonl'
    p.write_text('\n'+json.dumps({'instruction':'hello','response':'hi'})+'\n',encoding='utf-8')
    assert read_records(p)==[{'instruction':'hello','response':'hi'}]

@pytest.mark.parametrize('text',["[]\n",'{}\n', '{"instruction":"a"}\n',
    '{"instruction":"  ","response":"b"}\n',
    '{"instruction":3,"response":"b"}\n', '{not json}\n', '\n\n'])
def test_read_reject_invalid(tmp_path,text):
    p=tmp_path/'bad.jsonl';p.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError): read_records(p)

def test_encode_prefix_and_mask(fake_tokenizer):
    ex={'instruction':'Please classify this ticket','response':'billing'}
    result=encode_instruction_sample(fake_tokenizer,ex,max_length=200)
    prefix=fake_tokenizer.apply_chat_template([{'role':'user','content':ex['instruction']}],tokenize=True,add_generation_prompt=True,return_dict=False)
    assert result['input_ids'][:len(prefix)]==prefix
    assert result['labels'][:len(prefix)]==[-100]*len(prefix)
    assert result['labels'][len(prefix):]==result['input_ids'][len(prefix):]
    assert result['attention_mask']==[1]*len(result['input_ids'])
    assert len(result['labels'])==len(result['input_ids'])

def test_encode_max_length_reject(fake_tokenizer):
    with pytest.raises(ValueError):
        encode_instruction_sample(fake_tokenizer,{'instruction':'x','response':'y'},max_length=3)

def test_encode_bad_example(fake_tokenizer):
    with pytest.raises((ValueError,TypeError)):
        encode_instruction_sample(fake_tokenizer,{'instruction':'', 'response':'ok'})

def test_encode_no_answer_mask_failure():
    class BadTokenizer:
        def apply_chat_template(self, messages, **kw):
            return [1,2,3] if len(messages)==1 else [2,3,4]
    with pytest.raises(ValueError):
        encode_instruction_sample(BadTokenizer(), {'instruction':'hi','response':'ok'})

def test_batch_padding_and_dtype(fake_tokenizer):
    samples=[encode_instruction_sample(fake_tokenizer,{'instruction':'short','response':'billing'}),
             encode_instruction_sample(fake_tokenizer,{'instruction':'long sentence','response':'account'})]
    original=list(samples[0]['labels'])
    result=collate_sft_batch(samples,pad_token_id=0)
    assert set(result)=={'input_ids','attention_mask','labels'}
    assert result['labels'].dtype==torch.long
    assert result['input_ids'].shape==(2,max(len(s['input_ids']) for s in samples))
    assert torch.equal(result['attention_mask'][0,len(samples[0]['input_ids']):], torch.zeros(result['input_ids'].shape[1]-len(samples[0]['input_ids']),dtype=torch.long))
    assert torch.all(result['labels'][0,len(samples[0]['input_ids']):] == -100)
    assert samples[0]['labels']==original

def test_batch_bad_lengths():
    with pytest.raises(ValueError):
        collate_sft_batch([{'input_ids':[1,2],'attention_mask':[1], 'labels':[-100,2]}], pad_token_id=0)

def test_batch_empty():
    with pytest.raises(ValueError): collate_sft_batch([],pad_token_id=0)

def test_disjoint():
    train=[{'instruction':'ABC', 'response':'a'}]
    with pytest.raises(ValueError): ensure_disjoint(train,[{'instruction':' abc ', 'response':'b'}])
    ensure_disjoint(train,[{'instruction':'def','response':'b'}])

def test_examples_in_project_are_disjoint():
    from pathlib import Path
    base=Path(__file__).resolve().parents[1]
    train=read_records(base/'data/train.jsonl')
    valid=read_records(base/'data/valid.jsonl')
    assert len(train)>=8 and len(valid)>=2
    ensure_disjoint(train,valid)


def test_batch_rejects_fully_masked_example():
    with pytest.raises(ValueError):
        collate_sft_batch([{'input_ids':[1,2], 'attention_mask':[1,1], 'labels':[-100,-100]}],pad_token_id=0)
