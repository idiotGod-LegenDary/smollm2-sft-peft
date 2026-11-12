"""A线离线单元测试；不得为了通过测试而修改已实现的旧模块。"""
import pytest
import torch
from sft_lab.evaluation import (
    LABELS, normalize_prediction, validate_splits,
    classification_metrics, predict_one,
)


def samples(prefix):
    return [
        {"instruction": f"{prefix} billing example", "response": "billing"},
        {"instruction": f"{prefix} technical example", "response": "technical"},
        {"instruction": f"{prefix} account example", "response": "account"},
    ]


@pytest.mark.parametrize("raw,expected", [
    ("billing", "billing"), (" BILLING. ", "billing"),
    ("\naccount\n", "account"), ("Technical!", "technical"),
    ("billing because I think so", None), ("", None),
    ("not billing", None), ("technical/account", None),
])
def test_normalize(raw, expected):
    assert normalize_prediction(raw) == expected


def test_normalize_rejects_non_str():
    with pytest.raises(TypeError): normalize_prediction(None)


def test_split_ok():
    result = validate_splits(samples('train'), samples('valid'), samples('test'))
    assert result['sizes'] == {'train': 3, 'valid': 3, 'test': 3}
    assert result['class_counts']['test'] == dict.fromkeys(LABELS, 1)
    assert result['overlap'] == []


def test_split_cross_overlap():
    train = samples('train')
    test = samples('test')
    test[0] = {"instruction": "  TRAIN BILLING EXAMPLE ", "response": "billing"}
    with pytest.raises(ValueError): validate_splits(train, samples('valid'), test)


def test_split_duplicate_inside():
    train = samples('train'); train.append(train[0].copy())
    with pytest.raises(ValueError): validate_splits(train, samples('valid'), samples('test'))


def test_split_bad_labels():
    test = samples('test'); test[0]['response'] = 'other'
    with pytest.raises(ValueError): validate_splits(samples('train'), samples('valid'), test)


def test_split_missing_class():
    with pytest.raises(ValueError): validate_splits(samples('train'), samples('valid'), [samples('test')[0]])


def test_metrics_perfect():
    gold=list(LABELS); got=classification_metrics(gold, gold)
    assert got['accuracy'] == 1.0 and got['macro_f1'] == 1.0
    assert got['invalid_count'] == 0 and got['samples']==3
    assert got['confusion']['billing']['billing']==1


def test_metrics_invalid_counts_false_negative():
    result=classification_metrics(['billing', 'technical', 'account'], ['billing', None, 'billing'])
    assert result['accuracy']==pytest.approx(1/3)
    assert result['invalid_count']==1
    assert result['confusion']['technical']['INVALID']==1
    assert result['per_class']['technical']['recall']==0
    assert result['per_class']['billing']['precision']==pytest.approx(.5)


def test_metrics_all_invalid():
    m=classification_metrics(list(LABELS), [None]*3)
    assert m['accuracy']==0.0 and m['macro_f1']==0.0


def test_metrics_bad_shapes():
    with pytest.raises(ValueError): classification_metrics([], [])
    with pytest.raises(ValueError): classification_metrics(['billing'], [])
    with pytest.raises(ValueError): classification_metrics(['bad'], ['billing'])
    with pytest.raises(ValueError): classification_metrics(['billing'], ['bad'])


class FakeTokenizer:
    pad_token_id=0
    def __init__(self): self.calls=[]; self.decoded=None
    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages,kw))
        return [2,3,4]
    def decode(self, ids, skip_special_tokens):
        self.decoded=(list(ids),skip_special_tokens)
        return "billing"

class FakeModel:
    def __init__(self): self.kwargs=None; self.evaluated=False
    def eval(self): self.evaluated=True;return self
    def generate(self, **kw):
        self.kwargs=kw
        return torch.tensor([[2,3,4,77,78]], dtype=torch.long)


def test_predict_only_new_tokens():
    model=FakeModel(); tok=FakeTokenizer()
    out=predict_one(model,tok,"A payment was doubled")
    assert out=={'raw_output':'billing','prediction':'billing'}
    assert tok.decoded==([77,78],True)
    assert tok.calls[0][0]==[{'role':'user','content':'A payment was doubled'}]
    assert tok.calls[0][1]['add_generation_prompt'] is True
    assert model.kwargs['do_sample'] is False
    assert model.kwargs['max_new_tokens'] == 8
    assert model.kwargs['input_ids'].shape == (1,3)
    assert model.evaluated


def test_predict_empty_instruction():
    with pytest.raises(ValueError): predict_one(FakeModel(),FakeTokenizer(),'  ')


def test_predict_bad_steps():
    with pytest.raises(ValueError): predict_one(FakeModel(),FakeTokenizer(),'abc',max_new_tokens=0)
    with pytest.raises(ValueError): predict_one(FakeModel(),FakeTokenizer(),'abc',max_new_tokens=True)


def test_predict_no_new_token():
    class NoNew(FakeModel):
        def generate(self, **kw): return torch.tensor([[2,3,4]], dtype=torch.long)
    with pytest.raises(ValueError): predict_one(NoNew(), FakeTokenizer(), 'abc')
