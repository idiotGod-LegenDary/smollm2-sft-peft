import math
from types import SimpleNamespace
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from sft_lab.engine import train_one_step, build_lora_model

class TinyCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(40,8)
        self.proj=nn.Linear(8,40)
    def forward(self,input_ids,attention_mask,labels):
        logits=self.proj(self.emb(input_ids))
        loss=F.cross_entropy(logits[:, :-1].reshape(-1,40),labels[:, 1:].reshape(-1),ignore_index=-100)
        return SimpleNamespace(loss=loss,logits=logits)

def test_one_step_updates_weights():
    torch.manual_seed(3)
    model=TinyCausalLM()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-2)
    before=model.proj.weight.detach().clone()
    batch={'input_ids':torch.tensor([[2,3,4,5]]),
           'attention_mask':torch.ones((1,4),dtype=torch.long),
           'labels':torch.tensor([[-100,-100,4,5]])}
    value=train_one_step(model,opt,batch,device='cpu')
    assert isinstance(value,float) and math.isfinite(value)
    assert not torch.equal(model.proj.weight.detach(),before)
    assert batch['input_ids'].dtype==torch.long

def test_one_step_all_masked_rejected():
    model=TinyCausalLM()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-2)
    b={'input_ids':torch.tensor([[2,3]]),
       'attention_mask':torch.ones((1,2),dtype=torch.long),
       'labels':torch.tensor([[-100,-100]])}
    with pytest.raises(ValueError): train_one_step(model,opt,b,device='cpu')

def test_lora_reject_invalid_rank():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__();self.q_proj=nn.Linear(8,8);self.v_proj=nn.Linear(8,8)
    with pytest.raises(ValueError):build_lora_model(Dummy(),rank=0)

def test_lora_reject_missing_modules():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__();self.other=nn.Linear(8,8)
    with pytest.raises(ValueError):build_lora_model(Dummy(),rank=2)

def test_build_lora_uses_peft_config_and_freezes_base(monkeypatch):
    """不联网：模拟 PEFT 接口，验证用户确实按契约传递关键参数。"""
    import sys
    import types
    module = types.ModuleType('peft')
    received = {}
    class FakeLoraConfig:
        def __init__(self,**kwargs):
            received.update(kwargs)
    class FakePeftModel(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_model=base
            base.requires_grad_(False)
            self.lora_A=nn.Parameter(torch.ones(2,2))
    def fake_get_peft_model(base,config):
        assert isinstance(config,FakeLoraConfig)
        return FakePeftModel(base)
    module.LoraConfig=FakeLoraConfig
    module.TaskType=types.SimpleNamespace(CAUSAL_LM='CAUSAL_LM')
    module.get_peft_model=fake_get_peft_model
    monkeypatch.setitem(sys.modules,'peft',module)
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj=nn.Linear(8,8)
            self.v_proj=nn.Linear(8,8)
    base=Dummy()
    result=build_lora_model(base,rank=2,alpha=4.0,dropout=0.0)
    assert received['r']==2 and received['lora_alpha']==4.0
    assert received['target_modules']==['q_proj','v_proj']
    assert received['task_type']=='CAUSAL_LM' and received['bias']=='none'
    assert isinstance(result,nn.Module)
    assert [n for n,p in result.named_parameters() if p.requires_grad]==['lora_A']
