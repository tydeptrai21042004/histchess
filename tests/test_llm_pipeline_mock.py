"""Exercises batched HuggingFace orchestration with a tiny offline fake model.

These checks do not assert the behavior of genuine SmolLM2 weights or HF access.
"""
import json
import sys
import types
import pytest

from histchess.core import generate
from histchess.experiments import llm_experiment
from histchess.paper import read_csv


@pytest.mark.slow
def test_llm_batching_fixed_matched_pairs_and_invalids(monkeypatch,tmp_path):
    import torch
    pairs,_=generate(40,seed=3101)
    class MockTokenizer:
        pad_token_id=1
        eos_token='!'
        chat_template='fake'
        padding_side='left'
        @staticmethod
        def from_pretrained(*args,**kwargs):return MockTokenizer()
        def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
            return messages[0]['content']+'\nassistant:'
        def __call__(self,prompts,return_tensors,padding,truncation):
            length=max(2,max(len(s)//20 for s in prompts))
            return {'input_ids':torch.ones((len(prompts),length),dtype=torch.long),
                    'attention_mask':torch.ones((len(prompts),length),dtype=torch.long)}
        def batch_decode(self,ids,skip_special_tokens):
            return ['YES' if (j % 3 == 0) else 'NO' if (j % 3 == 1) else 'unknown'
                    for j in range(ids.shape[0])]
    class MockModel:
        config=types.SimpleNamespace(max_position_embeddings=2048)
        @staticmethod
        def from_pretrained(*args,**kwargs):return MockModel()
        def to(self,device):return self
        def eval(self):return self
        def generate(self,**kwargs):
            x=kwargs['input_ids']
            return torch.cat([x,torch.ones((x.shape[0],3),dtype=x.dtype,device=x.device)],dim=1)
    monkeypatch.setitem(sys.modules,'transformers',types.SimpleNamespace(
        AutoTokenizer=MockTokenizer,AutoModelForCausalLM=MockModel))
    llm_experiment(pairs,tmp_path,models=['fake-1','fake-2'],pairs_per_task=2,
                   seed=3101,batch_size=3,require_gpu=False)
    sample=json.loads((tmp_path/'llm_sample.json').read_text())
    assert len(sample['pair_ids'])==8
    records=read_csv(tmp_path/'llm_predictions.csv')
    assert len(records)==32
    for model in ('fake-1','fake-2'):
        assert {(r['pair_id'],r['variant']) for r in records if r['model']==model}=={
            (id_,v) for id_ in sample['pair_ids'] for v in ('positive','negative')}
    assert any(r['valid']=='False' for r in records)
    assert (tmp_path/'llm_metrics.csv').stat().st_size>0
