"""Prepare independent new targets while preserving NFE6 optimization/evaluation draws."""
import os
os.environ['SIT_SAMPLING_PLAN']='unconditional_retrain_sampling.json'
os.environ['SIT_TRAINING_PLAN']='unconditional_retrain_training.json'
import json,hashlib,time
from pathlib import Path
import torch
from paths import ROOT
from benchmark import digest
from experiment import atomic_json
from rbf_training_run import load_training_plan,validate_inputs,make_inputs,atomic_torch
from rbf_pipeline import prepare
from rbf_followup import plan


def main():
    p=load_training_plan(True);sp=plan();out=Path(p['output'])
    previous=json.loads((ROOT/'unconditional_training.json').read_text());oldout=Path(previous['output'])
    oldbank_path=oldout/'inputs.pt';oldbank=torch.load(oldbank_path,map_location='cpu',weights_only=True)
    validate_inputs(oldbank,previous)
    allowed={'name','nfes','output','sources','retrain_of'}
    assert {k:v for k,v in previous.items() if k not in allowed}=={k:v for k,v in p.items() if k not in allowed}
    gen=torch.Generator(device='cpu').manual_seed(previous['seed'])
    first_noise=torch.randn(previous['target_pairs'],4,32,32,generator=gen)
    first_labels=torch.randint(0,1000,(previous['target_pairs'],),generator=gen)
    assert torch.equal(first_noise,oldbank['noise']) and torch.equal(first_labels,oldbank['labels'])
    state=gen.get_state().clone()
    new_noise=torch.randn(p['target_pairs'],4,32,32,generator=gen)
    new_labels=torch.randint(0,1000,(p['target_pairs'],),generator=gen)
    def rows_hash(x):return [hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in x]
    assert not set(rows_hash(new_noise))&set(rows_hash(first_noise))
    assert len(set(rows_hash(new_noise)))==p['target_pairs']
    n=previous['nfes'].index(6);newbank=make_inputs(p)
    newbank.update(noise=new_noise,labels=new_labels,indices=oldbank['indices'][:,n:n+1].clone())
    validate_inputs(newbank,p)
    assert torch.equal(newbank['indices'][:,0],oldbank['indices'][:,n])
    protected=[oldbank_path,oldout/'cfg_0/teacher/targets.pt',ROOT/'unconditional_sampling.json',ROOT/'unconditional_training.json',Path(sp['retrain_of'])]
    protected+=list((oldout/'cfg_0/nfe_6').glob('*'))
    hashes={str(f):digest(f) for f in protected if f.is_file()}
    prepare(p)
    receipt=out/'input_provenance.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['protected_files_sha256']==hashes
        actual=torch.load(out/'inputs.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(actual[k],newbank[k]) for k in ['noise','labels','indices'])
    else:
        atomic_torch(out/'inputs.pt',newbank)
        atomic_json(receipt,dict(created_at=time.time(),seed=1234,original_input=str(oldbank_path),original_input_sha256=digest(oldbank_path),rng_boundary='After original torch.randn(128,4,32,32), then torch.randint(0,1000,(128,)); next exact calls generate new block',rng_boundary_sha256=hashlib.sha256(state.numpy().tobytes()).hexdigest(),new_noise_sha256=rows_hash(new_noise),no_exact_noise_overlap=True,optimization_indices='Exact original NFE6 20x16 replacement indices, unchanged',evaluation_input=sp['input_bank'],evaluation_input_sha256=digest(sp['input_bank']),protected_files_sha256=hashes))
    assert all(digest(Path(f))==h for f,h in hashes.items())
    print(json.dumps(dict(ready=True,new_target_noise_count=128,same_optimization_indices=True,same_evaluation_bank=True,new_output=str(out))))

if __name__=='__main__':main()
