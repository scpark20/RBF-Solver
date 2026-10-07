"""Reuse verified CFG-0 targets for the requested additional NFEs; preserve originals."""
import os
os.environ['SIT_SAMPLING_PLAN']='unconditional_extended_sampling.json'
os.environ['SIT_TRAINING_PLAN']='unconditional_extended_training.json'
from pathlib import Path
import ast,json,time,hashlib
import torch
from paths import ROOT
from experiment import atomic_json,plan_digest
from benchmark import digest
from rbf_training_run import load_training_plan,protocol_hash,source_hashes,make_inputs,validate_inputs,atomic_torch
from rbf_pipeline import prepare,target_identity,load_targets
from rbf_followup import plan


def main():
    """Check numerical dependencies and retain original provenance in the import receipt."""
    p=load_training_plan(True);sp=plan();out=Path(p['output'])
    old=json.loads((ROOT/'unconditional_training.json').read_text());oldout=Path(old['output'])
    allowed={'name','nfes','output','sources'}
    assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
    previous_sampling=json.loads((ROOT/'unconditional_sampling.json').read_text())
    sample_allowed={'name','nfes','output','training_output','sources','batch_sources','batch_tuning'}
    assert {k:v for k,v in sp.items() if k not in sample_allowed}=={k:v for k,v in previous_sampling.items() if k not in sample_allowed}
    manifest=json.loads((oldout/'manifest.json').read_text())
    assert manifest['protocol_sha256']==protocol_hash(old) and manifest['protocol']==old
    current=source_hashes();previous=manifest['source_sha256']
    if previous!=current:
        assert all(current[k]==v for k,v in previous.items() if k!='rbf_training_run.py')
        backup=next(x for x in sorted((ROOT/'.history').glob('before_cfg0_nfe40_*/rbf_training_run.py')) if digest(x)==previous['rbf_training_run.py'])
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name!='load_training_plan'}
        assert functions(backup)==functions(ROOT/'rbf_training_run.py'), 'Numerical training implementation changed'
    bank_path=oldout/'inputs.pt';oldbank=torch.load(bank_path,map_location='cpu',weights_only=True);validate_inputs(oldbank,old)
    source_target=oldout/'cfg_0/teacher/targets.pt';record=torch.load(source_target,map_location='cpu',weights_only=True)
    expected=dict(protocol_sha256=protocol_hash(old),inputs_sha256=digest(bank_path),source_sha256=previous,cfg=0.0)
    assert record['identity']==expected and record['done']==p['target_pairs']
    target=record['samples'];assert target.shape==(128,4,32,32) and target.dtype==torch.float32 and torch.isfinite(target).all()
    oldfiles={str(x):digest(x) for x in [bank_path,source_target,oldout/'manifest.json',ROOT/'unconditional_training.json',ROOT/'unconditional_sampling.json']}
    for file in Path(previous_sampling['output']).glob('cfg_0/*/result.json'):
        r=json.loads(file.read_text());assert r['config']['n']==10000 and r['config']['plan_sha256']==plan_digest(previous_sampling)
        oldfiles[str(file)]=digest(file)
    assert len(list(Path(previous_sampling['output']).glob('cfg_0/*/result.json')))==12
    prepare(p)
    newbank=make_inputs(p);full=make_inputs(dict(p,nfes=old['nfes']+p['nfes']))
    assert torch.equal(full['indices'][:,:len(old['nfes'])],oldbank['indices'])
    newbank['indices']=full['indices'][:,len(old['nfes']):].clone()
    assert torch.equal(newbank['noise'],oldbank['noise']) and torch.equal(newbank['labels'],oldbank['labels'])
    validate_inputs(newbank,p)
    dest=out/'cfg_0/teacher/targets.pt'
    receipt=out/'target_import.json'
    if receipt.exists():
        prior=json.loads(receipt.read_text());assert prior['original_files_sha256']==oldfiles
        load_targets(p,0.0)
    else:
        atomic_torch(out/'inputs.pt',newbank)
        newmanifest=json.loads((out/'manifest.json').read_text())
        atomic_torch(dest,dict(identity=target_identity(p,newmanifest,out/'inputs.pt',0.0),done=p['target_pairs'],samples=target))
        loaded,_=load_targets(p,0.0);assert torch.equal(loaded,target)
        atomic_json(receipt,dict(source=str(source_target),source_sha256=digest(source_target),source_identity=record['identity'],destination=str(dest),destination_sha256=digest(dest),tensor_sha256=hashlib.sha256(target.numpy().tobytes()).hexdigest(),original_files_sha256=oldfiles,noise_and_labels_exact=True,old_repeat_indices_preserved=True,new_repeat_indices='Seed 1234 full 11-NFE draw bank, additional NFE slice 4:11',numerical_source_functions_unchanged=True,created_at=time.time()))
        atomic_json(dest.parent/'status.json',dict(state='complete',done=128,total=128,reused=True,source=str(source_target),updated_at=time.time()))
    assert all(digest(Path(f))==h for f,h in oldfiles.items()), 'Original artifacts changed'
    print(json.dumps(dict(target_pairs=128,reused=True,additional_nfes=p['nfes'],existing_results_preserved=12,receipt=str(receipt))))

if __name__=='__main__':main()
