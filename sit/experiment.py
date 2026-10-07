"""Shared experiment configuration; no implicit scientific parameter choices."""
import json, hashlib
from pathlib import Path
from paths import ROOT, RESULTS
PLAN=ROOT/'experiment.json'

def load_plan(require_ready=False):
    p=json.loads(PLAN.read_text())
    assert p['gpus'] and len(set(p['gpus']))==len(p['gpus']) and all(isinstance(g,int) and g>=0 for g in p['gpus']), 'Use unique visible GPU indices'
    assert p['cfgs']==[1.5,3.5,5.5,7.5]
    assert p['methods']==['dpmpp','unipc'] and p['nfes']==[5,6,8,10]
    out=Path(p['output']).resolve()
    if not out.is_relative_to(RESULTS.resolve()) or out==RESULTS.resolve():
        raise ValueError('Experiment outputs must be below /data/RBF-Solver/results')
    if require_ready:
        if not p['ready'] or p['pending']: raise ValueError('Experiment settings are awaiting confirmation')
        assert p['num_samples']>=2 and p['order'] in (2,3)
        assert 0<p['t_end']<p['t_start']<=1
    return p

def job_id(method,cfg,nfe):
    return f'cfg_{cfg:g}/{method}_nfe{nfe}'

def jobs(p):
    return [dict(method=m,cfg=c,nfe=n,id=job_id(m,c,n)) for c in p['cfgs'] for n in p['nfes'] for m in p['methods']]

def plan_digest(p):
    return hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n'); tmp.replace(path)
