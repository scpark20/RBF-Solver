"""Shared experiment configuration; no implicit scientific parameter choices."""
import json, hashlib
from pathlib import Path
from paths import ROOT, RESULTS
PLAN=ROOT/'experiment.json'

def load_plan(require_ready=False):
    p=json.loads(PLAN.read_text())
    assert p['gpus'] and len(set(p['gpus']))==len(p['gpus']) and all(isinstance(g,int) and g>=0 for g in p['gpus']), 'Use unique visible GPU indices'
    assert p['cfgs']==[1.0] and p['conditioning']=='unconditional'
    assert p['methods']==['dpmpp','unipc'] and p['nfes']==[5,6,8,10,12,15,20,25,30,35,40]
    out=Path(p['output']).resolve()
    if not out.is_relative_to(RESULTS.resolve()) or out==RESULTS.resolve():
        raise ValueError('Experiment outputs must be below /data/RBF-Solver/results')
    validate_time_plan(p)
    if require_ready:
        if not p['ready'] or p['pending']: raise ValueError('Experiment settings are awaiting confirmation')
        assert p['num_samples']==50000 and p['order']==3
        assert p['t_start']==1.0 and p['t_end']==0.001 and p['skip_type']=='logSNR'
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


def validate_time_plan(p):
    """Check the approved source policy, each NFE grid and both coordinate scales."""
    from sampling import time_mapping
    if p['time_policy']!='project_cifar10_vp_logsnr_with_native_rf_wrapper':
        raise ValueError('Approved CIFAR-10 time policy required')
    for n in p['nfes']:
        expected=time_mapping(n,1.0,1e-3 if n<=10 else 1e-4)
        if p['sampling_times'].get(str(n))!=expected:
            raise ValueError(f'Time mapping mismatch at NFE {n}')
        if p['flow_timesteps'].get(str(n))!=expected['flow_timesteps']:
            raise ValueError(f'RBF grid mismatch at NFE {n}')
    if 'teacher' in p:
        expected=time_mapping(200,1.0,1e-4)
        if p.get('teacher_time_mapping')!=expected or p['time_grid']!='logSNR':
            raise ValueError('Teacher time mapping mismatch')
    elif p['unipc_variant']!='bh1' or p['skip_type']!='logSNR':
        raise ValueError('Original CIFAR-10 BH1/logSNR settings required')
