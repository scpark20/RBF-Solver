"""Finite actual-model checks; validation outputs never count as experiment samples."""
import argparse,json,time
from pathlib import Path
from paths import RESULTS
import torch
from sampling import load_model,sample
from experiment import load_plan,atomic_json
a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,required=True);args=a.parse_args()
p=load_plan();device=torch.device('cuda:0')
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
state=RESULTS/'comparison'/f'gpu{args.gpu}.json'
atomic_json(state,dict(gpu=args.gpu,state='validating',message='실제 RF 모델 연결 검증',updated_at=time.time()))
model=load_model(p['checkpoint'],device)
z=torch.randn(1,3,32,32,generator=torch.Generator().manual_seed(p['seed'])).to(device)
labels=torch.zeros(1,device=device,dtype=torch.long)
method='dpmpp' if args.gpu==0 else 'unipc'
rows=[]
for n in ([5,40] if args.gpu==0 else [5,200]):
    out,calls=sample(model,z,labels,method,n,cfg=1.,order=3,t_start=1.,
        t_end=1e-3 if n<=10 else 1e-4,skip_type='logSNR',precision='fp32')
    assert calls==n and torch.isfinite(out).all()
    rows.append(dict(method=method,nfe=n,actual_nfe=calls,min=float(out.min()),max=float(out.max())))
report=dict(passed=True,gpu=args.gpu,rows=rows,scope=__doc__)
atomic_json(RESULTS/f'validation_data/model_connection_gpu{args.gpu}.json',report)
atomic_json(state,dict(gpu=args.gpu,state='not_started',message='모델 연결 검증 통과·실험 실행 전',updated_at=time.time()))
print(json.dumps(report))
