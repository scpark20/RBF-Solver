"""Prepare CIFAR-10 reference moments using the existing CUDA accumulator."""
from pathlib import Path
import json
import numpy as np
import torch
from paths import ROOT
from experiment import load_plan,atomic_json
from benchmark import digest
from fid_gpu import Moments

@torch.inference_mode()
def main():
    p=load_plan();source=Path(p['reference_source']);target=Path(p['reference'])
    source_hash=digest(source)
    with np.load(source,allow_pickle=False) as archive:
        values=archive['pool_3']
        if values.shape!=(50000,2048) or not np.isfinite(values).all():
            raise ValueError('Expected 50000 finite official CIFAR-10 pool_3 features')
        moments=Moments('cuda:0')
        moments.update(torch.as_tensor(values,device='cuda:0'))
        mu,cov=moments.statistics()
    tmp=target.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f,mu=mu.cpu().numpy(),sigma=cov.cpu().numpy(),n=moments.n)
    tmp.replace(target)
    record=dict(source=str(source),source_sha256=source_hash,reference=str(target),reference_sha256=digest(target),n=moments.n,extractor='Official CIFAR-10 cached pool_3; ADM-compatible feature parity checked separately',moments='Existing SiT Moments, float64 CUDA, unbiased sample covariance')
    atomic_json(target.with_suffix('.json'),record)
    print(json.dumps(record),flush=True)

if __name__=='__main__':main()
