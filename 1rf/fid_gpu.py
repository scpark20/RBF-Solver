"""GPU Inception features and float64 streaming FID moments; no image I/O."""
import os
from pathlib import Path
import numpy as np
import torch
from paths import ROOT,WEIGHTS
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3

def extractor(device):
    return FeatureExtractorInceptionV3('inception-v3-compat',['2048']).eval().to(device)

class Moments:
    def __init__(self,device,dim=2048):
        self.n=0
        self.sum=torch.zeros(dim,device=device,dtype=torch.float64)
        self.cross=torch.zeros(dim,dim,device=device,dtype=torch.float64)
    def update(self,features):
        x=features.double()
        self.n+=len(x); self.sum+=x.sum(0); self.cross+=x.T@x
    def statistics(self):
        if self.n<2: raise ValueError('At least two images required')
        mean=self.sum/self.n
        cov=(self.cross-self.n*torch.outer(mean,mean))/(self.n-1)
        return mean,(cov+cov.T)/2
    def state(self): return dict(n=self.n,sum=self.sum.cpu(),cross=self.cross.cpu())
    def restore(self,state):
        self.n=state['n']; self.sum.copy_(state['sum']); self.cross.copy_(state['cross'])

def frechet(mu,cov,ref_mu,ref_cov):
    """Symmetric PSD sandwich avoids nonsymmetric square-root instability."""
    d=mu.device
    ref_mu=torch.as_tensor(ref_mu,device=d,dtype=torch.float64)
    ref_cov=torch.as_tensor(ref_cov,device=d,dtype=torch.float64)
    ref_cov=(ref_cov+ref_cov.T)/2
    values,vectors=torch.linalg.eigh(ref_cov)
    if values.min() < -1e-6: raise ValueError('Reference covariance not PSD')
    root=(vectors*values.clamp_min(0).sqrt())@vectors.T
    middle=root@cov@root
    ev=torch.linalg.eigvalsh((middle+middle.T)/2)
    value=(mu-ref_mu).square().sum()+cov.trace()+ref_cov.trace()-2*ev.clamp_min(0).sqrt().sum()
    if value < -1e-5: raise ValueError('Numerically invalid negative FID')
    return value.clamp_min(0).item()
