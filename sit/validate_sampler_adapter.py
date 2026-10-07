"""Validate exact ImageNet256 solvers with a SiT flow adapter, without generating images."""
from paths import ROOT,RESULTS
from experiment import atomic_json
import torch,hashlib,json
from sampling import sample
from diffusers import DPMSolverMultistepScheduler,UniPCMultistepScheduler
from pathlib import Path
class Field:
    def __init__(self,affine=False): self.affine=affine
    def __call__(self,x,t,y):
        v=torch.ones_like(x)
        if self.affine: v=v+x/4+t[:,None,None,None]/2
        return v
# Start is derived from the public flow DPM scheduler, not selected as a tolerance.
s=DPMSolverMultistepScheduler(use_flow_sigmas=True,prediction_type='flow_prediction',flow_shift=1.0)
s.set_timesteps(5)
start=1-1/s.config.num_train_timesteps
assert abs(float(s.sigmas[0])-start)<torch.finfo(torch.float32).eps
# End is the exact default in the referenced ImageNet256 solver (total_N=1000).
end=1/1000
rows=[]
for affine in (False,True):
    field=Field(affine)
    for method,cls in [('dpmpp',DPMSolverMultistepScheduler),('unipc',UniPCMultistepScheduler)]:
        for nfe in (5,6,8,10):
            z=torch.ones(2,4,4,4); labels=torch.zeros(2,dtype=torch.long)
            actual,calls=sample(field,z,labels,method,nfe,cfg=1,order=2,t_start=start,t_end=end)
            options=dict(solver_order=2,prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=1.0,lower_order_final=True)
            if method=='dpmpp': options.update(algorithm_type='dpmsolver++',solver_type='midpoint')
            else: options.update(solver_type='bh2',predict_x0=True)
            ref=cls(**options); ref.set_timesteps(nfe)
            # Equal continuous grid isolates solver and flow-conversion equivalence.
            ref.sigmas=torch.linspace(start,end,nfe+1)
            ref.timesteps=torch.arange(nfe-1,-1,-1)
            x=z.clone()
            for k,timestep in enumerate(ref.timesteps):
                sit_time=(1-ref.sigmas[k]).expand(len(x))
                x=ref.step(-field(x,sit_time,labels),timestep,x,return_dict=False)[0]
            err=float((actual-x).abs().max())
            # FP32 cancellation bound for epsilon -> x0 conversion at alpha_min.
            bound=float(torch.finfo(torch.float32).eps/(1-start)*nfe)
            assert err<=bound,(method,nfe,affine,err,bound)
            if not affine:
                exact=z+(start-end)
                assert float((actual-exact).abs().max())<=bound
            assert calls==nfe and torch.isfinite(actual).all()
            rows.append(dict(method=method,nfe=nfe,field='affine' if affine else 'constant',max_abs=err,bound=bound))
hashes={}
for n in ['rbf_solver.py','uni_pc.py','dpm_solver.py']:
    a=ROOT.parent/'guided-diffusion/samplers'/n; b=ROOT/'vendor/samplers'/n
    assert a.read_bytes()==b.read_bytes()
    hashes[n]=hashlib.sha256(b.read_bytes()).hexdigest()
report=dict(passed=True,t_start=start,t_end=end,rows=rows,source_hashes=hashes,
            start_source='Diffusers 0.38.0 DPMSolverMultistepScheduler.set_timesteps use_flow_sigmas: 1-1/num_train_timesteps; default num_train_timesteps=1000.',
            end_source='RBF-Solver guided-diffusion/samplers defaults: 1/total_N; ImageNet256 total_N=1000.',
            scope='Constant-flow analytic solution and equal-grid affine-flow parity; no claim of optimal SiT truncation or FID.')
atomic_json(RESULTS/'sampler_adapter_validation.json',report)
print(json.dumps(report,indent=2))
