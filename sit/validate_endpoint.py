"""CPU-only algebra and source-equivalence checks, no generated images."""
from paths import ROOT,RESULTS
import torch,json,hashlib
from sampling import sample,LinearFlowSchedule,EndpointUniPC
from vendor.samplers.dpm_solver import DPM_Solver
from vendor.samplers.uni_pc import UniPC
from experiment import atomic_json
class Field:
 def __init__(self,affine): self.affine=affine
 def __call__(self,x,t,y):
  return torch.ones_like(x)+(x/4+t[:,None,None,None]/2 if self.affine else 0)
rows=[]
for affine in [False,True]:
 for method in ['dpmpp','unipc']:
  for nfe in [5,6,8,10]:
   model=Field(affine); x=torch.ones(2,4,4,4); y=torch.zeros(2,dtype=torch.long)
   z,calls=sample(model,x,y,method,nfe,cfg=1,order=2,t_start=1.,t_end=.001)
   assert calls==nfe and torch.isfinite(z).all()
   if not affine: torch.testing.assert_close(z,x+1-.001)
   # Interior comparison: algebraic x0 adapter must agree with source noise adapter.
   # Interior value is a test input only, not a production endpoint.
   ns=LinearFlowSchedule()
   def eps(a,t): return a-(1-t)[:,None,None,None]*model(a,1-t,y)
   raw=(DPM_Solver(eps,ns,algorithm_type='dpmsolver++') if method=='dpmpp' else UniPC(eps,ns,variant='bh2'))
   interior=raw.sample(x,steps=nfe,order=2,t_start=.75,t_end=.001,lower_order_final=True,denoise_to_zero=False)
   stable,_=sample(model,x,y,method,nfe,cfg=1,order=2,t_start=.75,t_end=.001)
   torch.testing.assert_close(interior,stable)
   rows.append(dict(method=method,nfe=nfe,affine=affine,finite=True,calls=calls,interior_error=float((interior-stable).abs().max())))
# Verify the exact infinite-history coefficient limit from source R rho = b.
# Lambda perturbations below are synthetic test cases, not experimental settings.
errors=[]
for old_l in [-10.,-100.,-1000.]:
 class SyntheticSchedule(LinearFlowSchedule):
  def marginal_lambda(self,t):
   return torch.where(t==1,torch.full_like(t,old_l),super().marginal_lambda(t))
 ns=SyntheticSchedule(); t=torch.tensor([.4]); prev=[torch.tensor([1.]),torch.tensor([.7])]
 x=torch.ones(1,4,4,4); models=[x*2,x*3]
 fn=lambda x,t: x*.25
 raw=UniPC(fn,ns,variant='bh2'); raw.data_prediction_fn=raw.model
 z,_=raw.multistep_uni_pc_bh_update(x,models,prev,t,2)
 stable=EndpointUniPC(fn,LinearFlowSchedule(),variant='bh2'); stable.data_prediction_fn=stable.model
 lim,_=stable.multistep_uni_pc_bh_update(x,models,prev,t,2)
 errors.append(float((z-lim).abs().max()))
assert errors[2]<errors[1]<errors[0],errors
hashes={}
for n in ['rbf_solver.py','uni_pc.py','dpm_solver.py']:
 a=ROOT.parent/'guided-diffusion/samplers'/n; b=ROOT/'vendor/samplers'/n
 assert a.read_bytes()==b.read_bytes(); hashes[n]=hashlib.sha256(b.read_bytes()).hexdigest()
report=dict(passed=True,rows=rows,limit_convergence_errors=errors,source_hashes=hashes,
 t_start=1.,t_end=.001,scope='Algebraic SiT velocity-to-data conversion and exact source endpoint limit; not an FID claim',
 sources=['SiT/transport/path.py ICPlan alpha=s,sigma=1-s,velocity=data-noise','dpm_solver.py sample defaults T=1 and 1/total_N; continuous low-NFE recommendation 1e-3','uni_pc.py:641-734 BH2 coefficients and linear system'])
atomic_json(RESULTS/'endpoint_validation.json',report)
print(json.dumps(report,indent=2))
