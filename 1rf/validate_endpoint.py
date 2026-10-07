"""Verify the approved VP-to-native-RF bridge without running experiments."""
import hashlib,json,torch
from paths import ROOT,RESULTS
from sampling import sample,time_mapping,vp_schedule
from experiment import load_plan,atomic_json
from rbf_training_run import load_training_plan
from rbf_shape_training import LinearFMTrainingSchedule,FMShapeTrainer
from rbf_solver_fm import RBF_FM_Solver
from vendor.samplers.rbf_solver import RBFSolver
from vendor.samplers.dpm_solver import DPM_Solver
from vendor.samplers.uni_pc import UniPC

p=load_plan();training=load_training_plan()
rows=[]
for n in p['nfes']+[200]:
    mp=time_mapping(n,1.,1e-3 if n<=10 else 1e-4)
    grid=torch.tensor(mp['flow_timesteps'])
    z=torch.tensor([0.25,-0.5,1.25]).reshape(1,3,1,1)
    vel=torch.tensor([0.125,-0.25,0.5]).reshape_as(z)
    label=torch.zeros(1,dtype=torch.long)
    expected=(z/mp['initial_state_scale']+(grid[-1]-grid[0])*vel)*mp['terminal_state_scale']
    for method in p['methods']:
        out,calls=sample(lambda x,s,y:vel.expand_as(x),z,label,method,n,cfg=1.,order=3,
                         t_start=mp['t_start'],t_end=mp['t_end'],skip_type='logSNR')
        torch.testing.assert_close(out,expected,rtol=2e-5,atol=2e-5)
        rows.append(dict(method=method,nfe=n,calls=calls,finite=True,max_error=float((out-expected).abs().max())))
    rbf=RBF_FM_Solver(lambda x,s:vel.expand_as(x),timesteps=grid,history_size=3,
                      gamma_pred=float('inf'),gamma_corr=float('inf'),lower_order_final=True)
    out=rbf.sample(z/mp['initial_state_scale'])*mp['terminal_state_scale']
    torch.testing.assert_close(out,expected,rtol=2e-5,atol=2e-5)
    assert rbf.last_nfe==n
    rows.append(dict(method='rbf_adams_validation',nfe=n,calls=rbf.last_nfe,finite=True,max_error=float((out-expected).abs().max())))
    # Original VP forward targets expressed in native RF units.
    ns=vp_schedule();vp=torch.tensor(mp['vp_timesteps'])
    a=ns.marginal_alpha(vp);b=ns.marginal_std(vp);k=a+b
    rf_target=grid[:,None]*z.flatten()+(1-grid[:,None])*vel.flatten()
    vp_target=(a[:,None]*z.flatten()+b[:,None]*vel.flatten())/k[:,None]
    torch.testing.assert_close(rf_target,vp_target)
    fm=LinearFMTrainingSchedule(mp['initial_state_scale'])
    torch.testing.assert_close(fm.marginal_std(grid)[:,None]*(vel.flatten()/mp['initial_state_scale']), (1-grid[:,None])*vel.flatten())

# Cross-check direct x0 wrapper against original epsilon-to-x0 conversion.
mp=time_mapping(8,1.,1e-3);ns=vp_schedule()
def velocity(x,s,y):return .125*x+.25
def epsilon(x,t):
    a=ns.marginal_alpha(t);b=ns.marginal_std(t);k=a+b;s=a/k
    y=x/k.reshape(-1,1,1,1)
    return y-s.reshape(-1,1,1,1)*velocity(y,s,None)
for method in p['methods']:
    solver=DPM_Solver(epsilon,ns,algorithm_type='dpmsolver++') if method=='dpmpp' else UniPC(epsilon,ns,algorithm_type='data_prediction',variant='bh1')
    solver.get_time_steps=lambda skip_type,t_T,t_0,N,device:torch.tensor(mp['vp_timesteps'],device=device)
    original=solver.sample(z,steps=8,t_start=1.,t_end=1e-3,order=3,skip_type='logSNR',method='multistep',lower_order_final=True,denoise_to_zero=False)
    bridge,calls=sample(velocity,z,label,method,8,cfg=1.,order=3,t_start=1.,t_end=1e-3,skip_type='logSNR')
    torch.testing.assert_close(original,bridge,rtol=2e-5,atol=2e-5)
assert FMShapeTrainer.sample_with_optim is RBFSolver.sample_with_optim
report=dict(passed=True,rows=rows,variant='bh1',order=3,time_policy=p['time_policy'],
            direct_x0_matches_original_epsilon_conversion=True,shared_initial_and_terminal_units=True,
            original_optimizer_reused=True,source_sha256=hashlib.sha256((ROOT/'sampling.py').read_bytes()).hexdigest(),
            scope='Analytic constant flow; VP/RF target and input/output units; synthetic affine epsilon bridge. Real model checks are separate.')
atomic_json(RESULTS/'endpoint_validation.json',report)
print(json.dumps(dict(passed=True,cases=len(rows),max_error=max(r['max_error'] for r in rows))))
