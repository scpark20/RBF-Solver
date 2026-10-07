"""Check the original optimizer, saved shapes and native RF replay on the approved VP grid."""
import torch
from paths import RESULTS
from experiment import atomic_json
from rbf_training_run import load_training_plan,read_shapes,load_trained_solver
from rbf_shape_training import FMShapeTrainer
from vendor.samplers.rbf_solver import RBFSolver
p=load_training_plan(False);n=p['nfes'][0];mp=p['sampling_times'][str(n)]
times=torch.tensor(mp['flow_timesteps'])
out=RESULTS/'validation_data/approved_grid_optimizer';out.mkdir(parents=True,exist_ok=True)
noise=torch.zeros(1,3,1,1);target=torch.ones_like(noise);calls=0
def velocity(x,t):
    global calls
    calls+=1
    return torch.ones_like(x)
assert FMShapeTrainer.sample_with_optim is RBFSolver.sample_with_optim
trainer=FMShapeTrainer(velocity,out,timesteps=times,initial_state_scale=mp['initial_state_scale'],
    log_shape_min=p['log_gamma_min'],log_shape_max=p['log_gamma_max'],log_shape_num=p['log_gamma_points'])
actual=trainer.sample_with_optim(noise/mp['initial_state_scale'],target,steps=n,
    t_start=times[0].item(),t_end=times[-1].item(),order=p['history_size'],
    skip_type=p['time_grid'],lower_order_final=p['lower_order_final'])
assert calls==n
expected=torch.full_like(noise,float(times[-1]-times[0]))
torch.testing.assert_close(actual,expected)
shape=out/f'NFE={n},p={p["history_size"]}.npz';read_shapes(shape,n,p,with_losses=True)
solver=load_trained_solver(lambda x,t:torch.ones_like(x),shape,n,p)
replayed=solver.sample(noise/mp['initial_state_scale'])*mp['terminal_state_scale']
assert solver.last_nfe==n
torch.testing.assert_close(replayed,expected*mp['terminal_state_scale'])
report=dict(passed=True,original_optimizer_reused=True,actual_optimizer_nfe=calls,
    actual_replay_nfe=solver.last_nfe,history_size=p['history_size'],time_policy=p['time_policy'],
    grid_points=len(times),constant_flow_exact_within_float32=True,scope=__doc__)
atomic_json(RESULTS/'reused_optimizer_validation.json',report);print(report)
