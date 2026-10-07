"""Official 1-RF linear-flow adapter for the repository's original multistep solvers."""
import sys
from pathlib import Path
from paths import ROOT
import torch
sys.path.insert(0,str(ROOT/'vendor/python'))
sys.path.insert(0,str(ROOT/'vendor/RectifiedFlow/ImageGeneration'))
from vendor.samplers.dpm_solver import DPM_Solver
from vendor.samplers.uni_pc import UniPC
from vendor.samplers.utils import NoiseScheduleVP

class LinearFlowSchedule:
    # Solver time decreases 1 -> 0; RF time increases 0 -> 1.
    T, total_N, schedule = 1.0, 1000, 'linear_flow'
    def marginal_alpha(self,t): return 1-t
    def marginal_std(self,t): return t
    def marginal_log_mean_coeff(self,t): return torch.log1p(-t)
    def marginal_lambda(self,t): return torch.log1p(-t)-torch.log(t)
    def inverse_lambda(self,l): return torch.sigmoid(-l)

def load_model(checkpoint, device):
    """Load the official 1-RF architecture and EMA; expose physical RF time."""
    import types
    from configs.rectified_flow.cifar10_rf_gaussian_ddpmpp import get_config
    config=get_config()
    if config.model.fir is not False:
        raise RuntimeError('The official checkpoint requires fir=False for this loader')
    # The official FIR module imports CUDA build tools eagerly. This architecture
    # never calls FIR: refuse such a call rather than approximate its computation.
    if 'op' not in sys.modules:
        unused=types.ModuleType('op')
        def forbidden_fir(*args,**kwargs):
            raise RuntimeError('Unexpected FIR call in the official fir=False model')
        unused.upfirdn2d=forbidden_fir
        sys.modules['op']=unused
    from models.ncsnpp import NCSNpp
    from models.ema import ExponentialMovingAverage
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    raw=NCSNpp(config).eval()
    weights={k.removeprefix('module.'):v for k,v in state['model'].items()}
    raw.load_state_dict(weights,strict=True)
    parameters=list(raw.parameters())
    shadows=state['ema']['shadow_params']
    if len(parameters)!=len(shadows) or any(a.shape!=b.shape for a,b in zip(parameters,shadows)):
        raise ValueError('Official EMA parameter count/shape mismatch')
    ema=ExponentialMovingAverage(parameters,decay=config.model.ema_rate)
    ema.load_state_dict(state['ema']);ema.copy_to(parameters)
    step=int(state['step'])
    del state,weights,shadows,ema,parameters
    class PhysicalRF(torch.nn.Module):
        def __init__(self,model):
            super().__init__();self.model=model;self.training_step=step
        def forward(self,x,s,labels=None):
            if labels is not None and (labels!=0).any():
                raise ValueError('Unconditional model accepts only the zero condition sentinel')
            return self.model(x,s.reshape(-1).expand(x.shape[0])*999)
    return PhysicalRF(raw).eval().to(device).requires_grad_(False)

def vp_schedule():
    """Reuse the exact linear VP schedule configured by project CIFAR-10."""
    return NoiseScheduleVP('linear',continuous_beta_0=0.1,continuous_beta_1=20.0)

def time_mapping(nfe, t_start, t_end):
    """Build the source VP log-SNR grid and its algebraic native-RF coordinates."""
    ns=vp_schedule()
    source=DPM_Solver(lambda x,t:x,ns,algorithm_type='dpmsolver++')
    times=source.get_time_steps('logSNR',t_start,t_end,nfe,torch.device('cpu'))
    alpha=ns.marginal_alpha(times);sigma=ns.marginal_std(times)
    scale=alpha+sigma
    flow=alpha/scale
    if not torch.isfinite(flow).all() or not (flow[1:]>flow[:-1]).all():
        raise ValueError('Invalid VP-to-RF grid')
    return dict(t_start=t_start,t_end=t_end,vp_timesteps=times.tolist(),
                flow_timesteps=flow.tolist(),state_scales=scale.tolist(),
                flow_time_start=flow[0].item(),flow_time_end=flow[-1].item(),
                initial_state_scale=scale[0].item(),terminal_state_scale=scale[-1].item())

@torch.inference_mode()
def sample(model, noise, labels, method, nfe, *, cfg, order,
           t_start, t_end, skip_type, precision='fp32'):
    if cfg!=1.0 or not (labels==0).all(): raise ValueError('Unconditional RF input required')
    if skip_type!='logSNR': raise ValueError('The approved CIFAR-10 grid is logSNR')
    mapping=time_mapping(nfe,t_start,t_end)
    schedule=vp_schedule()
    calls=0
    def data_prediction(x,t):
        nonlocal calls
        calls+=1
        alpha=schedule.marginal_alpha(t);sigma=schedule.marginal_std(t)
        scale=alpha+sigma;s=alpha/scale
        native=x/scale.reshape(-1,1,1,1)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
            velocity=model(native,s.reshape(-1).expand(x.shape[0]),labels)
        return native+(1-s).reshape(-1,1,1,1)*velocity.float()
    if method=='dpmpp':
        solver=DPM_Solver(data_prediction,schedule,algorithm_type='dpmsolver++')
    elif method=='unipc':
        solver=UniPC(data_prediction,schedule,algorithm_type='data_prediction',variant='bh1')
    else: raise ValueError(method)
    solver.data_prediction_fn=solver.model
    # Source-generated CPU FP32 grid is shared verbatim with native RF RBF.
    # This avoids independently rounded grids across CPU/GPU implementations.
    times=torch.tensor(mapping['vp_timesteps'],dtype=torch.float32,device=noise.device)
    def fixed_grid(skip_type,t_T,t_0,N,device):
        if skip_type!='logSNR' or N!=nfe or t_T!=t_start or t_0!=t_end:
            raise ValueError('Sampler grid request changed')
        return times.to(device)
    solver.get_time_steps=fixed_grid
    out=solver.sample(noise,steps=nfe,order=order,method='multistep',
                      skip_type=skip_type,t_start=t_start,t_end=t_end,
                      lower_order_final=True,denoise_to_zero=False)
    if calls!=nfe: raise AssertionError(f'NFE mismatch: {calls} != {nfe}')
    if not torch.isfinite(out).all(): raise FloatingPointError('Nonfinite sample')
    return out,calls
