"""SiT linear-flow adapter for the repository's original multistep solvers."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor' / 'SiT'))
from models import SiT_models
from vendor.samplers.dpm_solver import DPM_Solver
from vendor.samplers.uni_pc import UniPC

class LinearFlowSchedule:
    # Solver time decreases 1 -> 0; SiT time increases 0 -> 1.
    T, total_N, schedule = 1.0, 1000, 'linear_flow'
    def marginal_alpha(self,t): return 1-t
    def marginal_std(self,t): return t
    def marginal_log_mean_coeff(self,t): return torch.log1p(-t)
    def marginal_lambda(self,t): return torch.log1p(-t)-torch.log(t)
    def inverse_lambda(self,l): return torch.sigmoid(-l)

def load_model(checkpoint, device):
    from safetensors.torch import load_file
    model = SiT_models['SiT-S/2'](input_size=32).eval().to(device)
    state = load_file(str(checkpoint), device='cpu')
    model.load_state_dict(state, strict=True)
    if state['final_layer.linear.weight'].count_nonzero() == 0:
        raise ValueError('Untrained zero output head')
    return model.requires_grad_(False)

@torch.inference_mode()
def sample(model, noise, labels, method, nfe, *, cfg, order,
           t_start, t_end, precision='fp32'):
    calls = 0
    def data_prediction(x,t):
        nonlocal calls
        calls += 1
        tt = (1-t).reshape(-1).expand(x.shape[0])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
            if cfg == 0:
                # Pure null-label velocity on every latent channel, one model batch.
                velocity = model(x, tt, torch.full_like(labels, model.y_embedder.num_classes))
            elif cfg == 1:
                velocity = model(x, tt, labels)
            else:
                xx = torch.cat([x,x]); ts = torch.cat([tt,tt])
                yy = torch.cat([labels, torch.full_like(labels,1000)])
                # Match official SiT CFG: guide the first three latent channels.
                pred = model(xx,ts,yy).float()
                cond,uncond = pred.chunk(2)
                velocity = cond.clone()
                velocity[:,:3] = uncond[:,:3] + cfg*(cond[:,:3]-uncond[:,:3])
        # SiT ICPlan: x=(1-t)*data+t*noise, velocity=data-noise.
        # x0=x+t*v is algebraically equal to the source epsilon->x0
        # conversion and remains defined at t=1 (no epsilon clipping).
        return x + t.reshape(-1,1,1,1)*velocity.float()
    schedule=LinearFlowSchedule()
    if method=='dpmpp':
        solver=DPM_Solver(data_prediction,schedule,algorithm_type='dpmsolver++')
    elif method=='unipc':
        solver=EndpointUniPC(data_prediction,schedule,algorithm_type='data_prediction',variant='bh2')
    else: raise ValueError(method)
    solver.data_prediction_fn=solver.model
    out=solver.sample(noise,steps=nfe,order=order,method='multistep',
                      skip_type='time_uniform',t_start=t_start,t_end=t_end,
                      lower_order_final=True,denoise_to_zero=False)
    if calls != nfe: raise AssertionError(f'NFE mismatch: {calls} != {nfe}')
    if not torch.isfinite(out).all(): raise FloatingPointError('Nonfinite sample')
    return out,calls


class EndpointUniPC(UniPC):
    """Exact t_old=1 limit of source UniPC BH2 order-2 formula.

    At the second update rk=(lambda_old-lambda_prev)/h -> -inf.
    D1=(m_old-m_prev)/rk -> 0. Solving source R*rho=b gives
    rho_old=(b1-b0)/(rk-1)->0, rho_current->b0.
    All other steps, including the first corrector, run the source code.
    No additional model call, endpoint truncation, or solver choice is added.
    """
    def multistep_uni_pc_bh_update(self,x,model_prev_list,t_prev_list,t,order,x_t=None,use_corrector=True):
        ns=self.noise_schedule
        if not (order==2 and self.predict_x0 and self.variant=='bh2'
                and bool(torch.isneginf(ns.marginal_lambda(t_prev_list[-2])).all())):
            return super().multistep_uni_pc_bh_update(x,model_prev_list,t_prev_list,t,order,x_t,use_corrector)
        s=t_prev_list[-1]; prev=model_prev_list[-1]
        h=ns.marginal_lambda(t)-ns.marginal_lambda(s)
        hh=-h; bh=torch.expm1(hh)
        alpha=ns.marginal_alpha(t)
        base=ns.marginal_std(t)/ns.marginal_std(s)*x-alpha*bh*prev
        if x_t is None: x_t=base
        model_t=None
        if use_corrector:
            model_t=self.model_fn(x_t,t)
            # bh*b0 = expm1(-h)/(-h)-1, directly avoiding 0*inf.
            x_t=base-alpha*(bh/hh-1)*(model_t-prev)
        return x_t,model_t
