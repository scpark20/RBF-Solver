# Regression fixture: positive-CFG behavior before null-label support.
def sample(model, noise, labels, method, nfe, *, cfg, order,
           t_start, t_end, precision='fp32'):
    calls = 0
    def data_prediction(x,t):
        nonlocal calls
        calls += 1
        tt = (1-t).reshape(-1).expand(x.shape[0])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision=='bf16'):
            if cfg == 1:
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
