# Regression fixture: positive-CFG behavior before null-label support.
def guided_velocity(model, x, t, labels, cfg):
    """Use SiT's original three-channel CFG directly in physical FM time."""
    t = t.reshape(-1).expand(x.shape[0])
    if cfg == 1:
        return model(x, t, labels).float()
    xx = torch.cat([x, x])
    yy = torch.cat([labels, torch.full_like(labels, 1000)])
    return model.forward_with_cfg(xx, torch.cat([t, t]), yy, cfg).chunk(2)[0].float()
