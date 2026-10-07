"""Minimal FM adapter for the EXISTING RBF training loop.

RBFSolver.sample_with_optim is inherited unchanged from the byte-identical
vendor copy. It retains grid search, model evaluation/history order, target
matching and per-run shape-file output. This adapter changes only the
coordinate/integral/update and makes loss and rollout share the Adams rule.
"""
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from vendor.samplers.rbf_solver import RBFSolver
from rbf_solver_fm import rbf_integral_coefficients


class LinearFMTrainingSchedule:
    """Provide RF target coordinates for the source VP target-matching objective.

    marginal_lambda is the inherited loop's coordinate hook, not a log-SNR.
    Both endpoints must be passed explicitly when invoking the trainer.
    """
    T = 0.0
    total_N = 1000

    def __init__(self, initial_state_scale):
        self.initial_state_scale=initial_state_scale

    def marginal_lambda(self, t):
        return t

    def marginal_alpha(self, t):
        return t

    def marginal_std(self, t):
        return (1-t)*self.initial_state_scale


class FMShapeTrainer(RBFSolver):
    """Reuse the original optimizer with an already-guided velocity function."""
    def __init__(self, velocity_fn, shape_dir, *, timesteps, initial_state_scale, log_shape_min, log_shape_max, log_shape_num):
        self.timesteps=torch.as_tensor(timesteps,dtype=torch.float32)
        if self.timesteps.ndim!=1 or len(self.timesteps)<2 or not torch.isfinite(self.timesteps).all() or not (self.timesteps[1:]>self.timesteps[:-1]).all():
            raise ValueError('Explicit increasing physical RF timesteps required')
        path = Path(shape_dir).resolve()
        allowed = Path('/data/RBF-Solver/results').resolve()
        if not path.is_relative_to(allowed) or path == allowed:
            raise ValueError('Shape files must be below /data/RBF-Solver/results')
        if log_shape_max != 2:
            raise ValueError('This paper-based preparation uses the original log_shape_max=2')
        super().__init__(velocity_fn, LinearFMTrainingSchedule(initial_state_scale), shape_dir=str(path),
                         log_shape_min=log_shape_min, log_shape_max=log_shape_max,
                         log_shape_num=log_shape_num)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Keep the original optimizer loop; supply the same approved grid as sampling."""
        if skip_type not in ('logSNR','time_uniform') or len(self.timesteps)!=N+1:
            raise ValueError('Grid specification mismatch')
        endpoints=torch.tensor([t_T,t_0],dtype=torch.float32)
        if not torch.equal(self.timesteps[[0,-1]],endpoints):
            raise ValueError('Optimizer endpoints differ from the saved grid')
        return self.timesteps.to(device)

    def data_prediction_fn(self, x, t):
        """Use model velocity directly; diffusion epsilon-to-data conversion is removed."""
        return self.model(x, t)

    def get_coefficients(self, start, end, nodes, beta):
        """Replace the original exp(lambda)-weighted integral by the FM integral."""
        gamma = 1/(float(beta)*abs(float(end-start)))
        return rbf_integral_coefficients(float(start), float(end), nodes, gamma=gamma).to(
            device=nodes.device, dtype=nodes.dtype)

    def get_lag_coefficients(self, start, end, nodes):
        """Integrate the reference Adams-limit basis without exp(lambda)."""
        return rbf_integral_coefficients(float(start), float(end), nodes, gamma=math.inf).to(
            device=nodes.device, dtype=nodes.dtype)

    def get_next_sample(self, sample, i, hist, signal_rates, noise_rates,
                        lambdas, p, beta, corrector=False, lagrange=False):
        """Keep original node/history selection; set diffusion prefactors to one."""
        return super().get_next_sample(sample, i, hist, signal_rates,
            torch.ones_like(noise_rates), lambdas, p, beta, corrector, lagrange)

    def get_loss_by_target_matching(self, i, x, target, hist, noise_rates,
                                    log_shape_p, log_shape_c, lambdas, p, p_prev):
        """Score the same FM corrector/predictor updates used by the inherited loop."""
        def step(base, index, count, log_shape, corrector):
            beta = 1/(np.exp(log_shape)*abs(lambdas[index+1]-lambdas[index]))
            return self.get_next_sample(base, index, hist, None, noise_rates,
                lambdas, count, beta, corrector=corrector,
                lagrange=bool(log_shape >= self.log_shape_max))
        candidate = x
        if log_shape_c is not None:
            candidate = step(candidate, i-1, p_prev, log_shape_c, True)
        candidate = step(candidate, i, p, log_shape_p, False)
        return F.mse_loss(candidate, target)
