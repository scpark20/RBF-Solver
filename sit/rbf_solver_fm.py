"""Gaussian RBF predictor-corrector for dx/dt = velocity(x, t).

Derivation: the user's "Sampling Equations for Flow-Matching" section;
reference PC loop: RBF-Solver arXiv:2603.13330, Algorithm 1 and
scpark20/RBF-Solver commit af306c69f016ed3ef5a72c96bbb08af285c79315,
guided-diffusion/samplers/rbf_solver.py.

Both stages integrate over [t_i, t_{i+1}]. Predictor centers are
(t_i, ..., t_{i-p+1}); corrector centers add t_{i+1}, evaluated at the
PREDICTED state. Cached velocities are not reevaluated at corrected states.
The Gaussian width in BOTH stages is gamma_stage * abs(t_{i+1} - t_i).
The constant basis imposes sum(w_j) = 0. Coefficients sum to the signed
step length. No diffusion schedule, alpha/sigma factor or exp(lambda)
weight belongs in this flow-matching integral.

history_size specifies the maximum number of predictor interpolation
nodes, NOT a guaranteed convergence order. Finite gamma does not in
general provide history_size-th order accuracy. Positive infinity is an
explicit request for the polynomial (Adams) limit. As in the reference
solver and paper Section 3.6, log(gamma) >= 2 also selects that limit.
This documented branch avoids the numerically flat Gaussian solve.

All experiment settings are explicit: timesteps, history_size, gamma_pred,
gamma_corr, lower_order_final. Scalar gamma is reused at each stage;
sequences must contain M predictor and M-1 corrector entries. There are
exactly M velocity calls for M intervals, with no last corrector/denoise.
Time can increase or decrease; velocity must be dx/dt in the supplied
coordinate. CFG, labels, weights, endpoints, precision and random inputs
are owned by the caller. This module does not load or save any files.

Usage (all uppercase names are caller-supplied settings):
    solver = RBF_FM_Solver(
        velocity_fn, timesteps=TIMESTEPS, history_size=HISTORY_SIZE,
        gamma_pred=GAMMA_PRED, gamma_corr=GAMMA_CORR,
        lower_order_final=LOWER_ORDER_FINAL,
    )
    result = solver.sample(noise, model_kwargs={"y": labels}, callback=show)

velocity_fn receives x and a batch-shaped time tensor. A dashboard can
consume StepInfo through callback; arrange dashboard display before real
image sampling. This module does not start or modify the existing sweep.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
from typing import Any

import torch

ADAMS_LOG_SHAPE = 2.0  # Original RBFSolver.log_shape_max and paper Section 3.6.

__all__ = ["RBF_FM_Solver", "StepInfo", "rbf_integral_coefficients"]


def _vector(value: Any, name: str) -> torch.Tensor:
    """Small coefficient problems are prepared once on CPU in float64."""
    try:
        result = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be a one-dimensional real sequence") from exc
    if result.ndim != 1 or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be one-dimensional and finite")
    return result.clone()


def _gamma(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be positive (or +inf for the Adams limit)")
    result = float(value)
    if math.isnan(result) or result <= 0:
        raise ValueError(f"{name} must be positive (or +inf for the Adams limit)")
    return result


def _schedule(value: Any, count: int, name: str) -> tuple[float, ...]:
    raw = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach()
    if raw.ndim == 0:
        return (_gamma(value, name),) * count
    if raw.ndim != 1 or raw.numel() != count:
        raise ValueError(f"{name} must be scalar or have {count} entries")
    return tuple(_gamma(v, name) for v in raw.tolist())


def _erf_difference(a: float, b: float) -> float:
    """erf(b)-erf(a), avoiding cancellation of 1-1 in either tail."""
    if a >= 0:
        return math.erfc(a) - math.erfc(b)
    if b <= 0:
        return math.erfc(-b) - math.erfc(-a)
    return math.erf(b) - math.erf(a)


def rbf_integral_coefficients(
    t_start: float,
    t_end: float,
    nodes: Sequence[float] | torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    """Return coefficients c_j for integral R(t) dt over explicit endpoints.

    The returned CPU float64 vector follows the input node order. With
    u=(t-t_start)/h, h=t_end-t_start, the matrix kernel is
    exp(-((u_j-u_k)/gamma)**2) and the normalized integral is
    gamma*sqrt(pi)/2 * (erf((1-u_j)/gamma)-erf(-u_j/gamma)).
    Solve [K, 1; 1^T, 0] * c_aug = [l_normalized, 1], then multiply
    its first n entries by h. The last component multiplies a zero
    velocity row and is therefore omitted. Endpoints and node locations
    are independent, which also defines the corrector unambiguously.
    gamma >= exp(ADAMS_LOG_SHAPE) uses the reference Adams branch.
    """
    start, end = float(t_start), float(t_end)
    h = end - start
    if not math.isfinite(start) or not math.isfinite(end) or not math.isfinite(h) or h == 0:
        raise ValueError("Integration endpoints must be finite and different")
    center = _vector(nodes, "nodes")
    n = center.numel()
    if n == 0 or torch.unique(center).numel() != n:
        raise ValueError("At least one distinct interpolation node is required")
    shape = _gamma(gamma, "gamma")
    if n == 1:
        return torch.tensor([h], dtype=torch.float64)
    r = (center - start) / h
    if not torch.isfinite(r).all():
        raise ValueError("Normalized node locations overflow float64")

    if math.isinf(shape) or math.log(shape) >= ADAMS_LOG_SHAPE:
        # Integrate each Lagrange cardinal polynomial exactly in u.
        coefficients = []
        for j in range(n):
            polynomial = [1.0]
            denominator = 1.0
            for k in range(n):
                if k == j:
                    continue
                root = float(r[k])
                product = [0.0] * (len(polynomial) + 1)
                for power, value in enumerate(polynomial):
                    product[power] -= root * value
                    product[power + 1] += value
                polynomial = product
                denominator *= float(r[j] - r[k])
            coefficients.append(h * math.fsum(value / (power + 1)
                                               for power, value in enumerate(polynomial))
                                / denominator)
        out = torch.tensor(coefficients, dtype=torch.float64)
    else:
        phi = torch.ones((n + 1, n + 1), dtype=torch.float64)
        phi[:n, :n] = torch.exp(-((r[:, None] - r[None, :]) / shape).square())
        phi[-1, -1] = 0.0
        integral = torch.ones(n + 1, dtype=torch.float64)
        for j, node in enumerate(r.tolist()):
            integral[j] = shape * math.sqrt(math.pi) / 2 * _erf_difference(
                -node / shape, (1 - node) / shape)
        # Reject numerical singularity; no hidden shape clipping, ridge,
        # polynomial fallback, or pseudoinverse changes the stated method.
        condition = float(torch.linalg.cond(phi))
        if not math.isfinite(condition) or condition * torch.finfo(torch.float64).eps >= 1:
            raise FloatingPointError(
                "Gaussian interpolation matrix is numerically singular in float64; "
                "provide a resolvable finite gamma or explicitly request gamma=inf")
        out = torch.linalg.solve(phi, integral)[:n] * h
    if not torch.isfinite(out).all():
        raise FloatingPointError("Nonfinite integration coefficients")
    return out


@dataclass(frozen=True)
class StepInfo:
    """One completed interval; callback must copy sample to retain it."""
    step: int
    total_steps: int
    time: float
    nfe: int
    stage: str
    sample: torch.Tensor


@dataclass(frozen=True)
class _Step:
    predictor: tuple[float, ...]
    corrector: tuple[float, ...] | None


class RBF_FM_Solver:
    """Precompute scalar coefficients, then reuse the solver across batches.

    Warmup uses p_i=min(i+1, history_size). If lower_order_final is true,
    the original RBF code's rule is applied: once i>=history_size,
    p_i=min(i+1, M-i, history_size). This flag reduces history usage;
    it does not assert an asymptotic accuracy order.
    """
    def __init__(
        self,
        velocity_fn: Callable[..., torch.Tensor],
        *,
        timesteps: Sequence[float] | torch.Tensor,
        history_size: int,
        gamma_pred: float | Sequence[float] | torch.Tensor,
        gamma_corr: float | Sequence[float] | torch.Tensor,
        lower_order_final: bool,
    ) -> None:
        if not callable(velocity_fn):
            raise TypeError("velocity_fn must be callable")
        if isinstance(history_size, bool) or not isinstance(history_size, numbers.Integral) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        if not isinstance(lower_order_final, bool):
            raise TypeError("lower_order_final must be bool")
        times = _vector(timesteps, "timesteps")
        if times.numel() < 2:
            raise ValueError("At least two timesteps are required")
        delta = times[1:] - times[:-1]
        if not (bool((delta > 0).all()) or bool((delta < 0).all())):
            raise ValueError("timesteps must be strictly increasing or strictly decreasing")
        self.timesteps = tuple(times.tolist())
        self.steps = len(self.timesteps) - 1
        self.history_size = int(history_size)
        self.velocity_fn = velocity_fn
        self.gamma_pred = _schedule(gamma_pred, self.steps, "gamma_pred")
        self.gamma_corr = _schedule(gamma_corr, self.steps - 1, "gamma_corr")
        self.lower_order_final = lower_order_final
        self.last_nfe = 0
        prepared = []
        for i in range(self.steps):
            p = min(i + 1, self.history_size)
            if lower_order_final and i >= self.history_size:
                p = min(p, self.steps - i)
            centers = list(reversed(self.timesteps[i - p + 1:i + 1]))
            start, end = self.timesteps[i:i + 2]
            cp = rbf_integral_coefficients(start, end, centers, gamma=self.gamma_pred[i])
            cc = None
            if i < self.steps - 1:
                # Same endpoints and same current h; only add the new center.
                cc = tuple(rbf_integral_coefficients(
                    start, end, [end, *centers], gamma=self.gamma_corr[i]).tolist())
            prepared.append(_Step(tuple(cp.tolist()), cc))
        self._prepared = tuple(prepared)

    @staticmethod
    def _advance(base: torch.Tensor, coefficients: tuple[float, ...], history: list[torch.Tensor]) -> torch.Tensor:
        if len(history) < len(coefficients):
            raise RuntimeError("Insufficient velocity history")
        result = base.clone()
        for coefficient, velocity in zip(coefficients, history):
            result.add_(velocity, alpha=coefficient)
        return result

    @torch.no_grad()
    def sample(
        self,
        x: torch.Tensor,
        *,
        model_kwargs: Mapping[str, Any] | None = None,
        callback: Callable[[StepInfo], None] | None = None,
    ) -> torch.Tensor:
        """Integrate a batch without modifying x; return the last prediction.

        No initial noise is generated. model_kwargs passes labels or other
        conditioning to velocity_fn. The caller supplies an already-guided
        velocity function when CFG is desired. last_nfe records actual
        velocity function invocations, including an invocation that fails.
        """
        if not isinstance(x, torch.Tensor) or not x.is_floating_point() or x.ndim < 1 or x.shape[0] == 0:
            raise ValueError("x must be a nonempty floating-point batch tensor")
        kwargs = dict(model_kwargs or {})
        self.last_nfe = 0
        time_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        model_times = torch.tensor(self.timesteps, device=x.device, dtype=time_dtype)
        difference = model_times[1:] - model_times[:-1]
        if not (bool((difference > 0).all()) or bool((difference < 0).all())):
            raise ValueError("Timesteps collapse at the model time dtype")

        def evaluate(state: torch.Tensor, index: int) -> torch.Tensor:
            self.last_nfe += 1
            velocity = self.velocity_fn(state, model_times[index].expand(state.shape[0]), **kwargs)
            if not isinstance(velocity, torch.Tensor) or not velocity.is_floating_point():
                raise TypeError("velocity_fn must return a floating-point tensor")
            if velocity.shape != state.shape or velocity.device != state.device:
                raise ValueError("Velocity shape and device must match the state")
            if not bool(torch.isfinite(velocity).all()):
                raise FloatingPointError(f"Nonfinite velocity at step {index}")
            return velocity.to(dtype=state.dtype)

        current = x
        history = [evaluate(current, 0)]
        for i, prepared in enumerate(self._prepared):
            predicted = self._advance(current, prepared.predictor, history)
            if prepared.corrector is None:
                current = predicted
                stage = "predictor"
            else:
                new_velocity = evaluate(predicted, i + 1)
                current = self._advance(current, prepared.corrector, [new_velocity, *history])
                history = [new_velocity, *history[:self.history_size - 1]]
                stage = "corrector"
            if callback is not None:
                callback(StepInfo(i + 1, self.steps, self.timesteps[i + 1], self.last_nfe, stage, current))
        if self.last_nfe != self.steps:
            raise RuntimeError(f"NFE mismatch: {self.last_nfe} != {self.steps}")
        if not bool(torch.isfinite(current).all()):
            raise FloatingPointError("Nonfinite final state")
        return current
