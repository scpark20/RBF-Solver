"""CPU mathematical regression checks; no image/model sampling or file I/O.

The numerical values below are analytic test cases, not sweep settings.
Run from sit: PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest -v test_rbf_solver_fm
"""
import math
import unittest

import numpy as np
import torch

from rbf_solver_fm import RBF_FM_Solver, rbf_integral_coefficients


class CoefficientTests(unittest.TestCase):
    def test_constant_integral_signed_nonuniform(self):
        for start, end, nodes in [(0., .3, [0., -.2, -.7]), (.3, 0., [.3, .5, 1.])]:
            for gamma in (.25, 1., 2., math.inf):
                with self.subTest(start=start, gamma=gamma):
                    c = rbf_integral_coefficients(start, end, nodes, gamma=gamma)
                    self.assertAlmostEqual(float(c.sum()), end-start, places=13)

    def test_two_node_corrector_is_trapezoidal(self):
        # The bracketed two-node Gaussian interpolant integrates symmetrically.
        for start, end in [(0., 1.), (1., -.25)]:
            for gamma in (.1, .5, 1., 2., math.inf):
                c = rbf_integral_coefficients(start, end, [end, start], gamma=gamma)
                expected = torch.full((2,), (end-start)/2, dtype=torch.float64)
                torch.testing.assert_close(c, expected, atol=1e-13, rtol=1e-13)

    def test_against_independent_quadrature_of_interpolant(self):
        # Solve for interpolation weights, independently integrate the fitted
        # function using Gauss-Legendre quadrature (not the solver's erf formula).
        nodes = np.array([.0, -.35, -.9]); h = .2; gamma = .8
        values = np.array([.7, -.2, 1.3])
        K = np.exp(-((nodes[:, None]-nodes[None, :])/(gamma*h))**2)
        phi = np.block([[K, np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]])
        w = np.linalg.solve(phi, np.r_[values, 0.])
        u, q = np.polynomial.legendre.leggauss(96)
        t = h*(u+1)/2
        fitted = np.exp(-((t[:, None]-nodes[None, :])/(gamma*h))**2)@w[:3]+w[3]
        expected = h/2*np.dot(q, fitted)
        c = rbf_integral_coefficients(0., h, nodes, gamma=gamma)
        self.assertAlmostEqual(float(c@torch.tensor(values)), expected, places=13)

    def test_explicit_adams_limit(self):
        predictor = rbf_integral_coefficients(0, 1, [0, -1], gamma=math.inf)
        corrector = rbf_integral_coefficients(0, 1, [1, 0, -1], gamma=math.inf)
        torch.testing.assert_close(predictor, torch.tensor([1.5, -.5], dtype=torch.float64))
        torch.testing.assert_close(corrector, torch.tensor([5/12, 2/3, -1/12], dtype=torch.float64))
        # The flat-limit corrector integrates quadratics exactly.
        self.assertAlmostEqual(float(corrector@torch.tensor([1., 0., 1.], dtype=torch.float64)), 1/3)

    def test_finite_gamma_does_not_assume_polynomial_exactness(self):
        c = rbf_integral_coefficients(0, 1, [0, -1], gamma=1.)
        self.assertAlmostEqual(float(c[1]), .016257724637381833, places=13)
        self.assertGreater(abs(float(c[1])*(-1)-.5), .5)

    def test_bad_coefficient_inputs(self):
        cases = [(0, 0, [0], 1), (0, 1, [0, 0], 1),
                 (0, 1, [0], 0), (0, 1, [0], float('nan')),
                 (0, 1, [], 1), (0, float('inf'), [0], 1)]
        for start, end, nodes, gamma in cases:
            with self.subTest(case=(start, end, nodes, gamma)), self.assertRaises(ValueError):
                rbf_integral_coefficients(start, end, nodes, gamma=gamma)
        for gamma in (math.exp(2), 100., 300., 1e100):
            c = rbf_integral_coefficients(0, 1, [1, 0, -1, -2], gamma=gamma)
            flat = rbf_integral_coefficients(0, 1, [1, 0, -1, -2], gamma=math.inf)
            torch.testing.assert_close(c, flat, atol=0, rtol=0)


class SolverTests(unittest.TestCase):
    def make_solver(self, velocity, times, **settings):
        args = dict(timesteps=times, history_size=2, gamma_pred=1., gamma_corr=1., lower_order_final=True)
        args.update(settings)
        return RBF_FM_Solver(velocity, **args)

    def test_constant_velocity_nfe_and_callbacks(self):
        for steps in (5, 6, 8, 10):
            for direction in (1, -1):
                times = torch.linspace(0., 1., steps+1, dtype=torch.float64)
                if direction < 0:
                    times = times.flip(0)
                x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)
                before = x.clone(); events = []; calls = []
                def velocity(state, time, *, speed):
                    calls.append(time.clone())
                    return torch.full_like(state, speed)
                solver = self.make_solver(velocity, times)
                out = solver.sample(x, model_kwargs={'speed': .25}, callback=events.append)
                torch.testing.assert_close(out, x+direction*.25, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(x, before, atol=0, rtol=0)
                self.assertEqual(len(calls), steps)
                self.assertEqual(solver.last_nfe, steps)
                self.assertEqual(len(events), steps)
                self.assertEqual(events[-1].stage, 'predictor')
                self.assertEqual(events[-1].nfe, steps)
                self.assertTrue(all(e.stage == 'corrector' for e in events[:-1]))
                # Final time is reached, but not evaluated by the network.
                self.assertEqual(float(calls[-1][0]), float(times[-2]))

    def test_cached_prediction_state_and_last_predictor(self):
        inputs = []
        def velocity(x, t):
            inputs.append((x.item(), t.item()))
            return x+t
        solver = self.make_solver(velocity, [0, 1, 2, 3], history_size=1, lower_order_final=False)
        out = solver.sample(torch.zeros(1, dtype=torch.float64))
        self.assertEqual(inputs, [(0., 0.), (0., 1.), (1.5, 2.)])
        self.assertAlmostEqual(out.item(), 6.25, places=13)
        self.assertEqual(solver.last_nfe, 3)

    def test_nonuniform_reverse_and_per_step_gamma(self):
        solver = self.make_solver(lambda x, t: torch.ones_like(x), [1., .8, .35, 0.],
                                  gamma_pred=[.5, 1., 2.], gamma_corr=[.7, 1.2])
        out = solver.sample(torch.zeros((2, 4), dtype=torch.float32))
        torch.testing.assert_close(out, -torch.ones_like(out))

    def test_single_interval_is_euler(self):
        solver = self.make_solver(lambda x, t: x+1, [0., .5], gamma_corr=[])
        out = solver.sample(torch.ones(1, dtype=torch.float64))
        self.assertEqual(out.item(), 2.)
        self.assertEqual(solver.last_nfe, 1)

    def test_invalid_settings_fail_before_model_call(self):
        def should_not_run(x, t):
            self.fail('Model called with invalid solver settings')
        for times in ([0.], [0., 0.], [0., 1., .5], [0., float('nan')]):
            with self.assertRaises(ValueError):
                self.make_solver(should_not_run, times)
        with self.assertRaises(ValueError):
            self.make_solver(should_not_run, [0, 1, 2], gamma_corr=[1, 1])
        with self.assertRaises(ValueError):
            self.make_solver(should_not_run, [0, 1], history_size=0)
        solver = self.make_solver(should_not_run, [1., 1.+1e-10])
        with self.assertRaises(ValueError):
            solver.sample(torch.zeros(1, dtype=torch.float32))

    def test_velocity_shape_and_nan_rejected(self):
        solver = self.make_solver(lambda x, t: x[:, :1], [0, 1])
        with self.assertRaises(ValueError):
            solver.sample(torch.ones((2, 3)))
        solver = self.make_solver(lambda x, t: torch.full_like(x, float('nan')), [0, 1])
        with self.assertRaises(FloatingPointError):
            solver.sample(torch.ones((2, 3)))


if __name__ == '__main__':
    unittest.main()
