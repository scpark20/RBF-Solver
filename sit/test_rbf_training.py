"""CPU preparation tests: synthetic latents only, never SiT image generation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from rbf_training_run import (load_training_plan, make_inputs, validate_inputs,
    protocol_hash, TrainingWorker, Worker, FMShapeTrainer, RBFSolver,
    read_shapes, average_shapes, load_trained_solver, guided_velocity)
from models import SiT
from paths import RESULTS


class FakeModel:
    """Deterministic CPU field using the original SiT CFG procedure."""
    def forward(self, x, t, y):
        return x*0.1 + t[:,None,None,None] + y[:,None,None,None]/1000
    __call__ = forward
    forward_with_cfg = SiT.forward_with_cfg


class TrainingPreparationTests(unittest.TestCase):
    def setUp(self):
        self.p = load_training_plan()
        # Synthetic outputs obey the same result-only storage boundary.
        self.temp = tempfile.TemporaryDirectory(prefix='rbf_preparation_cpu_', dir=RESULTS)
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.bank = make_inputs(self.p)

    def worker(self):
        w = object.__new__(TrainingWorker)
        w.p, w.out, w.bank = self.p, self.out, self.bank
        w.gpu, w.pid, w.device = 0, 0, torch.device('cpu')
        w.hash, w.input_hash = protocol_hash(self.p), 'synthetic-input-test'
        w.manifest = dict(source_sha256={'synthetic-test': 'no-SiT-weights'})
        w.model, w.event = FakeModel(), Mock()
        return w

    def test_original_loop_and_batch_search_are_inherited(self):
        self.assertIs(FMShapeTrainer.sample_with_optim, RBFSolver.sample_with_optim)
        self.assertIs(TrainingWorker.tune, Worker.tune)
        self.assertIs(TrainingWorker.trial, Worker.trial)

    def test_exact_target_count_and_reproducible_replacement_draws(self):
        validate_inputs(self.bank, self.p)
        other = make_inputs(self.p)
        for k in ('noise','labels','indices'):
            self.assertTrue(torch.equal(self.bank[k], other[k]))
        for k, value in [('noise', self.bank['noise'][:-1]),
                         ('indices', torch.full_like(self.bank['indices'],128)),
                         ('labels', torch.full_like(self.bank['labels'],1000))]:
            with self.subTest(k=k), self.assertRaises(ValueError):
                validate_inputs(dict(self.bank, **{k:value}), self.p)

    def test_cfg_matches_existing_teacher_guidance(self):
        model = FakeModel()
        x, y = self.bank['noise'][:2], self.bank['labels'][:2]
        t = torch.full((2,), self.p['flow_time_end'])
        cond, uncond = model(x,t,y), model(x,t,torch.full_like(y,1000))
        for cfg in self.p['cfgs']:
            expected = cond.clone()
            expected[:,:3] = uncond[:,:3]+cfg*(cond[:,:3]-uncond[:,:3])
            self.assertTrue(torch.equal(guided_velocity(model,x,t,y,cfg),expected))

    def test_teacher_count_resume_and_identity_rejection(self):
        w = self.worker()
        w.tune = Mock(return_value=self.p['target_pairs'])
        w.latent = Mock(side_effect=lambda job,off,b: self.bank['noise'][off:off+b]+1)
        target, h = w.target_pairs(self.p['cfgs'][0])
        self.assertEqual(len(target),128)
        self.assertTrue(torch.equal(target,self.bank['noise']+1))
        self.assertEqual(w.latent.call_count,1)
        w.latent.reset_mock()
        resumed, resumed_hash = w.target_pairs(self.p['cfgs'][0])
        self.assertTrue(torch.equal(resumed,target))
        self.assertEqual(h,resumed_hash)
        w.latent.assert_not_called()
        w.input_hash = 'changed'
        with self.assertRaises(ValueError):
            w.target_pairs(self.p['cfgs'][0])

    def test_full_original_grid_fit_replays_through_saved_shape_loader(self):
        noise = self.bank['noise'][:2,:,:2,:2].clone()
        field = lambda x,t: x*0.1 + t[:,None,None,None]
        trainer = FMShapeTrainer(field,self.out,log_shape_min=self.p['log_gamma_min'],
                                log_shape_max=self.p['log_gamma_max'],log_shape_num=self.p['log_gamma_points'])
        nfe = self.p['nfes'][0]
        with contextlib.redirect_stdout(io.StringIO()):
            fitted = trainer.sample_with_optim(noise,noise+1,steps=nfe,order=self.p['history_size'],
                       t_start=self.p['flow_time_start'],t_end=self.p['flow_time_end'],
                       skip_type=self.p['time_grid'],lower_order_final=self.p['lower_order_final'],number=0)
        path = self.out/f'NFE={nfe},p={self.p["history_size"]},number=0.npz'
        read_shapes(path,nfe,self.p,with_losses=True)
        solver = load_trained_solver(field,path,nfe,self.p)
        actual = solver.sample(noise)
        self.assertEqual(solver.last_nfe,nfe)
        torch.testing.assert_close(actual,fitted,rtol=1e-6,atol=1e-6)

    def test_driver_20_runs_mean_log_shapes_resume_and_missing_file_failure(self):
        w = self.worker()
        cfg = self.p['cfgs'][0]
        targets = self.bank['noise']+1
        calls = []
        def synthetic_optimizer(trainer,noise,target,*,steps,t_start,t_end,order,skip_type,lower_order_final,number):
            calls.append(dict(steps=steps,order=order,batch=len(noise),number=number,
                              t_start=t_start,t_end=t_end,skip_type=skip_type,lower_order_final=lower_order_final))
            for t in torch.linspace(t_start,t_end,steps+1)[:-1]:
                trainer.model_fn(noise,t)
            grid = np.linspace(trainer.log_shape_min,trainer.log_shape_max,trainer.log_shape_num)
            dest = Path(trainer.shape_dir)
            dest.mkdir(parents=True,exist_ok=True)
            np.savez(dest/f'NFE={steps},p={order},number={number}.npz',
                     optimal_log_shapes=np.full((2,steps),grid[number]),
                     loss_grid_list=np.zeros((steps-1,len(grid),len(grid))))
            return noise
        with patch.object(FMShapeTrainer,'sample_with_optim',synthetic_optimizer), patch('rbf_training_run.clear_memory'):
            for nfe in self.p['nfes']:
                w.fit_condition(cfg,nfe,targets,'synthetic-targets')
            self.assertEqual(len(calls),len(self.p['nfes'])*self.p['repetitions'])
            for call in calls:
                self.assertEqual((call['order'],call['batch']),(2,16))
                self.assertEqual((call['t_start'],call['t_end'],call['skip_type'],call['lower_order_final']),
                                 (0.0,0.999,'time_uniform',True))
            expected = np.linspace(-2,2,33)[:20].mean()
            for nfe in self.p['nfes']:
                dest = self.out/f'cfg_{cfg:g}/nfe_{nfe}'
                mean = read_shapes(dest/f'NFE={nfe},p=2.npz',nfe,self.p)
                np.testing.assert_array_equal(mean,np.full((2,nfe),expected))
                w.fit_condition(cfg,nfe,targets,'synthetic-targets')
            self.assertEqual(len(calls),len(self.p['nfes'])*self.p['repetitions'])
        nfe = self.p['nfes'][0]
        dest = self.out/f'cfg_{cfg:g}/nfe_{nfe}'
        identities = [json.loads((dest/f'NFE={nfe},p=2,number={i}.json').read_text())['identity'] for i in range(20)]
        (dest/f'NFE={nfe},p=2,number=19.npz').unlink()
        with self.assertRaises(FileNotFoundError):
            average_shapes(dest,nfe,self.p,identities)

    def test_learned_shapes_never_fall_back_on_missing_or_invalid_file(self):
        nfe = self.p['nfes'][0]
        path = self.out/'missing.npz'
        with self.assertRaises(FileNotFoundError):
            read_shapes(path,nfe,self.p)
        np.savez(path,optimal_log_shapes=np.full((2,nfe),np.nan))
        with self.assertRaises(ValueError):
            read_shapes(path,nfe,self.p)

    def test_dashboard_shows_all_training_conditions(self):
        from dashboard import rbf_status
        status = rbf_status()
        self.assertIn(self.p,status['protocols'])
        self.assertEqual(status['conditions'],27)
        self.assertEqual(len(status['targets']),5)
        self.assertEqual({(r['cfg'],r['nfe']) for r in status['items'] if r['cfg'] in self.p['cfgs']},
                         {(c,n) for c in self.p['cfgs'] for n in self.p['nfes']})


if __name__ == '__main__':
    unittest.main()
