"""Prepare/check or explicitly run paper-protocol RBF shape fitting.

Reuses RBFSolver.sample_with_optim through FMShapeTrainer, sampling.sample
for the teacher, and benchmark.Worker.trial/tune for measured GPU batches.
No target generation or training occurs without --run (or its child --worker).
"""
from pathlib import Path
import argparse
import contextlib
import fcntl
import json
import os
import subprocess
import time
import traceback
import urllib.request
import numpy as np
import torch
from paths import ROOT, PYTHON, RESULTS, WEIGHTS, validate_storage
from experiment import atomic_json, plan_digest, validate_time_plan
from benchmark import Worker, digest, locked, clear_memory
from sampling import load_model, sample, time_mapping
from rbf_shape_training import FMShapeTrainer
from vendor.samplers.rbf_solver import RBFSolver

PLAN = ROOT / 'rbf_training.json'


def load_training_plan(require_ready=False):
    """Validate approved scientific settings and storage before using them."""
    p = json.loads(PLAN.read_text())
    assert p['gpus'] and len(set(p['gpus']))==len(p['gpus']) and all(isinstance(g,int) and g>=0 for g in p['gpus']), 'Use unique visible GPU indices'
    expected = dict(cfgs=[1.0], nfes=[5,6,8,10,12,15,20,25,30,35,40],
                    history_size=3, target_pairs=128, optimization_batch=128,
                    repetitions=1, aggregation='single_full_target_fit', replacement=False,
                    lower_order_final=True, log_gamma_min=-2.0, log_gamma_max=2.0,
                    log_gamma_points=33, adams_log_shape=2.0, seed=42,
                    precision='fp32', loss_precision='fp32')
    for k, v in expected.items():
        if p.get(k) != v:
            raise ValueError(f'Approved protocol mismatch: {k}={p.get(k)!r}, expected {v!r}')
    teacher = dict(method='unipc', variant='bh1', nfe=200, order=3,
                   t_start=1.0, t_end=1e-4, lower_order_final=True, denoise_to_zero=False)
    if p['teacher'] != teacher:
        raise ValueError('Teacher must match the approved CIFAR-10 UniPC protocol')
    for k, base in [('output', RESULTS), ('checkpoint', WEIGHTS)]:
        path = Path(p[k]).resolve()
        if path == base.resolve() or not path.is_relative_to(base.resolve()):
            raise ValueError(f'Forbidden {k} path: {path}')
    validate_time_plan(p)
    if require_ready and (not p['ready'] or p['pending']):
        raise RuntimeError('Training preparation has not passed its checks')
    return p


def protocol_hash(p):
    """Keep readiness bookkeeping separate from the immutable run identity."""
    return plan_digest({k: v for k, v in p.items()
                        if k not in ('ready', 'phase', 'preparation_validation')})


def source_hashes():
    """Bind saved targets and learned shapes to the exact code used."""
    names = ['rbf_pipeline.py', 'rbf_training_run.py', 'rbf_shape_training.py', 'rbf_solver_fm.py',
             'sampling.py', 'benchmark.py','capacity.py', 'experiment.py', 'paths.py',
             'vendor/RectifiedFlow/ImageGeneration/models/ncsnpp.py', 'vendor/samplers/rbf_solver.py',
             'vendor/samplers/uni_pc.py', 'vendor/samplers/dpm_solver.py']
    return {name: digest(ROOT/name) for name in names}


def check_preparation(require_ready=False):
    """Read-only checks; neither allocate a CUDA model nor generate target pairs."""
    p = load_training_plan(require_ready)
    validate_storage()
    if FMShapeTrainer.sample_with_optim is not RBFSolver.sample_with_optim:
        raise RuntimeError('Original optimization loop is not being reused')
    original = ROOT.parent/'score_sde/samplers/rbf_solver.py'
    if original.read_bytes() != (ROOT/'vendor/samplers/rbf_solver.py').read_bytes():
        raise RuntimeError('Vendor RBF differs from original')
    if digest(p['checkpoint']) != p['checkpoint_sha256']:
        raise RuntimeError('Checkpoint hash mismatch')
    return dict(passed=True, target_pairs=p['target_pairs'], conditions=len(p['cfgs'])*len(p['nfes']),
                original_optimizer_reused=True, checkpoint_verified=True,
                protocol_sha256=protocol_hash(p), source_sha256=source_hashes(),
                scope='Static preparation; no 1-RF target generation, training, or GPU batch measurement')


def atomic_torch(path, value):
    """Commit one resumable CPU bundle below the validated result directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def atomic_npz(path, **value):
    """Commit a complete shape file without NumPy adding a second suffix."""
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **value)
    tmp.replace(path)


def make_inputs(p):
    """Preserve all 128 CIFAR-10 pairs once; no CIFAR-10 replacement draws."""
    gen=torch.Generator(device='cpu').manual_seed(p['seed'])
    noise=torch.randn(p['target_pairs'],*p['data_shape'],generator=gen)
    labels=torch.zeros(p['target_pairs'],dtype=torch.int64)
    shape=(len(p['cfgs']),len(p['nfes']),p['repetitions'],p['optimization_batch'])
    indices=torch.arange(p['target_pairs']).expand(shape).clone()
    return dict(noise=noise,labels=labels,indices=indices,seed=p['seed'],protocol_sha256=protocol_hash(p))



def validate_inputs(bank, p):
    """Reject wrong target counts, changed plans, invalid classes or saved draws."""
    n = p['target_pairs']
    if bank['seed'] != p['seed'] or bank['protocol_sha256'] != protocol_hash(p):
        raise ValueError('Input bank protocol mismatch')
    z, y, ix = bank['noise'], bank['labels'], bank['indices']
    if z.shape != (n, *p['data_shape']) or z.dtype != torch.float32 or not torch.isfinite(z).all():
        raise ValueError('Input bank must contain exactly 128 finite FP32 latent noises')
    if y.shape != (n,) or y.dtype != torch.int64 or not (y == 0).all():
        raise ValueError('Invalid CIFAR-10 labels')
    shape = (len(p['cfgs']), len(p['nfes']), p['repetitions'], p['optimization_batch'])
    if ix.shape != shape or ix.dtype != torch.int64 or not torch.equal(ix,torch.arange(n).expand(shape)):
        raise ValueError('Invalid replacement draw bank')


def read_shapes(path, nfe, p, with_losses=False):
    """Load mandatory learned shapes; never substitute defaults on failure."""
    with np.load(path, allow_pickle=False) as f:
        shapes = f['optimal_log_shapes'].copy()
        if shapes.shape != (2, nfe) or not np.isfinite(shapes).all():
            raise ValueError(f'Invalid learned shape array: {path}')
        if np.any(shapes < p['log_gamma_min']) or np.any(shapes > p['log_gamma_max']):
            raise ValueError(f'Learned shapes outside approved grid bounds: {path}')
        if with_losses:
            losses = f['loss_grid_list']
            expected = (nfe-1, p['log_gamma_points'], p['log_gamma_points'])
            if losses.shape != expected or not np.isfinite(losses).all():
                raise ValueError(f'Invalid or incomplete loss grids: {path}')
    return shapes


def average_shapes(dest, nfe, p, expected_identities):
    """Compatibility entry point: validate the single full-target fit, without averaging."""
    if p['repetitions']!=1 or len(expected_identities)!=1:
        raise ValueError('CIFAR-10 requires exactly one full-target fit')
    path=dest/f'NFE={nfe},p={p["history_size"]}.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    if meta['identity']!=expected_identities[0] or meta['sha256']!=digest(path):
        raise ValueError('Fitted shape provenance mismatch')
    read_shapes(path,nfe,p,with_losses=True)
    return path




def load_trained_solver(velocity_fn, shape_file, nfe, p):
    """Connect mandatory saved log shapes to the FM sampler without fallbacks."""
    from rbf_solver_fm import RBF_FM_Solver
    logs = read_shapes(shape_file, nfe, p)
    times = torch.tensor(p['flow_timesteps'][str(nfe)],dtype=torch.float32)
    if times.shape!=(nfe+1,) or not torch.isfinite(times).all() or not (times[1:]>times[:-1]).all():
        raise ValueError('Invalid approved physical RF time grid')
    return RBF_FM_Solver(velocity_fn, timesteps=times, history_size=p['history_size'],
                         gamma_pred=np.exp(logs[0]), gamma_corr=np.exp(logs[1, :-1]),
                         lower_order_final=p['lower_order_final'])


def guided_velocity(model, x, t, labels, cfg):
    """Retain the existing call interface; 1-RF uses no classifier-free guidance."""
    if cfg!=1.0 or not (labels==0).all():
        raise ValueError('Only the unconditional condition is supported')
    return model(x,t.reshape(-1).expand(x.shape[0]),labels).float()



def require_dashboard(p):
    """Ensure the exact training plan is visible before any model sampling."""
    with urllib.request.urlopen('http://127.0.0.1:8766/api/rbf-training', timeout=10) as f:
        status = json.load(f)
    if status['protocol'] != p or status['conditions'] != len(p['cfgs'])*len(p['nfes']):
        raise RuntimeError('Dashboard must show the exact RBF plan first')


class TrainingWorker(Worker):
    """CFG queue worker; inherited trial/tune retain the existing batch search."""
    def __init__(self, gpu):
        self.p = load_training_plan(require_ready=True)
        require_dashboard(self.p)
        self.gpu, self.pid = gpu, os.getpid()
        self.out = Path(self.p['output'])
        self.hash = protocol_hash(self.p)
        self.worker_file = self.out/f'gpu{gpu}.json'
        self.manifest = json.loads((self.out/'manifest.json').read_text())
        if self.manifest['protocol_sha256'] != self.hash or not source_compatible(self.manifest['source_sha256']):
            raise ValueError('Training code or plan changed after launch')
        self.event('loading', message='1-RF 모델 로딩')
        self.bank = torch.load(self.out/'inputs.pt', map_location='cpu', weights_only=True)
        validate_inputs(self.bank, self.p)
        self.input_hash = digest(self.out/'inputs.pt')
        self.device = torch.device('cuda:0')
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(max(1, len(os.sched_getaffinity(0))//len(self.p['gpus'])))
        if digest(self.p['checkpoint']) != self.p['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after preparation')
        self.model = load_model(self.p['checkpoint'], self.device)
        self.batch_cache, self.trials = {}, []
        self.batch_scope = 'Full UniPC-200 teacher path, FP32 1-RF resident; latent targets only, no VAE or FID'

    def latent(self, job, offset, b):
        """Reuse the verified teacher sampler with saved paired inputs."""
        t = self.p['teacher']
        z = self.bank['noise'][offset:offset+b].to(self.device)
        y = self.bank['labels'][offset:offset+b].to(self.device)
        return sample(self.model, z, y, t['method'], t['nfe'], cfg=job['cfg'],
                      order=t['order'], t_start=t['t_start'], t_end=t['t_end'],
                      skip_type=self.p['time_grid'],precision=self.p['precision'])[0]

    def target_pairs(self, cfg):
        """Resume exactly 128 teacher targets; measure maximum batch before generation."""
        p = self.p
        job = dict(id=f'cfg_{cfg:g}/teacher', cfg=cfg, method=p['teacher']['method'], nfe=p['teacher']['nfe'])
        dest = self.out/job['id']
        dest.mkdir(parents=True, exist_ok=True)
        path = dest/'targets.pt'
        identity = dict(protocol_sha256=self.hash, inputs_sha256=self.input_hash,
                        source_sha256=self.manifest['source_sha256'], cfg=cfg)
        record = dict(identity=identity, done=0, samples=torch.zeros_like(self.bank['noise']))
        if path.exists():
            record = torch.load(path, map_location='cpu', weights_only=True)
            if record['identity'] != identity:
                raise ValueError('Target resume identity mismatch')
        done, targets = record['done'], record['samples']
        if not isinstance(done, int) or not 0 <= done <= p['target_pairs']:
            raise ValueError('Invalid target progress')
        if targets.shape != self.bank['noise'].shape or targets.dtype != torch.float32 or not torch.isfinite(targets).all():
            raise ValueError('Invalid target bundle')
        def progress(state, **kw):
            atomic_json(dest/'status.json', dict(state=state, done=done, total=p['target_pairs'],
                        gpu=self.gpu, pid=self.pid, updated_at=time.time(), **kw))
        if done < p['target_pairs']:
            progress('tuning')
            batch = self.tune(job, cap=p['target_pairs']-done)
            while done < p['target_pairs']:
                b = min(batch, p['target_pairs']-done)
                self.event('teacher', job=job['id'], done=done, total=p['target_pairs'], batch_size=batch)
                progress('teacher', batch_size=batch)
                oom = False
                try:
                    x = self.latent(job, done, b)
                except torch.OutOfMemoryError:
                    if b == 1:
                        raise
                    oom = True
                if oom:
                    clear_memory()
                    batch = self.tune(job, cap=b-1)
                    continue
                targets[done:done+b] = x.cpu()
                del x
                done += b
                record['done'] = done
                atomic_torch(path, record)
                progress('teacher', batch_size=batch)
        if done != p['target_pairs']:
            raise RuntimeError('Target generation did not produce exactly 128 pairs')
        progress('complete')
        return targets, digest(path)

    def fit_condition(self, cfg, nfe, targets, target_hash):
        """Call the unchanged CIFAR-10 optimizer once using all 128 target pairs."""
        p = self.p
        job_id = f'cfg_{cfg:g}/nfe_{nfe}'
        dest = self.out/job_id
        dest.mkdir(parents=True, exist_ok=True)
        ci, ni = p['cfgs'].index(cfg), p['nfes'].index(nfe)
        identities = []
        for number in range(p['repetitions']):
            indexes = self.bank['indices'][ci, ni, number]
            identity = dict(protocol_sha256=self.hash, inputs_sha256=self.input_hash,
                            targets_sha256=target_hash, source_sha256=self.manifest['source_sha256'],
                            cfg=cfg, nfe=nfe, order=p['history_size'], number=number,
                            indexes=indexes.tolist(), actual_nfe=nfe)
            identities.append(identity)
            stem = f'NFE={nfe},p={p["history_size"]}'
            path, marker = dest/f'{stem}.npz', dest/f'{stem}.json'
            if marker.exists():
                meta = json.loads(marker.read_text())
                if meta['identity'] != identity or meta['sha256'] != digest(path):
                    raise ValueError(f'Repetition resume mismatch: {marker}')
                read_shapes(path, nfe, p, with_losses=True)
                continue
            atomic_json(dest/'status.json', dict(state='training', done=number, total=p['repetitions'],
                        gpu=self.gpu, pid=self.pid, updated_at=time.time()))
            noise = self.bank['noise'][indexes].to(self.device)/p['sampling_times'][str(nfe)]['initial_state_scale']
            target = targets[indexes].to(self.device)
            labels = self.bank['labels'][indexes].to(self.device)
            calls = 0
            def velocity(x, t):
                nonlocal calls
                calls += 1
                self.event('training', job=job_id, repeat=number+1, repetitions=p['repetitions'],
                           step=calls, nfe=nfe, batch_size=p['optimization_batch'])
                v = guided_velocity(self.model, x, t, labels, cfg)
                if not torch.isfinite(v).all():
                    raise FloatingPointError('Nonfinite 1-RF velocity')
                return v
            times=p['flow_timesteps'][str(nfe)]
            trainer = FMShapeTrainer(velocity, dest, timesteps=times, initial_state_scale=p['sampling_times'][str(nfe)]['initial_state_scale'], log_shape_min=p['log_gamma_min'],
                                     log_shape_max=p['log_gamma_max'], log_shape_num=p['log_gamma_points'])
            value = trainer.sample_with_optim(noise, target, steps=nfe,
                        t_start=times[0], t_end=times[-1], order=p['history_size'],
                        skip_type=p['time_grid'], lower_order_final=p['lower_order_final'])
            if calls != nfe or not torch.isfinite(value).all():
                raise RuntimeError('Original optimizer NFE/finite-output check failed')
            read_shapes(path, nfe, p, with_losses=True)
            atomic_json(marker, dict(identity=identity, sha256=digest(path)))
            atomic_json(dest/'status.json', dict(state='training', done=number+1, total=p['repetitions'],
                        gpu=self.gpu, pid=self.pid, updated_at=time.time()))
            del value, trainer, noise, target, labels
            clear_memory()
        self.event('validating', job=job_id, repetitions=p['repetitions'])
        learned = average_shapes(dest, nfe, p, identities)
        atomic_json(dest/'result.json', dict(complete=True, cfg=cfg, nfe=nfe, repetitions=p['repetitions'],
                    aggregation=p['aggregation'], shape_file=str(learned), sha256=digest(learned),
                    protocol_sha256=self.hash, source_sha256=self.manifest['source_sha256']))
        atomic_json(dest/'status.json', dict(state='complete', done=p['repetitions'], total=p['repetitions'],
                    gpu=self.gpu, pid=self.pid, updated_at=time.time()))

    @torch.inference_mode()
    def run(self):
        """Use the existing locked shared queue; each CFG produces four NFE fits."""
        while (job := self.claim()) is not None:
            try:
                targets, target_hash = self.target_pairs(job['cfg'])
                for nfe in self.p['nfes']:
                    self.fit_condition(job['cfg'], nfe, targets, target_hash)
                self.finish(job, 'complete')
            except Exception as exc:
                self.finish(job, 'error')
                atomic_json(self.out/job['id']/'error.json', dict(error=repr(exc), traceback=traceback.format_exc()))
                self.event('error', job=job['id'], error=repr(exc))
                raise
        self.event('complete', message='배정된 RBF 학습 완료')


def launch():
    """Explicit run only: display check, exclusive GPU use, then both GPU workers."""
    report = check_preparation(require_ready=True)
    p = load_training_plan(require_ready=True)
    require_dashboard(p)
    sweep = json.loads((ROOT/'experiment.json').read_text())
    # Hold the same sweep lock for the entire RBF run; never stop an existing worker.
    with contextlib.ExitStack() as stack:
        sweep_lock = stack.enter_context(open(Path(sweep['output'])/'launcher.lock', 'a'))
        try:
            fcntl.flock(sweep_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('DPM/UniPC sweep is active; no RBF sampling or training was started') from None
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError('GPU compute processes are active; refusing concurrent RBF training')
        out = Path(p['output'])
        out.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context(open(out/'launcher.lock', 'a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = out/'manifest.json'
        identity = dict(protocol_sha256=report['protocol_sha256'], source_sha256=report['source_sha256'])
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if any(previous[k] != v for k, v in identity.items()):
                raise ValueError('Existing RBF run belongs to different code or settings')
        else:
            atomic_json(manifest_path, dict(**identity, protocol=p, created_at=time.time()))
        bank_path = out/'inputs.pt'
        if not bank_path.exists():
            atomic_torch(bank_path, make_inputs(p))
        validate_inputs(torch.load(bank_path, map_location='cpu', weights_only=True), p)
        qfile = out/'queue.json'
        if qfile.exists():
            q = json.loads(qfile.read_text())
            if q['plan_sha256'] != report['protocol_sha256']:
                raise ValueError('Existing RBF queue protocol mismatch')
            for job in q['jobs']:
                job['state'] = 'pending'
        else:
            q = dict(plan_sha256=report['protocol_sha256'], jobs=[dict(id=f'cfg_{cfg:g}', cfg=cfg, state='pending') for cfg in p['cfgs']])
        atomic_json(qfile, q)
        procs, workers, codes = [], {}, {}
        try:
            for gpu in p['gpus']:
                log = stack.enter_context(open(out/f'gpu{gpu}.log', 'a'))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1')
                proc = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker', str(gpu)],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                procs.append((gpu, proc))
                workers[str(gpu)] = proc.pid
            atomic_json(out/'run_state.json', dict(state='running', workers=workers, started_at=time.time(), **identity))
            codes = {str(g): proc.wait() for g, proc in procs}
            q = json.loads(qfile.read_text())
            complete = all(c == 0 for c in codes.values()) and all(j['state'] == 'complete' for j in q['jobs'])
            atomic_json(out/'run_state.json', dict(state='complete' if complete else 'error', workers=workers,
                        exit_codes=codes, finished_at=time.time(), **identity))
            return 0 if complete else 1
        finally:
            for _, proc in procs:
                if proc.poll() is None:
                    proc.terminate()
            for _, proc in procs:
                proc.wait()


def main():
    """Default is a read-only preparation check, never an implicit training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--worker', type=int,  help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        TrainingWorker(args.worker).run()
        return 0
    if args.run:
        return launch()
    print(json.dumps(check_preparation(), indent=2, ensure_ascii=False))
    return 0



def source_compatible(recorded):
    """Preserve actual historical source hashes; allow only audited operational fixes."""
    current=source_hashes()
    if recorded==current:return True
    path=RESULTS/'validation_data/operational_source_compatibility.json'
    if not path.exists():return False
    report=json.loads(path.read_text())
    if not report.get('passed') or set(recorded)!=set(current):return False
    allowed=report['changes']
    return all(old==current[name] or
               (name in allowed and allowed[name]['before']==old and allowed[name]['after']==current[name])
               for name,old in recorded.items())

if __name__ == '__main__':
    raise SystemExit(main())
