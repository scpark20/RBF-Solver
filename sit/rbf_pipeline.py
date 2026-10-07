"""Explicit CFG-scoped target -> fitting -> RBF sampling, reusing existing engines."""
from pathlib import Path
import argparse, contextlib, fcntl, json, os, subprocess, time, traceback, urllib.request
import torch
from paths import ROOT, RESULTS, validate_storage, PYTHON
from experiment import load_plan, plan_digest, jobs, atomic_json
from benchmark import Worker, digest, clear_memory
from rbf_training_run import (TrainingWorker, load_training_plan, protocol_hash,
    source_hashes, check_preparation, atomic_torch, make_inputs, validate_inputs,
    read_shapes, load_trained_solver, guided_velocity)


def sampling_plan():
    """Use the previously approved comparison conditions and exactly the same inputs."""
    p=json.loads((ROOT/'rbf_sampling.json').read_text()); original=load_plan(True)
    for k in ['nfes','num_samples','order','seed','precision','t_start','t_end','skip_type',
              'lower_order_final','denoise_to_zero','reference','gpus','batch_policy']:
        if p[k]!=original[k]: raise ValueError(f'RBF comparison changed {k}')
    if p['cfgs']!=[1.5] or p['methods']!=['rbf']: raise ValueError('Only CFG 1.5 RBF is authorized for this run')
    for k in ['output','training_output','input_bank']:
        q=Path(p[k]).resolve()
        if q==RESULTS.resolve() or not q.is_relative_to(RESULTS.resolve()): raise ValueError(f'Invalid {k}')
    if Path(p['input_bank']).resolve()!=(Path(original['output'])/'inputs.pt').resolve():
        raise ValueError('RBF must use the original comparison input bank')
    return p


def target_identity(p, manifest, bank_path, cfg):
    return dict(protocol_sha256=protocol_hash(p),inputs_sha256=digest(bank_path),
                source_sha256=manifest['source_sha256'],cfg=cfg)


def load_targets(p,cfg):
    """Load a complete, matching target bundle; never silently generate missing targets."""
    out=Path(p['output']); manifest=json.loads((out/'manifest.json').read_text())
    if manifest['source_sha256']!=source_hashes(): raise ValueError('Target source manifest mismatch')
    path=out/f'cfg_{cfg:g}/teacher/targets.pt'
    record=torch.load(path,map_location='cpu',weights_only=True)
    identity=target_identity(p,manifest,out/'inputs.pt',cfg)
    if record['identity']!=identity or record['done']!=p['target_pairs']:
        raise ValueError('Target identity/count mismatch')
    x=record['samples']
    if x.shape!=(p['target_pairs'],4,32,32) or x.dtype!=torch.float32 or not torch.isfinite(x).all():
        raise ValueError('Invalid target tensor')
    return x,digest(path)


class StageWorker(TrainingWorker):
    """Reuse teacher sampling, maximum-batch search and the original fitting procedure."""
    def __init__(self,gpu,stage):
        super().__init__(gpu); self.stage=stage

    def latent(self,job,offset,b):
        return super().latent(job,job.get('offset',0)+offset,b)

    @torch.inference_mode()
    def target_shard(self,job):
        dest=self.out/job['id']; dest.mkdir(parents=True,exist_ok=True)
        path=dest/'targets.pt'; count=job['stop']-job['offset']
        identity=dict(target_identity(self.p,self.manifest,self.out/'inputs.pt',job['cfg']),
                      offset=job['offset'],stop=job['stop'])
        record=dict(identity=identity,done=0,samples=torch.zeros(count,4,32,32))
        if path.exists(): record=torch.load(path,map_location='cpu',weights_only=True)
        if record['identity']!=identity or not 0<=record['done']<=count: raise ValueError('Shard resume mismatch')
        if record['samples'].shape!=(count,4,32,32) or not torch.isfinite(record['samples']).all():
            raise ValueError('Invalid target shard')
        def status(state,**kw):
            atomic_json(dest/'status.json',dict(state=state,done=record['done'],total=count,
                gpu=self.gpu,pid=self.pid,updated_at=time.time(),**kw))
        if record['done']<count:
            if self.p.get('reuse_measured_batches'):
                # This complete target shard is smaller than previously executed SiT batches.
                batch=count-record['done']
            else:
                status('tuning')
                self.batch_cache[job['method']]=count-record['done']
                batch=self.tune(job,cap=count-record['done'])
            while record['done']<count:
                done=record['done']; b=min(batch,count-done)
                self.event('teacher',job=job['id'],done=done,total=count,batch_size=batch)
                status('teacher',batch_size=batch)
                oom=False
                try: x=self.latent(job,done,b)
                except torch.OutOfMemoryError:
                    if b==1: raise
                    oom=True
                if oom:
                    clear_memory()
                    batch=b//2 if self.p.get('reuse_measured_batches') else self.tune(job,cap=b-1)
                    continue
                if not torch.isfinite(x).all(): raise FloatingPointError('Nonfinite teacher target')
                record['samples'][done:done+b]=x.cpu(); del x
                record['done']+=b; atomic_torch(path,record); status('teacher',batch_size=batch)
        status('complete')

    @torch.inference_mode()
    def run(self):
        while (job:=self.claim()) is not None:
            try:
                if self.stage=='target': self.target_shard(job)
                elif self.stage=='fit':
                    target,h=load_targets(self.p,job['cfg'])
                    self.fit_condition(job['cfg'],job['nfe'],target,h)
                else: raise ValueError(self.stage)
                self.finish(job,'complete')
            except Exception as exc:
                self.finish(job,'error')
                atomic_json(self.out/job['id']/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
                self.event('error',job=job['id'],error=repr(exc)); raise
        self.event('complete',message=f'{self.stage} 배정 작업 완료')


class SamplingWorker(Worker):
    """Reuse VAE, GPU FID, paired inputs, batch search and resumable result writing."""
    def __init__(self,gpu):
        p=sampling_plan(); training=load_training_plan(True)
        with urllib.request.urlopen('http://127.0.0.1:8765/api/rbf-training',timeout=10) as f: visible=json.load(f)
        if visible.get('sampling',{}).get('protocol')!=p: raise RuntimeError('RBF sampling plan must be on dashboard first')
        super().__init__(gpu,plan=p,input_path=p['input_bank'])
        self.training=training; self.solvers={}; self.learned={}
        if self.bank['seed']!=p['seed'] or self.bank['noise'].shape!=(p['num_samples'],4,32,32):
            raise ValueError('Comparison input bank mismatch')
        for nfe in p['nfes']:
            dest=Path(training['output'])/f'cfg_1.5/nfe_{nfe}'
            result=json.loads((dest/'result.json').read_text())
            shape=Path(result['shape_file']).resolve()
            if not shape.is_relative_to(dest.resolve()): raise ValueError('Invalid trained shape path')
            if not result['complete'] or result['protocol_sha256']!=protocol_hash(training):
                raise ValueError('Training protocol mismatch')
            if result['source_sha256']!=source_hashes() or digest(shape)!=result['sha256']:
                raise ValueError('Training source or shape hash mismatch')
            read_shapes(shape,nfe,training)
            self.learned[nfe]=result
            def velocity(x,t,labels): return guided_velocity(self.model,x,t,labels,1.5)
            self.solvers[nfe]=load_trained_solver(velocity,shape,nfe,training)
        reference=json.loads((Path(load_plan()['output'])/'cfg_1.5/dpmpp_nfe5/result.json').read_text())
        if reference['config']['input_sha256']!=self.input_hash:
            raise ValueError('Sampling inputs differ from existing DPM/UniPC comparison')
        self.base.update(training_protocol_sha256=protocol_hash(training),flow_time_start=training['flow_time_start'],
                         flow_time_end=training['flow_time_end'],source_sha256=source_hashes())

    def latent(self,job,offset,b):
        z=self.bank['noise'][offset:offset+b].to(self.device)
        y=self.bank['labels'][offset:offset+b].to(self.device)
        solver=self.solvers[job['nfe']]
        x=solver.sample(z,model_kwargs={'labels':y})
        if solver.last_nfe!=job['nfe']: raise RuntimeError('RBF actual NFE mismatch')
        return x

    def run_job(self,job):
        result=self.learned[job['nfe']]
        self.base.update(shape_file=result['shape_file'],shape_sha256=result['sha256'])
        return super().run_job(job)


def merge_targets(p,cfg,shards):
    """Verify an exact partition of sample IDs, then persist the canonical 128 pairs."""
    out=Path(p['output']); manifest=json.loads((out/'manifest.json').read_text())
    identity=target_identity(p,manifest,out/'inputs.pt',cfg)
    targets=torch.empty(p['target_pairs'],4,32,32); covered=torch.zeros(p['target_pairs'],dtype=torch.bool)
    for j in shards:
        record=torch.load(out/j['id']/'targets.pt',map_location='cpu',weights_only=True)
        a,b=j['offset'],j['stop']; expected=dict(identity,offset=a,stop=b)
        if record['identity']!=expected or record['done']!=b-a or covered[a:b].any(): raise ValueError('Invalid shard coverage')
        targets[a:b]=record['samples'];covered[a:b]=True
    if not covered.all() or not torch.isfinite(targets).all(): raise ValueError('Incomplete teacher targets')
    dest=out/f'cfg_{cfg:g}/teacher'; atomic_torch(dest/'targets.pt',dict(identity=identity,done=p['target_pairs'],samples=targets))
    load_targets(p,cfg)
    atomic_json(dest/'status.json',dict(state='complete',done=p['target_pairs'],total=p['target_pairs'],gpus=p['gpus'],updated_at=time.time()))


def prepare(p):
    """Bind the exact original implementation and persist shared deterministic inputs."""
    report=check_preparation(True); out=Path(p['output']); out.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol_sha256=report['protocol_sha256'],source_sha256=report['source_sha256'])
    file=out/'manifest.json'
    if file.exists():
        old=json.loads(file.read_text())
        if any(old[k]!=v for k,v in identity.items()): raise ValueError('Existing run source/plan mismatch')
    else: atomic_json(file,dict(**identity,protocol=p,created_at=time.time()))
    bank=out/'inputs.pt'
    if not bank.exists(): atomic_torch(bank,make_inputs(p))
    validate_inputs(torch.load(bank,map_location='cpu',weights_only=True),p)
    return identity


def execute_stage(stage,p,queue,identity,cfg):
    """Launch both GPUs against one scoped queue and propagate any worker failure."""
    out=Path(p['output']); out.mkdir(parents=True,exist_ok=True)
    ph=protocol_hash(p) if stage!='sample' else plan_digest(p)
    for j in queue:
        j['state']='pending'
        dest=out/j['id']; dest.mkdir(parents=True,exist_ok=True)
        # A stale error belongs to a previous attempt; preserve it with an explicit timestamp.
        if (dest/'error.json').exists():
            (dest/'error.json').rename(dest/f'error_previous_{time.time_ns()}.json')
    atomic_json(out/'queue.json',dict(plan_sha256=ph,jobs=queue,stage=stage,cfgs=[cfg]))
    if stage=='sample': atomic_json(out/'manifest.json',dict(protocol=p,source_sha256=source_hashes(),created_at=time.time()))
    procs=[]
    with contextlib.ExitStack() as stack:
        try:
            for gpu in p['gpus']:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1')
                log=stack.enter_context(open(out/f'{stage}_gpu{gpu}.log','a'))
                proc=subprocess.Popen([str(PYTHON),str(ROOT/'rbf_pipeline.py'),'--worker',str(gpu),'--stage',stage,'--cfg',str(cfg)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                procs.append((gpu,proc))
            workers={str(g):v.pid for g,v in procs}
            atomic_json(out/'run_state.json',dict(state='running',stage=stage,workers=workers,started_at=time.time(),**identity))
            codes={str(g):v.wait() for g,v in procs}
            final=json.loads((out/'queue.json').read_text())
            good=all(c==0 for c in codes.values()) and all(j['state']=='complete' for j in final['jobs'])
            atomic_json(out/'run_state.json',dict(state='complete' if good else 'error',stage=stage,workers=workers,exit_codes=codes,finished_at=time.time(),**identity))
            if not good: raise RuntimeError(f'{stage} workers failed: {codes}')
        finally:
            for _,v in procs:
                if v.poll() is None:v.terminate()
            for _,v in procs:v.wait()


def launch(cfg,stage):
    """Run only the requested stage(s); stage-specific calls never run their prerequisites."""
    p=load_training_plan(True); sp=sampling_plan(); validate_storage()
    if cfg!=1.5: raise ValueError('This execution is authorized for CFG 1.5 only')
    stages=['target','fit','sample'] if stage=='all' else [stage]
    selection=Path(p['output'])/'selection.json'; out=Path(p['output']);out.mkdir(parents=True,exist_ok=True)
    with contextlib.ExitStack() as stack:
        for file in [Path(load_plan()['output'])/'launcher.lock',out/'launcher.lock']:
            lock=stack.enter_context(open(file,'a')); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if active:raise RuntimeError('GPUs are occupied; no existing process will be interrupted')
        identity=prepare(p)
        atomic_json(selection,dict(cfgs=[cfg],nfes=p['nfes'],stages=stages,state='running',stage=stages[0],pid=os.getpid(),updated_at=time.time()))
        with urllib.request.urlopen('http://127.0.0.1:8765/api/rbf-training',timeout=10) as f:visible=json.load(f)
        if visible.get('selection',{}).get('cfgs')!=[cfg] or visible.get('sampling',{}).get('protocol')!=sp:
            raise RuntimeError('Dashboard must display the scoped three-stage plan before execution')
        try:
            for current in stages:
                atomic_json(selection,dict(cfgs=[cfg],nfes=p['nfes'],stages=stages,state='running',stage=current,pid=os.getpid(),updated_at=time.time()))
                if current=='target':
                    queue=[]
                    for i,gpu in enumerate(p['gpus']):
                        a=p['target_pairs']*i//len(p['gpus']);b=p['target_pairs']*(i+1)//len(p['gpus'])
                        queue.append(dict(id=f'cfg_{cfg:g}/teacher/shard_{i}',cfg=cfg,method=p['teacher']['method'],nfe=p['teacher']['nfe'],offset=a,stop=b))
                    execute_stage(current,p,queue,identity,cfg); merge_targets(p,cfg,queue)
                elif current=='fit':
                    load_targets(p,cfg)
                    queue=[dict(id=f'cfg_{cfg:g}/nfe_{n}',cfg=cfg,nfe=n) for n in p['nfes']]
                    execute_stage(current,p,queue,identity,cfg)
                elif current=='sample':
                    # Verify every shape before allocating models or producing any image.
                    for n in sp['nfes']:
                        result=json.loads((out/f'cfg_{cfg:g}/nfe_{n}/result.json').read_text())
                        if not result['complete'] or digest(result['shape_file'])!=result['sha256']:raise ValueError('Learned shape incomplete or corrupted')
                    execute_stage(current,sp,jobs(sp),dict(plan_sha256=plan_digest(sp)),cfg)
            atomic_json(selection,dict(cfgs=[cfg],nfes=p['nfes'],stages=stages,state='complete',stage=stages[-1],pid=os.getpid(),updated_at=time.time()))
        except Exception as exc:
            atomic_json(selection,dict(cfgs=[cfg],nfes=p['nfes'],stages=stages,state='error',stage=current if 'current' in locals() else stages[0],error=repr(exc),pid=os.getpid(),updated_at=time.time()))
            raise
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfg',type=float,required=True,choices=[1.5])
    parser.add_argument('--stage',choices=['target','fit','sample','all'],required=True)
    parser.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    a=parser.parse_args()
    if a.worker is not None:
        if a.stage=='sample':SamplingWorker(a.worker).run()
        elif a.stage in ('target','fit'):StageWorker(a.worker,a.stage).run()
        else:raise ValueError('Worker must have one stage')
        return 0
    return launch(a.cfg,a.stage)

if __name__=='__main__':raise SystemExit(main())
