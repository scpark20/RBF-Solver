"""Continue requested CFGs after the active run, preserving its source and results."""
from pathlib import Path
import contextlib,fcntl,json,os,subprocess,time,traceback,argparse,urllib.request
import torch
from paths import ROOT,RESULTS,validate_storage, PYTHON
from experiment import load_plan,plan_digest,jobs,atomic_json
from benchmark import Worker,digest
from rbf_training_run import load_training_plan,protocol_hash,source_hashes,read_shapes,load_trained_solver,guided_velocity
from rbf_pipeline import StageWorker,prepare,load_targets,merge_targets


def plan():
    name=os.environ.get('SIT_SAMPLING_PLAN','rbf_followup_sampling.json')
    if name not in ('rbf_followup_sampling.json','unconditional_sampling.json','unconditional_extended_sampling.json','unconditional_retrain_sampling.json'):raise ValueError('Unknown approved sampling plan')
    p=json.loads((ROOT/name).read_text());base=load_plan(True)
    for key in ['nfes','num_samples','order','seed','precision','t_start','t_end','skip_type','lower_order_final','denoise_to_zero','reference','gpus','batch_policy']:
        expected_value=([6] if key=='nfes' and name=='unconditional_retrain_sampling.json' else [12,15,20,25,30,35,40] if key=='nfes' and name=='unconditional_extended_sampling.json' else base[key])
        if p[key]!=expected_value:raise ValueError(f'Comparison condition changed: {key}')
    expected=([0.0],['rbf']) if name=='unconditional_retrain_sampling.json' else ([0.0],['dpmpp','unipc','rbf']) if name.startswith('unconditional_') else ([3.5,5.5,7.5],['rbf'])
    if (p['cfgs'],p['methods'])!=expected:raise ValueError('Follow-up must match the requested CFG scope')
    for key in ['output','training_output','input_bank']:
        value=Path(p[key]).resolve()
        if value==RESULTS.resolve() or not value.is_relative_to(RESULTS.resolve()):raise ValueError(f'Invalid {key}')
    if Path(p['input_bank']).resolve()!=(Path(base['output'])/'inputs.pt').resolve():raise ValueError('Comparison bank mismatch')
    return p


def execution_identity():
    name=os.environ.get('SIT_SAMPLING_PLAN','rbf_followup_sampling.json')
    training_name=os.environ.get('SIT_TRAINING_PLAN','rbf_training.json')
    return {'rbf_followup.py':digest(ROOT/'rbf_followup.py'),name:digest(ROOT/name),training_name:digest(ROOT/training_name)}


class FollowupSamplingWorker(Worker):
    """Only CFG selection and coefficient binding differ; use the original evaluator."""
    def __init__(self,gpu):
        p=plan(); self.training=load_training_plan(True);self.solvers={};self.learned={}
        with urllib.request.urlopen('http://127.0.0.1:8765/api/rbf-training',timeout=10) as f:visible=json.load(f)
        if p not in visible.get('sampling',{}).get('protocols',[]):raise RuntimeError('Follow-up plan must be visible first')
        super().__init__(gpu,plan=p,input_path=p['input_bank'])
        reference=json.loads((Path(load_plan()['output'])/'cfg_1.5/dpmpp_nfe5/result.json').read_text())
        if self.input_hash!=reference['config']['input_sha256'] or self.bank['noise'].shape!=(p['num_samples'],4,32,32):raise ValueError('Comparison inputs mismatch')
        self.base.update(training_protocol_sha256=protocol_hash(self.training),flow_time_start=self.training['flow_time_start'],flow_time_end=self.training['flow_time_end'],source_sha256=source_hashes(),execution_source_sha256=execution_identity())

    def bind(self,cfg,nfe):
        key=(cfg,nfe)
        if key in self.solvers:return
        dest=Path(self.training['output'])/f'cfg_{cfg:g}/nfe_{nfe}';r=json.loads((dest/'result.json').read_text());shape=Path(r['shape_file']).resolve()
        if not shape.is_relative_to(dest.resolve()) or not r['complete'] or r['protocol_sha256']!=protocol_hash(self.training):raise ValueError('Training provenance mismatch')
        if r['source_sha256']!=source_hashes() or digest(shape)!=r['sha256']:raise ValueError('Shape source/checksum mismatch')
        read_shapes(shape,nfe,self.training)
        def velocity(x,t,labels):return guided_velocity(self.model,x,t,labels,cfg)
        self.solvers[key]=load_trained_solver(velocity,shape,nfe,self.training);self.learned[key]=r

    def latent(self,job,offset,b):
        if job['method']!='rbf':return super().latent(job,offset,b)
        self.bind(job['cfg'],job['nfe']);solver=self.solvers[(job['cfg'],job['nfe'])]
        z=self.bank['noise'][offset:offset+b].to(self.device);y=self.bank['labels'][offset:offset+b].to(self.device)
        value=solver.sample(z,model_kwargs={'labels':y})
        if solver.last_nfe!=job['nfe']:raise RuntimeError('Actual NFE mismatch')
        return value

    def run_job(self,job):
        self.base['adapter']='trained_rbf_velocity_fm' if job['method']=='rbf' else 'velocity_to_x0_exact_endpoint_limit'
        if job['method']=='rbf':
            self.bind(job['cfg'],job['nfe']);r=self.learned[(job['cfg'],job['nfe'])]
            self.base.update(shape_file=r['shape_file'],shape_sha256=r['sha256'])
        else:
            self.base.pop('shape_file',None);self.base.pop('shape_sha256',None)
        return super().run_job(job)


def dispatch(stage,p,queue,identity):
    """Keep the inherited queue/worker methods; dispatch this CFG-general wrapper."""
    out=Path(p['output']);out.mkdir(parents=True,exist_ok=True)
    ph=plan_digest(p) if stage=='sample' else protocol_hash(p)
    for j in queue:
        j['state']='pending';dest=out/j['id'];dest.mkdir(parents=True,exist_ok=True)
        if (dest/'error.json').exists():(dest/'error.json').rename(dest/f'error_previous_{time.time_ns()}.json')
        if stage=='sample' and (dest/'result.json').exists():
            r=json.loads((dest/'result.json').read_text())
            if r['config']['plan_sha256']!=ph or r['config'].get('execution_source_sha256')!=execution_identity():raise ValueError('Existing sampling identity changed')
            j['state']='complete'
    atomic_json(out/'queue.json',dict(plan_sha256=ph,jobs=queue,stage=stage,cfgs=plan()['cfgs']))
    if stage=='sample':atomic_json(out/'manifest.json',dict(protocol=p,source_sha256=source_hashes(),execution_source_sha256=execution_identity(),created_at=time.time()))
    procs=[]
    with contextlib.ExitStack() as stack:
        try:
            for gpu in p['gpus']:
                log=stack.enter_context(open(out/f'followup_{stage}_gpu{gpu}.log','a'))
                proc=subprocess.Popen([str(PYTHON),str(ROOT/'rbf_followup.py'),'--worker',str(gpu),'--stage',stage],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1'),stdout=log,stderr=subprocess.STDOUT)
                procs.append((gpu,proc))
            workers={str(g):v.pid for g,v in procs}
            atomic_json(out/'run_state.json',dict(state='running',stage=stage,workers=workers,started_at=time.time(),**identity))
            codes={str(g):v.wait() for g,v in procs};q=json.loads((out/'queue.json').read_text())
            good=all(c==0 for c in codes.values()) and all(j['state']=='complete' for j in q['jobs'])
            atomic_json(out/'run_state.json',dict(state='complete' if good else 'error',stage=stage,workers=workers,exit_codes=codes,finished_at=time.time(),**identity))
            if not good:raise RuntimeError(f'{stage} failed: {codes}')
        finally:
            for _,v in procs:
                if v.poll() is None:v.terminate()
            for _,v in procs:v.wait()


def run():
    sp=plan();p=load_training_plan(True);out=Path(p['output']);out.mkdir(parents=True,exist_ok=True);request=out/'followup_request.json'
    record=dict(cfgs=sp['cfgs'],nfes=sp['nfes'],stages=['target','fit','sample'],pid=os.getpid(),execution_source_sha256=execution_identity())
    atomic_json(request,dict(record,state='queued',stage='waiting_for_resources',updated_at=time.time()))
    with contextlib.ExitStack() as stack:
        # Blocking lock queues the continuation without interrupting the active run.
        for file in [Path(load_plan()['output'])/'launcher.lock',out/'launcher.lock']:
            lock=stack.enter_context(open(file,'a'));fcntl.flock(lock,fcntl.LOCK_EX)
        selection=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else {}
        if selection.get('cfgs')==[1.5] and selection.get('state')!='complete':raise RuntimeError('CFG 1.5 did not complete; continuation must not skip a failed run')
        busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if busy:raise RuntimeError('GPU compute process remains active')
        identity=prepare(p)
        atomic_json(out/'followup_manifest.json',dict(record,protocol=p,source_sha256=source_hashes(),created_at=time.time()))
        with urllib.request.urlopen('http://127.0.0.1:8765/api/rbf-training',timeout=10) as f:visible=json.load(f)
        if sp not in visible.get('sampling',{}).get('protocols',[]):raise RuntimeError('Dashboard is missing follow-up scope')
        try:
            # Finish each low-NFE comparison before advancing to a higher NFE.
            sequence=([('target',p['nfes'])]+[(stage,[n]) for n in p['nfes'] for stage in ('fit','sample')]
                      if sp.get('conditioning')=='unconditional_null_label'
                      else [(stage,p['nfes']) for stage in ('target','fit','sample')])
            for stage,selected_nfes in sequence:
                state=dict(record,state='running',stage=stage,active_nfes=selected_nfes,updated_at=time.time())
                atomic_json(request,state);atomic_json(out/'selection.json',state)
                if stage=='target':
                    queue=[];groups={}
                    for cfg in sp['cfgs']:
                        if (out/f'cfg_{cfg:g}/teacher/targets.pt').exists():load_targets(p,cfg);continue
                        group=[]
                        for i,_ in enumerate(p['gpus']):
                            a=p['target_pairs']*i//len(p['gpus']);b=p['target_pairs']*(i+1)//len(p['gpus'])
                            group.append(dict(id=f'cfg_{cfg:g}/teacher/shard_{i}',cfg=cfg,method=p['teacher']['method'],nfe=p['teacher']['nfe'],offset=a,stop=b))
                        queue.extend(group);groups[cfg]=group
                    if queue:dispatch(stage,p,queue,identity)
                    for cfg,group in groups.items():merge_targets(p,cfg,group)
                elif stage=='fit':
                    for cfg in sp['cfgs']:load_targets(p,cfg)
                    dispatch(stage,p,[dict(id=f'cfg_{cfg:g}/nfe_{n}',cfg=cfg,nfe=n) for cfg in sp['cfgs'] for n in selected_nfes],identity)
                    for cfg in sp['cfgs']:
                        subprocess.run([str(PYTHON),str(ROOT/'plot_rbf_coefficients.py'),'--cfg',str(cfg)],cwd=ROOT,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'))
                else:
                    dispatch(stage,sp,[j for j in jobs(sp) if j['nfe'] in selected_nfes],dict(plan_sha256=plan_digest(sp),execution_source_sha256=execution_identity()))
            state=dict(record,state='complete',stage='sample',updated_at=time.time());atomic_json(request,state);atomic_json(out/'selection.json',state)
        except Exception as exc:
            state=dict(record,state='error',stage=stage,error=repr(exc),updated_at=time.time());atomic_json(request,state);atomic_json(out/'selection.json',state);raise
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');parser.add_argument('--worker',type=int,choices=[0,1]);parser.add_argument('--stage',choices=['target','fit','sample']);a=parser.parse_args()
    if a.worker is not None:
        manifest=json.loads((Path(load_training_plan()['output'])/'followup_manifest.json').read_text())
        if manifest['execution_source_sha256']!=execution_identity():raise ValueError('Continuation source changed after launch')
        if a.stage=='sample':FollowupSamplingWorker(a.worker).run()
        elif a.stage in ('target','fit'):StageWorker(a.worker,a.stage).run()
        else:raise ValueError('Worker stage required')
        return 0
    if not a.run:raise ValueError('Explicit --run is required')
    return run()

if __name__=='__main__':raise SystemExit(main())
