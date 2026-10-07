"""Run both GPUs against one locked condition queue; no unapproved defaults."""
from paths import ROOT,PYTHON,validate_storage
from experiment import load_plan,plan_digest,jobs,atomic_json
import os,json,subprocess,fcntl,urllib.request,time,hashlib
from pathlib import Path
import torch

def ensure_inputs(p):
 out=Path(p['output']);out.mkdir(parents=True,exist_ok=True)
 bank=out/'inputs.pt'
 if not bank.exists():
  gen=torch.Generator(device='cpu').manual_seed(p['seed'])
  noise=torch.randn(p['num_samples'],*p['data_shape'],generator=gen)
  labels=torch.zeros(p['num_samples'],dtype=torch.int64)
  tmp=bank.with_suffix('.tmp'); torch.save(dict(noise=noise,labels=labels,seed=p['seed']),tmp); tmp.replace(bank)
 else:
  data=torch.load(bank,map_location='cpu',weights_only=True)
  assert data['seed']==p['seed'] and data['noise'].shape==(p['num_samples'],*p['data_shape'])
 return bank


def main():
 p=load_plan(require_ready=True); validate_storage(); out=Path(p['output']); out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'launcher.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip(),'GPUs must be free before changing queues'
 ph=plan_digest(p)
 status=json.load(urllib.request.urlopen('http://127.0.0.1:8766/api/status',timeout=10))
 assert status['protocol']==p and status['conditions']==len(jobs(p)) and status['total']==len(jobs(p))*p['num_samples'],'Dashboard must show exact plan before sampling'
 bank=ensure_inputs(p)
 qfile=out/'queue.json'
 if qfile.exists():
  prior=json.loads(qfile.read_text()); assert prior['plan_sha256']==ph,'Refuse changing an existing run protocol'
 q=dict(plan_sha256=ph,jobs=[dict(j,state='complete' if (out/j['id']/'result.json').exists() else 'pending') for j in jobs(p)])
 atomic_json(qfile,q)
 hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}
 for f in (ROOT/'vendor/samplers').glob('*.py'): hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 if not (out/'manifest.json').exists():atomic_json(out/'manifest.json',dict(protocol=p,source_sha256=hashes,created_at=time.time()))
 from rbf_pipeline import sampling_plan
 from rbf_training_run import source_hashes
 from benchmark import digest
 sp=sampling_plan()
 visible=json.load(urllib.request.urlopen('http://127.0.0.1:8766/api/rbf-training',timeout=10))
 assert visible['sampling']['protocol']==sp,'RBF plan must be visible before sampling'
 groups=[p,sp];pending=[];bank_hash=digest(bank)
 for plan in groups:
  dest=Path(plan['output']);dest.mkdir(parents=True,exist_ok=True);rows=[]
  for job in jobs(plan):
   file=dest/job['id']/'result.json'
   if file.exists():
    r=json.loads(file.read_text());c=r['config']
    assert c['plan_sha256']==plan_digest(plan) and c['n']==plan['num_samples'] and c['input_sha256']==bank_hash and r['actual_nfe_per_image']==job['nfe'],'Completed result identity mismatch'
    state='complete'
   else:
    pending.append((plan,job));state='pending'
    error=dest/job['id']/'error.json'
    if error.exists():error.rename(error.with_name('error_previous_'+str(time.time_ns())+'.json'))
   rows.append(dict(job,state=state))
  atomic_json(dest/'queue.json',dict(plan_sha256=plan_digest(plan),jobs=rows))
  if not (dest/'manifest.json').exists():
   atomic_json(dest/'manifest.json',dict(protocol=plan,source_sha256=source_hashes(),created_at=time.time()))
 pending.sort(key=lambda entry:entry[1]['nfe'])
 handles={};dispatch=[];schedule=out.parent/'sampling_schedule.json'
 while pending or handles:
  for gpu,(proc,log,plan,job) in list(handles.items()):
   code=proc.poll()
   if code is None:continue
   log.close();del handles[gpu]
   if code:
    atomic_json(schedule,dict(state='error',error=f"Worker {proc.pid} exited {code}",dispatch=dispatch,updated_at=time.time()))
    raise RuntimeError(f"Sampling failed: {job['id']}; other active work is preserved")
   result=json.loads((Path(plan['output'])/job['id']/'result.json').read_text())
   assert result['config']['plan_sha256']==plan_digest(plan) and result['config']['input_sha256']==bank_hash
  for gpu in p['gpus']:
   if gpu in handles or not pending:continue
   occupied=subprocess.check_output(['nvidia-smi','--id='+str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
   if occupied:raise RuntimeError(f'GPU {gpu} is occupied; do not overlap sampling workers')
   plan,job=pending.pop(0);dest=Path(plan['output'])
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',PYTORCH_ALLOC_CONF='expandable_segments:True',RF_JOB_ID=job['id'])
   command=([str(PYTHON),str(ROOT/'rbf_pipeline.py'),'--worker',str(gpu),'--stage','sample','--cfg','1']
            if job['method']=='rbf' else [str(PYTHON),str(ROOT/'benchmark.py'),'--gpu',str(gpu)])
   log=open(dest/f'gpu{gpu}.log','a')
   proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
   handles[gpu]=(proc,log,plan,job)
   dispatch.append(dict(job,gpu=gpu,pid=proc.pid,started_at=time.time()))
   print(json.dumps(dict(event='dispatch',**dispatch[-1])),flush=True)
  for plan in groups:
   selected={str(gpu):v[0].pid for gpu,v in handles.items() if v[2]['output']==plan['output']}
   atomic_json(Path(plan['output'])/'run_state.json',dict(state='running',stage='sample',workers=selected,plan_sha256=plan_digest(plan),updated_at=time.time(),execution_order='ascending NFE among unfinished conditions'))
  atomic_json(schedule,dict(state='running',priority='lowest unfinished NFE',dispatch=dispatch,pending=[j for _,j in pending],active=[v[3] for v in handles.values()],updated_at=time.time()))
  if handles:time.sleep(1)
 for plan in groups:
  rows=[json.loads((Path(plan['output'])/j['id']/'result.json').read_text()) for j in jobs(plan)]
  assert len(rows)==len(jobs(plan)) and len({r['config']['input_sha256'] for r in rows})==1
  for r in rows:assert r['config']['n']==plan['num_samples'] and r['config']['order']==plan['order'] and r['actual_nfe_per_image']==r['config']['nfe']
  atomic_json(Path(plan['output'])/'comparison.json',dict(complete=True,results=rows))
  atomic_json(Path(plan['output'])/'run_state.json',dict(state='complete',workers={},plan_sha256=plan_digest(plan),finished_at=time.time()))
 atomic_json(schedule,dict(state='complete',priority='lowest unfinished NFE',dispatch=dispatch,pending=[],active=[],finished_at=time.time()))
 return 0
if __name__=='__main__': raise SystemExit(main())
