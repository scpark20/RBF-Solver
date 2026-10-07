"""Run both GPUs against one locked condition queue; no unapproved defaults."""
from paths import ROOT,validate_storage, PYTHON
from experiment import load_plan,plan_digest,jobs,atomic_json
import os,json,subprocess,fcntl,urllib.request,time,hashlib
from pathlib import Path
import torch

def ensure_inputs(p):
 out=Path(p['output']);out.mkdir(parents=True,exist_ok=True)
 bank=out/'inputs.pt'
 if not bank.exists():
  gen=torch.Generator(device='cpu').manual_seed(p['seed'])
  noise=torch.randn(p['num_samples'],4,32,32,generator=gen)
  labels=torch.randint(0,1000,(p['num_samples'],),generator=gen)
  tmp=bank.with_suffix('.tmp'); torch.save(dict(noise=noise,labels=labels,seed=p['seed']),tmp); tmp.replace(bank)
 else:
  data=torch.load(bank,map_location='cpu',weights_only=True)
  assert data['seed']==p['seed'] and data['noise'].shape==(p['num_samples'],4,32,32)
 return bank


def main():
 p=load_plan(require_ready=True); validate_storage(); out=Path(p['output']); out.mkdir(parents=True,exist_ok=True)
 lock=open(out/'launcher.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 ph=plan_digest(p)
 status=json.load(urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=10))
 assert status['protocol']==p and status['conditions']==32 and status['total']==320000,'Dashboard must show exact plan before sampling'
 bank=ensure_inputs(p)
 qfile=out/'queue.json'
 if qfile.exists():
  prior=json.loads(qfile.read_text()); assert prior['plan_sha256']==ph,'Refuse changing an existing run protocol'
 q=dict(plan_sha256=ph,jobs=[dict(j,state='complete' if (out/j['id']/'result.json').exists() else 'pending') for j in jobs(p)])
 atomic_json(qfile,q)
 hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.glob('*.py')}
 for f in (ROOT/'vendor/samplers').glob('*.py'): hashes[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 atomic_json(out/'manifest.json',dict(protocol=p,source_sha256=hashes,created_at=time.time()))
 handles=[]; workers={}
 for gpu in p['gpus']:
  env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
  log=open(out/f'gpu{gpu}.log','a')
  proc=subprocess.Popen([str(PYTHON),str(ROOT/'benchmark.py'),'--gpu',str(gpu)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
  handles.append((gpu,proc,log)); workers[str(gpu)]=proc.pid
 atomic_json(out/'run_state.json',dict(state='running',workers=workers,plan_sha256=ph,started_at=time.time()))
 codes={str(gpu):proc.wait() for gpu,proc,log in handles}
 for _,_,log in handles: log.close()
 rows=[json.loads((out/j['id']/'result.json').read_text()) for j in jobs(p) if (out/j['id']/'result.json').exists()]
 passed=len(rows)==32 and all(c==0 for c in codes.values())
 if passed:
  assert len({r['config']['input_sha256'] for r in rows})==1
  for r in rows:
   assert r['config']['n']==10000 and r['config']['order']==2 and r['actual_nfe_per_image']==r['config']['nfe']
 atomic_json(out/'comparison.json',dict(complete=passed,results=rows))
 atomic_json(out/'run_state.json',dict(state='complete' if passed else 'error',workers=workers,exit_codes=codes,plan_sha256=ph,finished_at=time.time()))
 print(json.dumps(dict(complete=passed,conditions=len(rows),exit_codes=codes)),flush=True)
 return 0 if passed else 1
if __name__=='__main__': raise SystemExit(main())
