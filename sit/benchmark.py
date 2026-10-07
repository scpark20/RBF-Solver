"""Plan-driven GPU worker: shared queue, measured batches, paired inputs, GPU FID."""
from paths import ROOT,WEIGHTS,RESULTS,validate_storage
from experiment import load_plan,jobs,plan_digest,atomic_json
from pathlib import Path
import os,time,json,hashlib,argparse,fcntl,gc,traceback,contextlib
import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from sampling import load_model,sample
from fid_gpu import Moments,extractor,frechet

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for c in iter(lambda:f.read(8<<20),b''): h.update(c)
 return h.hexdigest()

def clear_memory():
 gc.collect(); torch.cuda.empty_cache()

@contextlib.contextmanager
def locked(path):
 with open(path,'a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try: yield
  finally: fcntl.flock(f,fcntl.LOCK_UN)

class Worker:
 def __init__(self,gpu,*,plan=None,input_path=None):
  self.p=load_plan(require_ready=True) if plan is None else plan; self.gpu=gpu; self.out=Path(self.p['output'])
  self.out.mkdir(parents=True,exist_ok=True); self.pid=os.getpid(); self.hash=plan_digest(self.p)
  self.worker_file=self.out/f'gpu{gpu}.json'; self.event('loading',message='모델·VAE·GPU FID 로딩')
  self.device=torch.device('cuda:0'); torch.cuda.set_device(self.device)
  torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
  # CPU threads are operational, bounded by the two workers' available CPU affinity.
  torch.set_num_threads(max(1,len(os.sched_getaffinity(0))//len(self.p['gpus'])))
  validate_storage()
  for name in ['fid_parity.json','endpoint_validation.json']:
   if not json.loads((RESULTS/name).read_text()).get('passed'): raise RuntimeError(name+' must pass')
  bank_path=Path(input_path) if input_path is not None else self.out/'inputs.pt'
  self.bank=torch.load(bank_path,map_location='cpu',weights_only=True)
  self.input_hash=digest(bank_path)
  self.ckpt=WEIGHTS/'SiT-diffusers/SiT-S-2-256/transformer/diffusion_pytorch_model.safetensors'
  self.model=load_model(self.ckpt,self.device)
  self.vae=AutoencoderKL.from_pretrained(WEIGHTS/'SiT-diffusers/SiT-S-2-256/vae',local_files_only=True).eval().to(self.device).requires_grad_(False)
  self.inc=extractor(self.device).requires_grad_(False)
  self.stats=Moments(self.device)
  ref=np.load(self.p['reference']); self.ref_mu=torch.as_tensor(ref['mu'],device=self.device,dtype=torch.float64); self.ref_cov=torch.as_tensor(ref['sigma'],device=self.device,dtype=torch.float64)
  self.decode_limit=self.p['num_samples']; self.feature_limit=self.p['num_samples']; self.batch_cache={}; self.trials=[]
  self.base=dict(model='SiT-S/2',checkpoint_sha256=digest(self.ckpt),checkpoint_source='BiliSakura/SiT-diffusers',checkpoint_provenance='third-party converted; original training recipe not documented',n=self.p['num_samples'],seed=self.p['seed'],cfg_channels=3,precision=self.p['precision'],order=self.p['order'],lower_order_final=True,schedule=self.p['schedule'],prediction='velocity',adapter=self.p['adapter'],time_grid='time_uniform',t_start=self.p['t_start'],t_end=self.p['t_end'],denoise_to_zero=False,reference=self.p['reference'],reference_sha256=digest(self.p['reference']),extractor='torch-fidelity-0.4.0 inception-v3-compat float32',moments='float64 CUDA',torch_version=str(torch.__version__),input_sha256=self.input_hash,plan_sha256=self.hash)
 def event(self,state,**kw):
  self.event_state=dict(state=state,gpu=self.gpu,pid=self.pid,updated_at=time.time(),**kw)
  atomic_json(self.worker_file,self.event_state)
 def claim(self):
  with locked(self.out/'queue.lock'):
   q=json.loads((self.out/'queue.json').read_text())
   if q['plan_sha256']!=self.hash: raise RuntimeError('Plan changed during run')
   for j in q['jobs']:
    if j['state']=='pending':
     j.update(state='claimed',gpu=self.gpu,pid=self.pid); atomic_json(self.out/'queue.json',q); return j.copy()
  return None
 def finish(self,job,state):
  with locked(self.out/'queue.lock'):
   q=json.loads((self.out/'queue.json').read_text())
   next(j for j in q['jobs'] if j['id']==job['id'])['state']=state; atomic_json(self.out/'queue.json',q)
 def latent(self,job,offset,b):
  z=self.bank['noise'][offset:offset+b].to(self.device); y=self.bank['labels'][offset:offset+b].to(self.device)
  return sample(self.model,z,y,job['method'],job['nfe'],cfg=job['cfg'],order=self.p['order'],t_start=self.p['t_start'],t_end=self.p['t_end'],precision=self.p['precision'])[0]
 def trial(self,job,b):
  self.event('tuning',job=job['id'],stage='sampling',candidate=b,trials=self.trials)
  clear_memory(); torch.cuda.reset_peak_memory_stats(); began=time.perf_counter(); passed=False
  try:
   z=self.latent(job,0,b); torch.cuda.synchronize(); del z; passed=True
  except torch.OutOfMemoryError: pass
  peak=torch.cuda.max_memory_allocated(); clear_memory()
  record=dict(batch=b,passed=passed,peak_bytes=peak,seconds=time.perf_counter()-began)
  self.trials.append(record); self.event('tuning',job=job['id'],stage='sampling',candidate=b,trials=self.trials)
  return passed
 def tune(self,job,cap=None):
  cap=self.p['num_samples'] if cap is None else cap
  self.trials=[]; low=0; high=None; b=min(cap,self.batch_cache.get(job['method'],1))
  while True:
   if self.trial(job,b):
    low=b
    if b==cap: break
    b=min(cap,b+1 if not high and len(self.trials)==1 and job['method'] in self.batch_cache else b*2)
   else: high=b; break
  if high is not None:
   while high-low>1:
    b=(low+high)//2
    if self.trial(job,b): low=b
    else: high=b
  if not low: raise RuntimeError('Even one image does not fit GPU memory')
  # Every candidate executes the entire NFE path with model, VAE, Inception,
  # reference statistics and accumulator resident. Postprocessing is streamed.
  self.batch_cache[job['method']]=low
  report=dict(gpu=self.gpu,job=job['id'],batch_size=low,first_failing_batch=high,cap=cap,trials=self.trials,scope=getattr(self,'batch_scope','Full NFE sampling with all persistent allocations present; VAE and FID streamed in GPU chunks'),plan_sha256=self.hash)
  atomic_json(self.out/job['id']/f'batch_gpu{self.gpu}.json',report)
  return low
 def measured_batches(self,job):
  """Read successful SiT runs; a new NFE does not trigger a capacity search."""
  candidates=[]
  keys=('model','checkpoint_sha256','precision','order','reference_sha256','input_sha256')
  for directory in self.p.get('batch_sources',[]):
   root=Path(directory).resolve()
   if not root.is_relative_to(RESULTS.resolve()):raise ValueError('Invalid batch source')
   for file in root.glob('cfg_*/*/result.json'):
    r=json.loads(file.read_text());c=r['config']
    if r['gpu']!=self.gpu or c['method']!=job['method']:continue
    if any(c.get(k)!=self.base.get(k) for k in keys):continue
    b=r['batch_size']
    if not isinstance(b,int) or b<1:raise ValueError('Invalid successful batch record')
    passed={b}
    report=file.parent/f'batch_gpu{self.gpu}.json'
    if report.exists():
     trial=json.loads(report.read_text())
     if trial['gpu']==self.gpu and trial['plan_sha256']==c['plan_sha256']:
      passed.update(t['batch'] for t in trial['trials'] if t['passed'])
    candidates.append(dict(batch_size=b,passed=sorted(passed),source=str(file),decode_chunk=r['decode_chunk'],fid_chunk=r['fid_chunk']))
  if not candidates:raise RuntimeError('No compatible successful SiT batch; refusing an unsolicited search')
  return candidates
 def reuse_batch(self,job):
  from capacity import reuse
  return reuse(self,job)
 def recover_batch(self,job,failed_batch):
  from capacity import recover
  return recover(self,job,failed_batch)
 def decode(self,x):
  return (127.5*self.vae.decode(x/self.vae.config.scaling_factor).sample+128).clamp(0,255).to(torch.uint8)
 def features(self,images):
  off=0
  while off<len(images):
   b=min(self.feature_limit,len(images)-off)
   try: f=self.inc(images[off:off+b])[0]
   except torch.OutOfMemoryError:
    if b==1: raise
    clear_memory(); self.feature_limit=b//2; continue
   yield f; off+=b
 def image_chunks(self,latents):
  off=0
  while off<len(latents):
   b=min(self.decode_limit,len(latents)-off)
   try: images=self.decode(latents[off:off+b])
   except torch.OutOfMemoryError:
    if b==1: raise
    clear_memory(); self.decode_limit=b//2; continue
   yield off,images; off+=b
 def postprocess(self,latents,update,preview=False):
  first=[]
  for offset,images in self.image_chunks(latents):
   if preview and offset<64: first.append(images[:64-offset].cpu())
   for feat in self.features(images):
    if update: self.stats.update(feat)
   del images
  return torch.cat(first) if first else None
 def status(self,dest,state,done,seconds,batch,**kw):
  atomic_json(dest/'status.json',dict(state=state,done=done,total=self.p['num_samples'],seconds=seconds,batch_size=batch,gpu=self.gpu,pid=self.pid,updated_at=time.time(),**kw))
 @torch.inference_mode()
 def run_job(self,job):
  dest=self.out/job['id']; dest.mkdir(parents=True,exist_ok=True)
  config=dict(self.base,method=job['method'],cfg=job['cfg'],nfe=job['nfe'])
  if job['cfg']==0:
   config.update(conditioning='unconditional_null_label',null_label=self.model.y_embedder.num_classes,cfg_channels=0,velocity_channels=self.model.in_channels)
  checkpoint=dest/'progress.pt'; done=0; seconds=0.
  self.stats=Moments(self.device)
  if checkpoint.exists():
   ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
   if ck['config']!=config: raise RuntimeError('Resume configuration mismatch')
   self.stats.restore(ck['moments']); done=ck['done']; seconds=ck['seconds']
  batch=self.reuse_batch(job)
  self.event('running',job=job['id'],batch_size=batch,decode_chunk=self.decode_limit,fid_chunk=min(self.feature_limit,self.decode_limit))
  while done<self.p['num_samples']:
   b=min(batch,self.p['num_samples']-done); torch.cuda.synchronize(); began=time.perf_counter()
   self.status(dest,'running',done,seconds,batch)
   clear_memory()
   oom=False
   try: lat=self.latent(job,done,b)
   except torch.OutOfMemoryError:
    if b==1: raise
    oom=True
   # Leave the exception handler before probing, so its traceback cannot
   # retain the failed allocation and bias the measured capacity.
   if oom:
    clear_memory()
    batch=self.recover_batch(job,b)
    self.status(dest,'running',done,seconds,batch,message='OOM: reuse a smaller previously successful batch')
    self.event('running',job=job['id'],batch_size=batch,decode_chunk=self.decode_limit,fid_chunk=min(self.feature_limit,self.decode_limit))
    continue
   preview=self.postprocess(lat,True,preview=done==0); del lat
   torch.cuda.synchronize(); seconds+=time.perf_counter()-began; done+=b
   if preview is not None:
    save_image(preview.float()/255,dest/'preview.png',nrow=8); del preview
   state=dict(config=config,moments=self.stats.state(),done=done,seconds=seconds)
   tmp=checkpoint.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(checkpoint)
   self.status(dest,'running',done,seconds,batch)
   print(json.dumps(dict(gpu=self.gpu,job=job['id'],done=done,batch=batch,seconds=seconds)),flush=True)
  self.status(dest,'evaluating',done,seconds,batch); self.event('evaluating',job=job['id'],batch_size=batch)
  mu,cov=self.stats.statistics(); fid=frechet(mu,cov,self.ref_mu,self.ref_cov)
  np.savez(dest/'statistics.npz',mu=mu.cpu().numpy(),sigma=cov.cpu().numpy(),n=self.stats.n)
  result=dict(config=config,fid=fid,total_seconds=seconds,images_per_second=done/seconds,actual_nfe_per_image=job['nfe'],network_examples_per_image=(1 if job['cfg'] in (0,1) else 2)*job['nfe'],gpu=self.gpu,batch_size=batch,decode_chunk=self.decode_limit,fid_chunk=min(self.feature_limit,self.decode_limit),timing_scope='sampling + VAE + GPU Inception/moments; excludes tuning, loading, file I/O and final eigensolve')
  atomic_json(dest/'result.json',result); self.status(dest,'complete',done,seconds,batch)
  self.finish(job,'complete'); print(json.dumps(result),flush=True)
 @torch.inference_mode()
 def run(self):
  while (job:=self.claim()) is not None:
   try: self.run_job(job)
   except Exception as exc:
    self.finish(job,'error'); atomic_json(self.out/job['id']/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
    self.event('error',job=job['id'],error=repr(exc)); raise
  self.event('complete',message='배정된 작업 완료')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--gpu',type=int,required=True); args=p.parse_args()
 Worker(args.gpu).run()
