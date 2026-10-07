"""Read-only CFG comparison dashboard; status is derived from actual job records."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse,parse_qs
import argparse,json,subprocess,time,threading
from experiment import load_plan,jobs,job_id,plan_digest
from paths import ROOT,RESULTS, PYTHON
GPU_CACHE={'at':0.,'value':[]}; GPU_LOCK=threading.Lock()

def read_json(path):
    try: return json.loads(path.read_text())
    except (ValueError,OSError): return None

def worker_alive(pid,program='benchmark.py'):
    if not isinstance(pid,int): return False
    try:
        cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
        programs=[program]+(['rbf_pipeline.py','rbf_followup.py'] if program=='rbf_training_run.py' else ['rbf_followup.py'] if program=='rbf_pipeline.py' else [])
        return str(PYTHON).encode() in cmd and any(name.encode() in cmd for name in programs)
    except OSError: return False

def gpus():
    with GPU_LOCK:
        if time.monotonic()-GPU_CACHE['at']<10: return GPU_CACHE['value']
        data=[]
        try:
            out=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=5)
            for row in out.splitlines():
                i,name,util,used,total,temp,power=[p.strip() for p in row.split(',')]
                data.append(dict(index=int(i),name=name,util=int(util),used=int(used),total=int(total),temperature=int(temp),power=float(power)))
        except (OSError,subprocess.SubprocessError,ValueError): pass
        GPU_CACHE.update(at=time.monotonic(),value=data)
        return data

def status(p=None, program='benchmark.py'):
    p=load_plan() if p is None else p; output=Path(p['output']); run=read_json(output/'run_state.json') or {}
    workers=run.get('workers',{}); items=[]; rates={}; running=False
    for j in jobs(p):
        dest=output/j['id']; result=read_json(dest/'result.json'); state=read_json(dest/'status.json') or {}
        alive=worker_alive(state.get('pid'),program)
        done=state.get('done',0); seconds=state.get('seconds',0.); fid=None; error=state.get('error')
        failure=read_json(dest/'error.json')
        if failure: error=failure.get('error')
        label='pending' if p['ready'] else 'awaiting_configuration'
        if result:
            label='complete'; done=result['config']['n']; seconds=result['total_seconds']; fid=result['fid']
        elif state:
            label=state.get('state','pending')
            if label in ('running','evaluating','starting','tuning') and not alive: label='paused'
        if failure: label='error'
        if label in ('running','evaluating','starting','tuning'): running=True
        rate=done/seconds if seconds>0 and done else None
        if rate: rates[j['method']]=rate
        image=dest/'preview.png'
        items.append(dict(**j,state=label,done=done,total=p['num_samples'],seconds=seconds,fid=fid,
                          images_per_second=rate,worker_alive=alive,error=error,
                          gpu=(result or {}).get('gpu',state.get('gpu')),batch_size=(result or {}).get('batch_size',state.get('batch_size')),
                          updated_at=state.get('updated_at'),result=result,preview=f"/preview/{j['cfg']:g}/{j['method']}/{j['nfe']}" if image.is_file() else None))
    completed=sum(i['state']=='complete' for i in items)
    active_rates=[i['images_per_second'] for i in items if i['state']=='running' and i['images_per_second']]
    remaining=sum(i['total']-i['done'] for i in items)
    estimate=remaining/sum(active_rates) if active_rates else None
    return dict(updated_at=time.time(),items=items,gpus=gpus(),done=sum(i['done'] for i in items),
                total=len(items)*p['num_samples'] if p['num_samples'] else None,
                completed=completed,conditions=len(items),running=running,workers_active=any(worker_alive(v,program) for v in workers.values()),
                eta_seconds=0 if completed==len(items) else estimate,
                output=p['output'],fid_parity=read_json(RESULTS/'fid_parity.json'),protocol=p,gpu_workers=[read_json(output/f'gpu{g}.json') or dict(gpu=g,state='not_started') for g in p['gpus']],run_state=run.get('state','not_started'))


def training_plans():
    return [p for name in ('rbf_training.json','unconditional_training.json','unconditional_extended_training.json','unconditional_retrain_training.json') if (p:=read_json(ROOT/name))]


def coefficient_root(cfg):
    try:value=float(cfg)
    except (TypeError,ValueError):return None
    matches=[p for p in training_plans() if value in p['cfgs']]
    p=next((p for p in reversed(matches) if (Path(p['output'])/f'cfg_{value:g}/analysis/coefficients.json').is_file()),matches[-1] if matches else None)
    return Path(p['output'])/f'cfg_{value:g}/analysis' if p else None


def rbf_status():
    """Merge display only; validate completed conditions against their own original plan."""
    parts=[training_status(p) for p in training_plans()]
    selected=next((r for r in reversed(parts) if r.get('selection') and r['selection']['state'] in ('running','queued','error')),parts[-1])
    p=dict(parts[0]['protocol'],cfgs=sorted({c for r in parts for c in r['protocol']['cfgs']}),nfes=sorted({n for r in parts for n in r['protocol']['nfes']}))
    items=[]
    for part in parts:
        for row in part['items']:
            previous=next((i for i in items if (i['cfg'],i['nfe'])==(row['cfg'],row['nfe'])),None)
            if previous is not None:
                row=dict(row,previous=previous,retrained=True);items.remove(previous)
            items.append(row)
    return dict(selected,protocol=p,protocols=[r['protocol'] for r in parts],
                items=items,targets=list({i['cfg']:i for r in parts for i in r['targets']}.values()),
                conditions=len(items),completed=sum(i['state']=='complete' for i in items),
                ready=all(r['ready'] for r in parts))


def training_status(p):
    """Read one immutable training plan without starting work."""
    if not p:
        raise RuntimeError('RBF preparation plan is missing or invalid')
    output = Path(p['output']).resolve()
    if output == RESULTS.resolve() or not output.is_relative_to(RESULTS.resolve()):
        raise RuntimeError('Invalid RBF output path')
    ph = plan_digest({k:v for k,v in p.items() if k not in ('ready','phase','preparation_validation')})
    run = read_json(output/'run_state.json') or {}
    targets, items = [], []
    active = ('loading','tuning','teacher','training','averaging','running')
    for cfg in p['cfgs']:
        base = output/f'cfg_{cfg:g}'
        failure = read_json(base/'error.json')
        target = read_json(base/'teacher/status.json') or dict(state='pending',done=0,total=p['target_pairs'])
        shards=[read_json(f) for f in sorted((base/'teacher').glob('shard_*/status.json'))]
        shards=[r for r in shards if r]
        if shards and target['state']!='complete':
            for r in shards:
                if r['state'] in active and not worker_alive(r.get('pid'),'rbf_training_run.py'):r['state']='paused'
            running_shard=next((r for r in shards if r['state'] in active),None)
            shard_error=next((read_json(f) for f in sorted((base/'teacher').glob('shard_*/error.json')) if read_json(f)),None)
            label='error' if shard_error else running_shard['state'] if running_shard else 'paused' if any(r['state']=='paused' for r in shards) else 'merging' if sum(r['done'] for r in shards)==p['target_pairs'] else 'pending'
            target=dict(state=label,done=sum(r['done'] for r in shards),total=p['target_pairs'],shards=shards,
                        pid=running_shard.get('pid') if running_shard else None,
                        updated_at=max(r.get('updated_at',0) for r in shards),error=shard_error.get('error') if shard_error else None)
        if target['state'] in active and not worker_alive(target.get('pid'),'rbf_training_run.py'):
            target['state'] = 'paused'
        if failure and target['state'] != 'complete':
            target.update(state='error',error=failure.get('error'))
        targets.append(dict(target,cfg=cfg))
        for nfe in p['nfes']:
            dest = base/f'nfe_{nfe}'
            item = read_json(dest/'status.json') or dict(state='pending',done=0,total=p['repetitions'])
            result = read_json(dest/'result.json')
            if result and result.get('complete') and result.get('protocol_sha256') == ph:
                item.update(state='complete',done=p['repetitions'],result=result)
            elif failure:
                item.update(state='error',error=failure.get('error'))
            elif item['state'] in active and not worker_alive(item.get('pid'),'rbf_training_run.py'):
                item['state'] = 'paused'
            items.append(dict(item,cfg=cfg,nfe=nfe))
    workers = []
    for gpu in p['gpus']:
        w = read_json(output/f'gpu{gpu}.json') or dict(gpu=gpu,state='not_started')
        w['alive'] = worker_alive(w.get('pid'),'rbf_training_run.py')
        if w['state'] in active and not w['alive']:
            w['state'] = 'paused'
        workers.append(w)
    state = run.get('state','not_started')
    if state == 'running' and not any(w['alive'] for w in workers):
        state = 'paused'
    return dict(protocol=p,conditions=len(items),items=items,targets=targets,gpu_workers=workers,
                completed=sum(i['state']=='complete' for i in items),run_state=state,
                updated_at=time.time(),ready=p['ready'],selection=read_json(output/'selection.json'),
                followup=read_json(output/'followup_request.json'),sampling=rbf_sampling_status(),capabilities=dict(target=True,fit=True,sample=(ROOT/'rbf_pipeline.py').is_file(),independent_stages=True))


def rbf_sampling_status():
    plans=[read_json(ROOT/name) for name in ['rbf_sampling.json','rbf_followup_sampling.json','unconditional_sampling.json','unconditional_extended_sampling.json','unconditional_retrain_sampling.json']]
    plans=[p for p in plans if p];rows=[];runs=[];workers=[]
    for p in plans:
        output=Path(p['output']).resolve()
        if output==RESULTS.resolve() or not output.is_relative_to(RESULTS.resolve()):raise ValueError('Invalid RBF output')
        for j in jobs(p):
            if j['method']!='rbf':continue
            dest=output/j['id'];st=read_json(dest/'status.json') or dict(state='pending',done=0,total=p['num_samples'])
            r=read_json(dest/'result.json');error=read_json(dest/'error.json')
            if r:st.update(state='complete',done=r['config']['n'],fid=r['fid'],seconds=r['total_seconds'],images_per_second=r['images_per_second'],batch_size=r['batch_size'],gpu=r['gpu'],result=r)
            elif st['state'] in ('running','tuning','starting','evaluating') and not worker_alive(st.get('pid'),'rbf_pipeline.py'):st['state']='paused'
            if error:st.update(state='error',error=error.get('error'))
            if (dest/'preview.png').is_file():st['preview']=f"/rbf-preview/{j['cfg']:g}/{j['nfe']}"
            row=dict(st,**j)
            if p.get('retrain_of'):
                previous=next(i for i in rows if (i['cfg'],i['nfe'])==(j['cfg'],j['nfe']))
                row.update(previous=previous,retrained=True,retrain_of=p['retrain_of'])
                if previous.get('preview'):previous['preview']+='?variant=original'
                rows.remove(previous)
            rows.append(row)
        run=read_json(output/'run_state.json') or dict(state='not_started')
        if run['state']=='running' and not any(worker_alive(pid,'rbf_pipeline.py') for pid in run.get('workers',{}).values()):run['state']='paused'
        runs.append(run)
        current=[read_json(output/f'gpu{g}.json') or dict(gpu=g,state='not_started') for g in p['gpus']]
        if not workers or run['state'] in ('running','error','paused') or all(w['state']=='not_started' for w in workers):workers=current
    state='complete' if rows and all(r['state']=='complete' for r in rows) else 'error' if any(r['state']=='error' for r in runs) else 'running' if any(r['state']=='running' for r in runs) else 'not_started'
    return dict(protocol=plans[0] if plans else None,protocols=plans,items=rows,conditions=len(rows),done=sum(r['done'] for r in rows),total=sum(r['total'] for r in rows),completed=sum(r['state']=='complete' for r in rows),run_state=state,gpu_workers=workers)


def comparison_status():
    """Join real result records for all three samplers without changing execution plans."""
    base=status();rbf=rbf_sampling_status();rows=base['items']+rbf['items']
    for name in ('unconditional_sampling.json','unconditional_extended_sampling.json'):
        extra_plan=read_json(ROOT/name)
        if not extra_plan:continue
        extra=status(extra_plan,'rbf_followup.py')
        rows += [i for i in extra['items'] if i['method']!='rbf']
        if extra['workers_active']:base['gpu_workers']=extra['gpu_workers']
    rows.sort(key=lambda i:(i['cfg'],i['nfe'],['dpmpp','unipc','rbf'].index(i['method'])))
    protocol=dict(base['protocol'],methods=base['protocol']['methods']+['rbf'],
                  cfgs=sorted({i['cfg'] for i in rows}),nfes=sorted({i['nfe'] for i in rows}),
                  conditioning={'0.0':'unconditional null label, all latent channels','positive_cfg':'original three-channel classifier-free guidance'})
    completed=sum(i['state']=='complete' for i in rows)
    active=[i for i in rows if i['state'] in ('running','tuning','evaluating','starting')]
    rate=sum(i.get('images_per_second') or 0 for i in active if i['state']=='running')
    total=sum(i['total'] for i in rows);done=sum(i['done'] for i in rows)
    state='complete' if completed==len(rows) else 'error' if any(i['state']=='error' for i in rows) else 'running' if active else 'pending'
    return dict(base,items=rows,protocol=protocol,run_protocols=[base['protocol'],*rbf['protocols']],conditions=len(rows),completed=completed,done=done,total=total,running=bool(active),run_state=state,eta_seconds=0 if state=='complete' else (total-done)/rate if rate else None)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/api/comparison':
            body=json.dumps(comparison_status(),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
        elif path=='/api/status':
            body=json.dumps(status(),ensure_ascii=False,allow_nan=False).encode(); mime='application/json; charset=utf-8'
        elif path=='/api/rbf-coefficients':
            cfg=parse_qs(urlparse(self.path).query).get('cfg',['1.5'])[0]
            root=coefficient_root(cfg)
            if root is None:self.send_error(404);return
            file=root/'coefficients.json'
            data=read_json(file)
            body=json.dumps(dict(ready=bool(data),updated_at=file.stat().st_mtime if data else None,**(data or {})),ensure_ascii=False,allow_nan=False).encode();mime='application/json; charset=utf-8'
        elif path.startswith('/rbf-coefficients/'):
            parts=path.split('/');name=parts[-1];cfg=parts[-2] if len(parts)==4 else '1.5'
            allowed={f'coefficients_{part}.{ext}' for part in ['all']+[f'nfe{n}' for p in training_plans() for n in p['nfes']] for ext in ['png','svg']}
            root=coefficient_root(cfg)
            if name not in allowed or root is None or len(parts) not in [3,4]:self.send_error(404);return
            file=root/name
            if not file.is_file():self.send_error(404);return
            body=file.read_bytes();mime='image/svg+xml' if name.endswith('.svg') else 'image/png'
        elif path=='/api/rbf-training':
            body=json.dumps(rbf_status(),ensure_ascii=False,allow_nan=False).encode(); mime='application/json; charset=utf-8'
        elif path in ('/dashboard.css','/dashboard.js'):
            body=(ROOT/path.lstrip('/')).read_bytes()
            mime='text/css; charset=utf-8' if path.endswith('.css') else 'text/javascript; charset=utf-8'
        elif path in ('/','/index.html'):
            body=(ROOT/'dashboard.html').read_bytes(); mime='text/html; charset=utf-8'
        elif path.startswith('/rbf-preview/'):
            parts=path.split('/');plans=[read_json(ROOT/f) for f in ['rbf_sampling.json','rbf_followup_sampling.json','unconditional_sampling.json','unconditional_extended_sampling.json','unconditional_retrain_sampling.json']];p=next((p for p in (plans if parse_qs(urlparse(self.path).query).get('variant')==['original'] else list(reversed(plans))) if p and len(parts)==4 and parts[2] in [f'{c:g}' for c in p['cfgs']] and parts[3] in [str(n) for n in p['nfes']]),None)
            if not p or len(parts)!=4 or parts[2] not in [f'{c:g}' for c in p['cfgs']] or parts[3] not in [str(n) for n in p['nfes']]:
                self.send_error(404);return
            file=Path(p['output'])/job_id('rbf',float(parts[2]),int(parts[3]))/'preview.png'
            if not file.is_file():self.send_error(404);return
            body=file.read_bytes();mime='image/png'
        elif path.startswith('/preview/'):
            parts=path.split('/'); plans=[load_plan(),read_json(ROOT/'unconditional_sampling.json'),read_json(ROOT/'unconditional_extended_sampling.json')]
            p=next((p for p in plans if p and len(parts)==5 and parts[2] in [f'{c:g}' for c in p['cfgs']] and parts[4] in [str(n) for n in p['nfes']]),None)
            if not p or len(parts)!=5 or parts[3] not in p['methods'] or parts[4] not in [str(n) for n in p['nfes']]:
                self.send_error(404); return
            image=Path(p['output'])/job_id(parts[3],float(parts[2]),int(parts[4]))/'preview.png'
            if not image.is_file(): self.send_error(404); return
            body=image.read_bytes(); mime='image/png'
        else: self.send_error(404); return
        self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.end_headers()
        try: self.wfile.write(body)
        except (BrokenPipeError,ConnectionResetError): pass
    def log_message(self,*args): pass

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--host',default='0.0.0.0'); p.add_argument('--port',type=int,default=8765)
    a=p.parse_args(); print(f'Dashboard {a.host}:{a.port}',flush=True)
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()
