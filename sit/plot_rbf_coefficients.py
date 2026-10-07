"""Render actual SiT coefficients using paper Appendix E, Eq.33 / Figure 5 layout."""
from pathlib import Path
import json, os, argparse
from paths import ROOT, RESULTS
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from benchmark import digest
from experiment import atomic_json
from rbf_training_run import load_training_plan,read_shapes,load_trained_solver,protocol_hash


def collect(cfg, p=None):
    """Verify all repeats and read the very coefficients prepared by the sampling solver."""
    p=load_training_plan(True) if p is None else p; conditions=[]
    for n in p['nfes']:
        dest=Path(p['output'])/f'cfg_{cfg:g}/nfe_{n}'
        if p.get('conditioning')=='unconditional_null_label' and not (dest/'result.json').exists():continue
        result=json.loads((dest/'result.json').read_text()); path=Path(result['shape_file'])
        if not result['complete'] or result['protocol_sha256']!=protocol_hash(p) or digest(path)!=result['sha256']:
            raise ValueError('Training result identity mismatch')
        logs=read_shapes(path,n,p)
        repeats=[]
        for k in range(p['repetitions']):
            file=dest/f'NFE={n},p={p["history_size"]},number={k}.npz'
            meta=json.loads(file.with_suffix('.json').read_text())
            if meta['sha256']!=digest(file):raise ValueError('Repeat checksum mismatch')
            repeats.append(read_shapes(file,n,p,with_losses=True))
        if not np.array_equal(np.mean(np.stack(repeats),axis=0),logs):
            raise ValueError('Stored log shapes are not the exact 20-run mean')
        solver=load_trained_solver(lambda x,t:x,path,n,p)
        pred=[list(v.predictor) for v in solver._prepared]
        corr=[list(v.corrector) if v.corrector is not None else None for v in solver._prepared]
        def ratios(c):
            if c is None:return None
            v=np.abs(c); r=v/v.sum()
            if not np.isclose(r.sum(),1,atol=1e-12):raise ValueError('CMR normalization mismatch')
            return r.tolist()
        for i,(cp,cc) in enumerate(zip(pred,corr)):
            h=solver.timesteps[i+1]-solver.timesteps[i]
            for c in [cp,cc]:
                if c is not None and not np.isclose(sum(c),h,atol=1e-12,rtol=1e-12):
                    raise ValueError('FM coefficient sum differs from time interval')
        conditions.append(dict(nfe=n,steps=list(range(n)),timesteps=solver.timesteps,
            raw_optimal_log_shapes=logs.tolist(),
            predictor_log_gamma=[float(logs[0,i]) if len(pred[i])>1 else None for i in range(n)],
            corrector_log_gamma=[float(logs[1,i]) if corr[i] is not None else None for i in range(n)],
            predictor_coefficients=pred,corrector_coefficients=corr,
            predictor_cmr=[ratios(c) for c in pred],corrector_cmr=[ratios(c) for c in corr],
            shape_file=str(path),shape_sha256=result['sha256'],repetitions=p['repetitions']))
    if p.get('name')=='SiT-S/2 CFG 0.0 / RBF shape training NFE 12–40':
        previous=collect(cfg,json.loads((ROOT/'unconditional_training.json').read_text()))
        conditions=previous['conditions']+conditions
        p=dict(p,nfes=previous['expected_nfes']+p['nfes'])
    if p.get('retrain_of'):
        previous=collect(cfg,json.loads((ROOT/'unconditional_extended_training.json').read_text()))
        replace={r['nfe']:r for r in conditions}
        conditions=[dict(replace[r['nfe']],previous=r,retrained=True) if r['nfe'] in replace else r for r in previous['conditions']]
        p=dict(p,nfes=previous['expected_nfes'])
    if not conditions:raise ValueError('No completed coefficients to plot')
    return dict(cfg=cfg,model='SiT-S/2',history_size=p['history_size'],conditions=conditions,
        complete=len(conditions)==len(p['nfes']),expected_nfes=p['nfes'],
        source='https://arxiv.org/html/2603.13330v1',source_detail='Appendix E, Eq. (33), Figure 5',
        cmr_definition='abs(c_j) / sum_k(abs(c_k))',coefficient_source='Actual FM solver prepared predictor/corrector coefficients',
        log_gamma_bounds=[p['log_gamma_min'],p['log_gamma_max']],
        validation=dict(exact_mean_of_20=True,shape_hashes_verified=True,cmr_sums_one=True,coefficient_sums_equal_fm_step=True),
        unused_parameters='Single-node predictors have no effective shape parameter; final corrector is not executed.')


def draw(data,selected,path):
    """Three aligned rows per NFE: learned log shape, predictor CMR, corrector CMR."""
    conditions=[r for r in data['conditions'] if selected is None or r['nfe']==selected]
    columns=len(conditions); fig,axes=plt.subplots(3,columns,figsize=(4.5*columns,6.7),squeeze=False,sharex='col')
    colors=['#2ca02c','#d62728','#9467bd','#8c564b']
    for k,r in enumerate(conditions):
        x=np.array(r['steps']); ax=axes[0,k]
        for values,color in [(r['predictor_log_gamma'],'#1f77b4'),(r['corrector_log_gamma'],'#ff7f0e')]:
            y=np.array([np.nan if v is None else v for v in values])
            ax.plot(x,y,color=color,marker='x',ms=5,lw=1.7)
        ax.set_title(f'NFE = {r["nfe"]}',fontsize=14,pad=9)
        a,b=data['log_gamma_bounds'];ax.set_ylim(a-.15,b+.15);ax.set_yticks([a,(a+b)/2,b])
        for row,key,style in [(1,'predictor_cmr','-'),(2,'corrector_cmr','--')]:
            series=r[key]; count=max(len(c) for c in series if c is not None)
            for j in range(count):
                y=[np.nan if c is None else c[j] if j<len(c) else 0 for c in series]
                axes[row,k].plot(x,y,style,color=colors[j],marker='x',ms=5,lw=1.5)
            axes[row,k].set_ylim(-.035,1.035);axes[row,k].set_yticks([0,.5,1])
        for row,ax in enumerate(axes[:,k]):
            ax.set_xticks(x);ax.grid(True,ls='--',lw=.6,alpha=.45);ax.tick_params(labelsize=12)
            ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(-.2,r['nfe']-1+.2)
        axes[2,k].set_xlabel('Step',fontsize=13)
    for row,label in enumerate([r'$\log\gamma$','Pred CMRs','Corr CMRs']):axes[row,0].set_ylabel(label,fontsize=13)
    fig.suptitle(f'SiT-S/2 · CFG {data["cfg"]:g} · learned shape parameters and coefficient magnitude ratios',fontsize=14,y=.995)
    handles=[Line2D([],[],color='#1f77b4',marker='x',label='Predictor log gamma'),Line2D([],[],color='#ff7f0e',marker='x',label='Corrector log gamma')]
    handles += [Line2D([],[],color=c,marker='x',label=f'CMR: j = {j}') for j,c in enumerate(colors[:3])]
    fig.legend(handles=handles,loc='lower center',ncol=5 if columns>1 else 2,frameon=False,fontsize=11,bbox_to_anchor=(.5,.005))
    fig.tight_layout(rect=(0,.055 if columns>1 else .095,1,.96),h_pad=1.3,w_pad=1.5)
    for suffix in ['.png','.svg']:
        temporary=path.with_suffix('.tmp'+suffix)
        fig.savefig(temporary,dpi=180,facecolor='white',metadata={'Creator':'SiT RBF coefficient analysis'})
        temporary.replace(path.with_suffix(suffix))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cfg',type=float,default=1.5,choices=load_training_plan()['cfgs']);a=parser.parse_args()
    data=collect(a.cfg);out=Path(load_training_plan()['output'])/f'cfg_{a.cfg:g}/analysis';out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'coefficients.json',data)
    draw(data,None,out/'coefficients_all')
    for r in data['conditions']:draw(data,r['nfe'],out/f'coefficients_nfe{r["nfe"]}')
    print(json.dumps(dict(output=str(out),validation=data['validation'],nfes=[r['nfe'] for r in data['conditions']]),ensure_ascii=False))

if __name__=='__main__':main()
