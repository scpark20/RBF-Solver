"""Verify CFG 0 is pure unconditional in every solver path and preserves positive CFG."""
import ast, json, types, unittest
from pathlib import Path
import torch
from paths import ROOT
import sampling
from rbf_training_run import guided_velocity

class LabelSensitiveModel:
    y_embedder=types.SimpleNamespace(num_classes=1000)
    in_channels=4
    def __init__(self):self.calls=[]
    def __call__(self,x,t,y):
        self.calls.append(y.clone())
        channels=torch.arange(1,5,dtype=x.dtype,device=x.device).reshape(1,4,1,1)
        return torch.ones_like(x)*channels*(y.to(x.dtype).reshape(-1,1,1,1)+1)/1001
    def forward_with_cfg(self,x,t,y,cfg):
        pred=self(x,t,y);a,b=pred.chunk(2)
        out=a.clone();out[:,:3]=b[:,:3]+cfg*(a[:,:3]-b[:,:3])
        rest=pred[:,3:]
        return torch.cat([torch.cat([out[:,:3],out[:,:3]]),rest],dim=1)

def previous_function(file,name,namespace):
    backup=ROOT/'tests/fixtures'
    module=ast.parse((backup/file).read_text())
    node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(backup/file),'exec'),namespace)
    return namespace[name]

class UnconditionalTests(unittest.TestCase):
    def setUp(self):
        self.z=torch.linspace(-1,1,2*4*4*4).reshape(2,4,4,4)
        self.y=torch.tensor([5,31]);self.t=torch.tensor([0.,.999])
    def test_null_velocity_all_channels_and_labels(self):
        model=LabelSensitiveModel()
        actual=guided_velocity(model,self.z,self.t,self.y,0.0)
        expected=model(self.z,self.t,torch.full_like(self.y,1000))
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        changed=guided_velocity(model,self.z,self.t,self.y+77,0.0)
        torch.testing.assert_close(actual,changed,rtol=0,atol=0)
        self.assertTrue(all(y.shape==self.y.shape and bool((y==1000).all()) for y in model.calls))
    def test_unconditional_solver_outputs_and_nfe(self):
        p=json.loads((ROOT/'experiment.json').read_text())
        for method in p['methods']:
            for nfe in sorted(set(p['nfes']+json.loads((ROOT/'unconditional_extended_sampling.json').read_text())['nfes'])):
                with self.subTest(method=method,nfe=nfe):
                    model=LabelSensitiveModel()
                    out,calls=sampling.sample(model,self.z,self.y,method,nfe,cfg=0.0,order=p['order'],t_start=p['t_start'],t_end=p['t_end'])
                    v=torch.arange(1,5).reshape(1,4,1,1)
                    torch.testing.assert_close(out,self.z+(p['t_start']-p['t_end'])*v,rtol=2e-4,atol=2e-4)
                    self.assertEqual(calls,nfe);self.assertEqual(len(model.calls),nfe)
                    self.assertTrue(all(y.shape==self.y.shape and bool((y==1000).all()) for y in model.calls))
    def test_positive_cfg_regression(self):
        old_sample=previous_function('sampling.py','sample',dict(vars(sampling)))
        old_velocity=previous_function('rbf_training_run.py','guided_velocity',dict(torch=torch))
        for cfg in [1.,1.5,3.5,5.5,7.5]:
            expected=old_velocity(LabelSensitiveModel(),self.z,self.t,self.y,cfg)
            actual=guided_velocity(LabelSensitiveModel(),self.z,self.t,self.y,cfg)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            for method in ['dpmpp','unipc']:
                args=dict(cfg=cfg,order=2,t_start=1.,t_end=.001)
                old=old_sample(LabelSensitiveModel(),self.z,self.y,method,5,**args)[0]
                new=sampling.sample(LabelSensitiveModel(),self.z,self.y,method,5,**args)[0]
                torch.testing.assert_close(old,new,rtol=0,atol=0)
    def test_dashboard_preserves_existing_results(self):
        import dashboard
        state=dashboard.comparison_status()
        self.assertEqual(state['protocol']['cfgs'],[0.,1.5,3.5,5.5,7.5])
        self.assertEqual(state['conditions'],81)
        old=[i for i in state['items'] if i['cfg']!=0]
        new=[i for i in state['items'] if i['cfg']==0]
        self.assertEqual(len(old),48);self.assertTrue(all(i['state']=='complete' for i in old))
        self.assertEqual(len(new),33)
        self.assertEqual(len({(i['cfg'],i['method'],i['nfe']) for i in state['items']}),81)
        rbf=dashboard.rbf_status()
        self.assertEqual(rbf['conditions'],27)
        self.assertEqual(len(rbf['targets']),5)
        self.assertEqual(dashboard.coefficient_root('0'),dashboard.coefficient_root('0.0'))
if __name__=='__main__':unittest.main()
