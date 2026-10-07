"""Analytic constant-velocity flow checks signs, endpoints and actual NFE."""
import unittest
import json
from pathlib import Path
from experiment import load_plan
import torch
from sampling import sample,LinearFlowSchedule
class ConstantFlow:
    def __call__(self,x,t,y): return torch.ones_like(x)*.25
class SamplingTests(unittest.TestCase):
    def test_schedule_inverse(self):
        s=LinearFlowSchedule(); t=torch.linspace(.001,.999,101,dtype=torch.float64)
        torch.testing.assert_close(s.inverse_lambda(s.marginal_lambda(t)),t)
    def test_exact_flow_and_nfe(self):
        z=torch.randn(2,4,4,4,dtype=torch.float32)
        labels=torch.zeros(2,dtype=torch.long)
        p=load_plan()
        teacher=json.loads((Path(__file__).parent/'rbf_training.json').read_text())['teacher']
        for method in p['methods']:
            for nfe in p['nfes'] + ([teacher['nfe']] if method==teacher['method'] else []):
                with self.subTest(method=method,nfe=nfe):
                    out,calls=sample(ConstantFlow(),z,labels,method,nfe,cfg=1,order=p['order'],t_start=p['t_start'],t_end=p['t_end'])
                    # Physical SiT time advances by the solver interval length.
                    torch.testing.assert_close(out,z+(p['t_start']-p['t_end'])*.25,atol=2e-4,rtol=2e-4)
                    self.assertEqual(calls,nfe)
if __name__=='__main__': unittest.main()
