"""CPU checks for device identity and safe capacity reuse across NFEs."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import capacity
from paths import RESULTS

class CapacityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=RESULTS)
        self.addCleanup(self.temp.cleanup)
        self.worker=SimpleNamespace(out=Path(self.temp.name),gpu=0,hash="protocol",
                                    p={"num_samples":100},decode_limit=100,feature_limit=100)
        self.job={"method":"rbf","id":"cfg_0/rbf_nfe5"}
        self.record=dict(plan_sha256="protocol",environment={"device":"synthetic"},
                         methods={"rbf":dict(batch_size=8,passed=[1,2,4,8],failed=[],
                                             decode_chunk=4,fid_chunk=2)})
        self.patch=patch("capacity.environment",return_value={"device":"synthetic"})
        self.patch.start();self.addCleanup(self.patch.stop)
    def save(self):
        (self.worker.out/"capacity_gpu0.json").write_text(json.dumps(self.record))
    def test_missing_and_wrong_device_rejected(self):
        with self.assertRaises(RuntimeError):capacity.reuse(self.worker,self.job)
        self.record["environment"]={"device":"different"};self.save()
        with self.assertRaises(RuntimeError):capacity.reuse(self.worker,self.job)
    def test_plan_mismatch_rejected(self):
        self.record["plan_sha256"]="other";self.save()
        with self.assertRaises(RuntimeError):capacity.reuse(self.worker,self.job)
    def test_other_nfe_reuses_capacity_and_chunks(self):
        self.save()
        self.assertEqual(capacity.reuse(self.worker,self.job),8)
        self.assertEqual(capacity.reuse(self.worker,dict(self.job,id="cfg_0/rbf_nfe40")),8)
        self.assertEqual((self.worker.decode_limit,self.worker.feature_limit),(4,2))
    def test_oom_uses_only_measured_candidates(self):
        self.save()
        self.assertEqual(capacity.recover(self.worker,self.job,7),4)
        self.assertEqual(capacity.recover(self.worker,self.job,4),2)
        with self.assertRaises(RuntimeError):capacity.recover(self.worker,self.job,1)
if __name__=="__main__":unittest.main()
