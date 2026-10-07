from pathlib import Path
import json
import numpy as np
import torch
from fid_gpu import extractor, Moments, frechet
from paths import ROOT,RESULTS,VALIDATION_DATA
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
images=np.load(VALIDATION_DATA/'parity_images.npy')
expected=np.load(VALIDATION_DATA/'parity_tf_features.npy')
with torch.inference_mode():
    f=extractor('cuda')
    actual=f(torch.from_numpy(images).permute(0,3,1,2).cuda())[0].cpu().numpy()
error=actual-expected
report=dict(n=len(images),max_abs=float(abs(error).max()),rmse=float(np.sqrt(np.mean(error**2))),
 relative_l2=float(np.linalg.norm(error)/np.linalg.norm(expected)),
 passed=bool(np.allclose(actual,expected,atol=3e-3,rtol=3e-3)),atol=3e-3,rtol=3e-3,
 source='OpenAI ADM frozen GraphDef CPU vs torch-fidelity 0.4.0 CUDA float32, 32 deterministic CIFAR-resolution validation images')
(RESULTS/'fid_parity.json').write_text(json.dumps(report,indent=2))
print(report)
if not report['passed']: raise SystemExit('Feature parity failed: do not combine ADM statistics with these features')
# Streaming moments versus independent NumPy, and GPU FID versus SciPy.
from scipy.linalg import sqrtm
rng=np.random.default_rng(42)
x=rng.normal(size=(512,32)); y=rng.normal(size=(640,32))*.7+.2
m=Moments('cuda',dim=32)
for b in np.array_split(x,7): m.update(torch.tensor(b,device='cuda'))
mu,cov=m.statistics()
np.testing.assert_allclose(mu.cpu(),x.mean(0),atol=1e-12)
np.testing.assert_allclose(cov.cpu(),np.cov(x,rowvar=False),atol=1e-12)
a=np.cov(x,rowvar=False); b=np.cov(y,rowvar=False)
expected_fid=((x.mean(0)-y.mean(0))**2).sum()+np.trace(a+b-2*sqrtm(a@b).real)
actual_fid=frechet(mu,cov,y.mean(0),b)
np.testing.assert_allclose(actual_fid,expected_fid,atol=1e-9)
print('Streaming moments and GPU FID match NumPy/SciPy',actual_fid)
