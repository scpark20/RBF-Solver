"""Final experiment acceptance checks; reports only to the approved result root."""
import json,math,hashlib,os
import numpy as np
from paths import ROOT,WEIGHTS,RESULTS,validate_storage
validate_storage()
rows=[]; baseline=None
for method in ['dpmpp','unipc']:
 for nfe in [5,6,8,10]:
  folder=RESULTS/'full'/f'{method}_nfe{nfe}'
  result=json.loads((folder/'result.json').read_text())
  c=result['config']
  assert c['n']==50000 and c['nfe']==nfe and c['method']==method
  assert result['actual_nfe_per_image']==nfe
  assert result['network_examples_per_image']==2*nfe
  assert math.isfinite(result['fid']) and result['fid']>=0
  common={k:v for k,v in c.items() if k not in ['method','nfe']}
  if baseline is None: baseline=common
  else: assert common==baseline,'Unpaired experiment configuration'
  with np.load(folder/'statistics.npz') as z:
   assert int(z['n'])==50000
   assert z['mu'].shape==(2048,) and z['sigma'].shape==(2048,2048)
   assert np.isfinite(z['mu']).all() and np.isfinite(z['sigma']).all()
   assert np.allclose(z['sigma'],z['sigma'].T,atol=1e-12,rtol=0)
   assert np.diag(z['sigma']).min()>=0
  assert (folder/'preview.png').is_file()
  rows.append(dict(method=method,nfe=nfe,fid=result['fid'],n=50000))
model=WEIGHTS/'SiT-diffusers/SiT-S-2-256/transformer/diffusion_pytorch_model.safetensors'
h=hashlib.sha256()
with model.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''): h.update(b)
assert h.hexdigest()==baseline['checkpoint_sha256']
assert json.loads((RESULTS/'fid_parity.json').read_text())['passed']
for base,dirs,files in os.walk(ROOT):
 dirs[:]=[d for d in dirs if d not in ['.venv','.tfvenv','.git','vendor']]
 for name in files:
  assert not name.endswith(('.pt','.pth','.safetensors','.ckpt','.npz','.npy','.pb')),str(os.path.join(base,name))
report=dict(passed=True,conditions=8,total_images=400000,paired_config=True,weights_sha256=h.hexdigest(),rows=rows,
            code_root=str(ROOT),weight_root=str(WEIGHTS),result_root=str(RESULTS))
(RESULTS/'full/final_validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
