"""Reproduce the pinned model download, strictly outside the code directory."""
import hashlib,json,urllib.request
from pathlib import Path
from paths import ROOT,WEIGHTS,RESULTS,validate_storage
from huggingface_hub import snapshot_download
WEIGHTS.mkdir(parents=True,exist_ok=True); RESULTS.mkdir(parents=True,exist_ok=True)
validate_storage()
revision='2f02393a3079dfa6210c10302cc927f59d2cb142'
snapshot_download('BiliSakura/SiT-diffusers',revision=revision,
    allow_patterns=['SiT-S-2-256/transformer/*.safetensors','SiT-S-2-256/transformer/config.json','SiT-S-2-256/vae/*.safetensors','SiT-S-2-256/vae/config.json'],
    local_dir=WEIGHTS/'SiT-diffusers',cache_dir=WEIGHTS/'hf_cache'/'hub')
model=WEIGHTS/'SiT-diffusers/SiT-S-2-256/transformer/diffusion_pytorch_model.safetensors'
h=hashlib.sha256()
with model.open('rb') as f:
    for b in iter(lambda:f.read(8<<20),b''): h.update(b)
assert h.hexdigest()=='3b0754c57c2b6e2e4e74b181d1730a8bd824a30eafeaae9eaf8bc4015e8e4f39'
graph=WEIGHTS/'classify_image_graph_def.pb'
if not graph.exists():
    tmp=graph.with_suffix('.download')
    urllib.request.urlretrieve('https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/classify_image_graph_def.pb',tmp)
    tmp.replace(graph)
print('Pinned weights ready:',model)
