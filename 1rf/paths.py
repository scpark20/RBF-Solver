"""Approved locations: code in 1rf, model data and outputs in /data/RBF-Solver."""
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor/python'))
PYTHON=Path(sys.executable)
DATA=Path('/data/RBF-Solver')
WEIGHTS=DATA/'weights'
RESULTS=DATA/'results/1rf'
VALIDATION_DATA=RESULTS/'validation_data'
os.environ['HF_HOME']=str(WEIGHTS/'hf_cache')
os.environ['TORCH_HOME']=str(WEIGHTS/'torch')
os.environ['XDG_CACHE_HOME']=str(ROOT/'.cache')
os.environ['CUDA_CACHE_PATH']=str(ROOT/'.cache/cuda')
os.environ['TMPDIR']=str(ROOT/'.cache/tmp')
for p in (ROOT/'.cache/cuda',ROOT/'.cache/tmp'):
    p.mkdir(parents=True,exist_ok=True)
def validate_storage():
    for p in (ROOT/'weights',ROOT/'results',ROOT/'.cache/hf'):
        if p.exists() or p.is_symlink():
            raise RuntimeError(f'Forbidden model/output storage inside code: {p}')
    if not WEIGHTS.is_dir() or not RESULTS.is_dir():
        raise RuntimeError('Approved weight and result directories must already exist')
