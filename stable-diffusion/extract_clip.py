import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''Stable Diffusion Init.'''
from txt2img_sample import get_parser, load_model_from_config

parser = get_parser()

# 2) args_list definition
args_list = [
    "--config", "configs/stable-diffusion/v1-inference.yaml",
    "--ckpt", "models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
    "--H", "512",
    "--W", "512",
    "--C", "4",
    "--f", "8",
]

opt = parser.parse_args(args_list)

import torch
from omegaconf import OmegaConf

config = OmegaConf.load(f"{opt.config}")
model = load_model_from_config(config, f"{opt.ckpt}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
model.eval()

''' CLIP Model Init.'''
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

'''Decode and PIL'''
import numpy as np
import os
import torch
from einops import rearrange
from PIL import Image

# 1) decode() 안에서 PIL Image까지 바로 꺼내도록 살짝 손봐두면 편함
def decode_to_pil(image):
    with torch.no_grad():
        x = model.decode_first_stage(image.to(device))
    x = torch.clamp((x + 1.0) / 2.0, 0, 1)          # [0,1]
    x = (x * 255).byte().cpu()                      # uint8
    # BCHW  ->  list[HWC]
    imgs = [Image.fromarray(t.permute(1,2,0).numpy()) for t in x]
    return imgs

import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

def extract_clip(pt_path):
    print(pt_path)
    files = [os.path.join(pt_path, f) for f in os.listdir(pt_path) if '.pt' in f]
    for file in tqdm(files):
        data = torch.load(file)
        samples = torch.stack([preprocess(sample) for sample in decode_to_pil(data['sample_raw'])]).to(device)
        
        with torch.no_grad():
            clip_features = clip_model.encode_image(samples)
        data.update({"clip_features": clip_features.cpu()})
        torch.save(data, file)

''' Run '''
root_dir = 'outputs/'
model_names = ['dpm_solver++', 'uni_pc', 'rbf_solver']
ORDERS = [2, 3]
SCALES = [1.5, 3.5, 5.5, 7.5, 9.5]
NFES = [5, 6, 8, 10, 12, 15, 20, 200]
for order in ORDERS:
    for scale in SCALES:
        for model_name in model_names:
            for nfe in NFES:
                path = os.path.join(root_dir, f"{model_name}_{order}_{nfe}_{scale}")
                if os.path.exists(path):
                    extract_clip(path)