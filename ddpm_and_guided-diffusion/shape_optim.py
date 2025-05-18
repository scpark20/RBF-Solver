import numpy as np
import torch
import torch.nn.functional as F
import random
from main import parse_args_and_config, Diffusion
import sys

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.benchmark = False

pt_file = 'experiments/imagenet64/unipc_order3_200/image_samples/images/target.pt'
data = torch.load(pt_file)
noise = data['noise_raw']
sample = data['sample_raw']

sys.argv = [
            "main.py",
            "--config", "imagenet64.yml",
            "--sample",
            "--dpm_solver_type", "data_prediction",
            "--dpm_solver_order", "3",
            "--skip_type", "logSNR",
            "--ni",
            "--sample_type", "rbf_solver",
            "--timesteps", "5",
            "--shape_dir", "shape_dir",
            "--lower_order_final",
        ]

args, config = parse_args_and_config()
diffusion = Diffusion(args, config, rank=0)
diffusion.prepare_model()
diffusion.model.eval()
device = diffusion.device

for order in [3, 4]:
    for NFE in [5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40]:
        diffusion.args.dpm_solver_order = order
        diffusion.args.timesteps = NFE
        
        noise_batch = noise.to(device)
        sample_batch = sample.to(device)
        pred, _ = diffusion.sample_image(noise_batch, diffusion.model, target=sample_batch)
        loss = F.mse_loss(pred, sample_batch)
        print('NFE :', NFE, 'order :', order, 'loss :', loss)