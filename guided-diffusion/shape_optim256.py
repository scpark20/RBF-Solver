import sys
import os
import numpy as np
import torch
import torch.nn.functional as F
from sample import parse_args_and_config, Diffusion

N = 128
M = 16
K = 20
SCALES = [2.0, 4.0, 6.0, 8.0]
ORDERS = [2, 3]
NFEs = [5, 6, 8, 10, 12, 15, 20]

# Init. Diffusion Model
sys.argv = [
    "sample.py",
    "--config", "imagenet256_guided.yml",  # 사용하려는 config
    "--sample_type", "rbf_solver",
    "--timesteps", "5",
    "--scale", "2.0",
    "--order", "2",
    "--lower_order_final",
    "--shape_dir", "shape_dir/imagenet256/scale2.0"
]

args, config = parse_args_and_config()
diffusion = Diffusion(args, config, rank=0)
diffusion.prepare_model()

for scale in SCALES:
    shape_dir = f"shape_dir/imagenet256/scale{scale}"
    os.makedirs(shape_dir, exist_ok=True)

    # Load target-pairs
    noises = []
    samples = []
    classes = []
    for i in range(100):
        pt_file = f'samples/256x256_diffusion/unipc_200_scale{scale}/images/target_{i}.pt'
        if not os.path.exists(pt_file):
            break
        data = torch.load(pt_file)
        noises.append(data['noise_raw'])
        samples.append(data['sample_raw'])
        classes.append(data['classes'])

    noises = torch.cat(noises, dim=0)
    samples = torch.cat(samples, dim=0)
    classes = torch.cat(classes, dim=0)
    print(noises.shape, samples.shape, classes.shape)

    for order in ORDERS: 
        for NFE in NFEs:
            diffusion.args.scale = scale
            diffusion.args.dpm_solver_order = order
            diffusion.args.timesteps = NFE
            diffusion.args.shape_dir = shape_dir
            
            for number in range(K):
                indexes = np.random.randint(0, len(noises), size=(M,))
                noise_batch = noises[indexes].to(device=diffusion.device)
                sample_batch = samples[indexes].to(device=diffusion.device)
                classes_batch = classes[indexes].to(device=diffusion.device)
                
                with torch.no_grad():
                    sampled_x, _ = diffusion.sample_image(noise_batch, diffusion.model, classifier=diffusion.classifier, classes=classes_batch, target=sample_batch, number=number)
                    print(f"number={number}, NFE={NFE}, order={order}, loss={F.mse_loss(sample_batch, sampled_x)}")

            # Average K shape parameters
            optimal_log_shapes_list = []
            for number in range(K):
                npz_file = os.path.join(shape_dir, f'NFE={NFE},p={order},number={number}.npz')
                if not os.path.exists(npz_file):
                    continue
                data = np.load(npz_file)
                optimal_log_shapes_list.append(data['optimal_log_shapes'])
            optimal_log_shapes = np.stack(optimal_log_shapes_list, axis=0)
            optimal_log_shapes = np.mean(optimal_log_shapes, axis=0)
            save_file = os.path.join(shape_dir, f'NFE={NFE},p={order}.npz')
            np.savez(save_file, optimal_log_shapes=optimal_log_shapes)
