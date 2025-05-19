import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

print(sys.path)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.rbf_solver import RBFSampler

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast"
    )
    
    return parser

def main():
    parser = get_parser()
    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = RBFSampler(model)

    N = 128
    M = 6
    K = 20
    ORDERS = [2, 3]
    #SCALES = [1.5, 3.5, 5.5, 7.5, 9.5]
    SCALES = [7.5]
    NFES = [5, 6, 8, 10, 12, 15, 20]
    for SCALE in SCALES:
        pt_path = f'outputs/uni_pc_200_{SCALE}'
        noise_raws = []
        sample_raws = []
        prompts = []
        for i in range(16):
            pt_file = os.path.join(pt_path, f"{i}.pt")
            if not os.path.exists(pt_file):
                continue
            data = torch.load(pt_file)
            noise_raws.append(data['noise_raw'])
            sample_raws.append(data['sample_raw'])
            prompts += data['prompt']
        # (N, C, H, W)
        noise_raws = torch.cat(noise_raws, dim=0)[:N]
        sample_raws = torch.cat(sample_raws, dim=0)[:N]
        prompts = prompts[:N]
        shape_dir = f'shape_dir/scale{SCALE}'
        for ORDER in ORDERS:
            for NFE in NFES:
                precision_scope = autocast if opt.precision == "autocast" else nullcontext
                with torch.no_grad():
                    with precision_scope("cuda"):
                        with model.ema_scope():
                            for number in range(K):
                                indexes = np.random.randint(0, len(noise_raws), size=(M,))
                                noise_batch = noise_raws[indexes].to(device=device)
                                sample_batch = sample_raws[indexes].to(device=device)
                                prompt_batch = [prompts[i] for i in indexes]
                                uc = model.get_learned_conditioning(len(prompt_batch) * [""])
                                c = model.get_learned_conditioning(prompt_batch)
                                sampler.sample(
                                    S=NFE,
                                    order=ORDER,
                                    conditioning=c,
                                    batch_size=len(noise_batch),
                                    shape=[opt.C, opt.H // opt.f, opt.W // opt.f],
                                    verbose=False,
                                    unconditional_guidance_scale=SCALE,
                                    unconditional_conditioning=uc,
                                    x_T=noise_batch,
                                    x_0=sample_batch,
                                    shape_dir=shape_dir,
                                    number=number
                                )

                            # Average K shape parameters
                            optimal_log_shapes_list = []
                            for number in range(K):
                                npz_file = os.path.join(shape_dir, f'NFE={NFE},p={ORDER},number={number}.npz')
                                if not os.path.exists(npz_file):
                                    continue
                                data = np.load(npz_file)
                                optimal_log_shapes_list.append(data['optimal_log_shapes'])
                            optimal_log_shapes = np.stack(optimal_log_shapes_list, axis=0)
                            optimal_log_shapes = np.mean(optimal_log_shapes, axis=0)
                            save_file = os.path.join(shape_dir, f'NFE={NFE},p={ORDER}.npz')
                            np.savez(save_file, optimal_log_shapes=optimal_log_shapes)

if __name__ == "__main__":
    main()
