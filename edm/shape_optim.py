# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import app
from absl import flags
import logging
import os
import torch
import io
import time
import numpy as np


from torchvision.utils import make_grid, save_image
from samplers.rbf_solver import RBFSolver
from samplers.dpm_solver import DPM_Solver
from samplers.uni_pc import UniPC
from samplers.dpm_solver_v3 import DPM_Solver_v3
from samplers.heun import Heun
from samplers.utils import NoiseScheduleEDM, model_wrapper
import functools
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string("ckp_path", None, "Checkpoint path.")
flags.DEFINE_string("statistics_dir", None, "Statistics path for DPM-Solver-v3.")
flags.DEFINE_string("method", None, "Method: heun/dpm_solver++/uni_pc/dpm_solver_v3")
flags.DEFINE_string("eval_folder", "samples", "The folder name for storing evaluation results")
flags.DEFINE_string("sample_folder", "sample", "The folder name for storing samples")
flags.DEFINE_string("unipc_variant", "bh1", "UniPC variant: bh1/bh2")
flags.DEFINE_integer("steps", default=10, help="Number of sampling steps")
flags.DEFINE_integer("order", default=3, help="Order for sampling")
flags.DEFINE_boolean("denoise_to_zero", default=False, help="Denoise at the last step")
flags.DEFINE_string("skip_type", "logSNR", "The timestep schedule for sampling")
flags.DEFINE_string("shape_dir", "shape_dir", "The dir to save shape parameters")
flags.DEFINE_string("pair_pt", None, "The target pair to optimize the shape parameters")
flags.mark_flags_as_required(["ckp_path", "method"])


def main(argv):
    sample(
        FLAGS.ckp_path,
        FLAGS.method,
        FLAGS.steps,
        FLAGS.order,
        FLAGS.skip_type,
        FLAGS.denoise_to_zero,
        FLAGS.shape_dir,
        FLAGS.pair_pt,
    )


def sample(
    ckp_path,
    method,
    steps,
    order,
    skip_type,
    denoise_to_zero,
    shape_dir,
    pair_pt,
    batch_size=128,
    sigma_min=0.002,
    sigma_max=80,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Fix the seed for z = sde.prior_sampling(shape).to(device) in deterministic sampling
    torch.manual_seed(10)

    # Load network.
    print(f'Loading network from "{ckp_path}"...')
    with open(ckp_path, "rb") as f:
        net = pickle.load(f)["ema"].to(device)

    ns = NoiseScheduleEDM()

    if method == 'rbf_solver':
        rbf_solver = RBFSolver(ns, shape_dir=shape_dir)

        def rbf_solver_sampler(model_fn, z, x):
            with torch.no_grad():
                x = rbf_solver.sample_with_optim(
                    model_fn,
                    z,
                    x,
                    steps=steps - 1 if denoise_to_zero else steps,
                    t_start=sigma_max,
                    t_end=sigma_min,
                    order=order,
                    skip_type=skip_type,
                    lower_order_final=True,
                )
                return x, steps

        sampling_fn = rbf_solver_sampler        
    else:
        assert False, f"Method {method} not supported."

    data = torch.load(pair_pt)
    noise_raw = data['noise_raw'].to(device)
    sample_raw = data['sample_raw'].to(device)
    latents = noise_raw.to(torch.float64) * sigma_max
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim)[torch.randint(net.label_dim, size=[batch_size])].to(device)
    noise_pred_fn = model_wrapper(net, ns, class_labels)
    sampling_fn(noise_pred_fn, latents, sample_raw)

if __name__ == "__main__":
    app.run(main)
