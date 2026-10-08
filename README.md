<div align="center">

# RBF-Solver

**A Multistep Sampler for Diffusion Probabilistic Models<br>via Radial Basis Functions**

[![arXiv](https://img.shields.io/badge/arXiv-2603.13330-b31b1b?style=flat-square)](https://arxiv.org/abs/2603.13330)

[Paper](https://arxiv.org/abs/2603.13330) · [Core implementation](rbf_solver.py) · [Experiment pipelines](#experiment-pipelines) · [Citation](#citation)

</div>

RBF-Solver is a multistep sampler that interpolates model evaluations with Gaussian radial basis functions. Its learned shape parameters adapt the predictor and corrector while keeping the generative model fixed.

This repository provides the core solver and the experiment pipelines accompanying the [paper](https://arxiv.org/abs/2603.13330).

## Core implementation

[**`rbf_solver.py`**](rbf_solver.py) contains the RBF-Solver implementation and shape-parameter optimization.

## Experiment pipelines

| Directory | Model and experiment |
|---|---|
| [**`score_sde/`**](score_sde/README.md) | **Score-SDE** · CIFAR-10 |
| [**`edm/`**](edm/README.md) | **Elucidated Diffusion Model (EDM)** · CIFAR-10 |
| [**`ddpm_and_guided-diffusion/`**](ddpm_and_guided-diffusion/README.md) | **Improved-Diffusion (DDPM)** · ImageNet 64×64 |
| [**`guided-diffusion/`**](guided-diffusion/README.md) | **Guided-Diffusion** · ImageNet 128×128 and 256×256 |
| [**`stable-diffusion/`**](stable-diffusion/README.md) | **Stable Diffusion v1.4** · Text-to-image |
| [**`sit/`**](sit/README.md) | **SiT-S/2** · ImageNet 256×256 · Additional flow-matching experiments |
| [**`1rf/`**](1rf/README.md) | **1-Rectified Flow** · CIFAR-10 · Additional flow-matching experiments |

Each original diffusion pipeline includes a README with step-by-step instructions for reproducing the corresponding paper experiments. The SiT and 1-RF directories contain separate additional experiments; their setup, results, and dashboards are documented within those directories.

## Citation

If you use RBF-Solver in your research, please cite:

```bibtex
@article{park2026rbfsolver,
  title   = {{RBF-Solver}: A Multistep Sampler for Diffusion Probabilistic Models via Radial Basis Functions},
  author  = {Park, Soochul and Lee, Yeon Ju and Yoon, SeongJin and Shin, Jiyub and Lee, Juhee and Jo, Seongwoon},
  journal = {arXiv preprint arXiv:2603.13330},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.13330},
  url     = {https://arxiv.org/abs/2603.13330}
}
```
