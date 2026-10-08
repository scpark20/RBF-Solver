<div align="center">

# RBF-Solver

**A Multistep Sampler for Diffusion Probabilistic Models<br>via Radial Basis Functions**

[![Paper](https://img.shields.io/badge/arXiv-2603.13330-b31b1b?style=flat-square)](https://arxiv.org/abs/2603.13330) [![Flow matching](https://img.shields.io/badge/Flow_matching-SiT_%C2%B7_1--RF-5264C8?style=flat-square)](#flow-matching-experiments) [![Dashboard guide](https://img.shields.io/badge/Dashboard-Visual_guide-168B83?style=flat-square)](DASHBOARD_GUIDE.md)

[Paper](https://arxiv.org/abs/2603.13330) · [SiT](README.md) · [1-RF](../1rf/README.md) · [Dashboard](DASHBOARD_GUIDE.md) · [Reproduction](HANDOFF.md) · [Results](handoff/results.csv)

</div>

RBF-Solver uses Gaussian radial basis functions to interpolate model evaluations in a multistep sampler. Learned shape parameters adapt the predictor and corrector while the generative model remains fixed. See the [paper](https://arxiv.org/abs/2603.13330) for the formulation, analysis, and diffusion benchmarks.

This repository contains the original diffusion experiment pipelines and additional **flow-matching experiments with SiT-S/2 and non-distilled 1-Rectified Flow**, comparing **RBF-Solver, DPM-Solver++, and UniPC**.

<p align="center">
  <a href="DASHBOARD_GUIDE.md">
    <img src="docs/dashboard/screenshots/01-sit-comparison.png" alt="SiT comparison dashboard with CFG selection, FID curves, and DPM-Solver++, UniPC, and RBF results at eleven NFE values" width="1100">
  </a>
  <br>
  <sub>SiT comparison dashboard · actual completed records · select the image for the illustrated guide.</sub>
</p>

## Start here

| Goal | Entry point |
|---|---|
| Understand the method | [Paper](https://arxiv.org/abs/2603.13330) · [Core solver](../rbf_solver.py) |
| Explore SiT-S/2 on ImageNet 256×256 | [SiT experiments](README.md) |
| Explore non-distilled 1-RF on CIFAR-10 | [1-RF experiments](../1rf/README.md) |
| Read the dashboard and learned coefficients | [Illustrated dashboard guide — 11 screenshots](DASHBOARD_GUIDE.md) |
| Set up another machine or continue an experiment | [Environment, assets, validation, and execution](HANDOFF.md) |
| Inspect the published FM results | [CSV](handoff/results.csv) · [JSON](handoff/results.json) |

The FM experiment, dashboard, and handoff guides are written in Korean (한국어).

## Flow-matching experiments

The `sit/` and `1rf/` directories extend the sampler comparison to flow-matching models. These are **additional experiments**; their results are separate from the diffusion benchmarks reported in the paper.

### Evaluation coverage

Published snapshot: **8 October 2026**. Each configuration is one model/conditioning × sampler × NFE combination.

| Model & dataset | Conditioning | NFE | Images / configuration | Completed configurations |
|---|---|---|---:|---:|
| **SiT-S/2 · ImageNet 256×256** | CFG 1.5, 3.5, 5.5, 7.5 | 5, 6, 8, 10 | 10,000 | 48 |
| **SiT-S/2 · ImageNet 256×256** | CFG 0.0 · null-label branch | 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40 | 10,000 | 33 |
| **1-Rectified Flow · CIFAR-10** | Unconditional | 5, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40 | 50,000 | 33 |

All three samplers are included in each row. SiT uses second-order sampling with UniPC BH2; 1-RF uses third-order sampling with UniPC BH1. Full numerical settings and their sources are documented in the [handoff guide](HANDOFF.md#8-수치-조건과-코드-대응).

SiT CFG 0.0 evaluates the null-label branch of the class-conditional checkpoint; it is not a separately trained unconditional-only checkpoint. The SiT weight source and conversion provenance are recorded in the [model documentation](HANDOFF.md#9-자산-출처와-결과-위치).

The counts describe the currently adopted results. The requested SiT CFG 0.0 / RBF / NFE 6 retraining is retained with its previous result and does not add another unique configuration. See the [retraining comparison](DASHBOARD_GUIDE.md#nfe-6의-새-target-재학습) for the measured difference.

### From targets to evaluation

| Stage | Purpose | Saved evidence |
|---|---|---|
| **01 · Target sampling** | Produce matched noise–target pairs with a teacher sampler | Inputs, targets, settings, and hashes |
| **02 · Shape optimization** | Optimize RBF shape parameters using the existing solver optimization loop | Per-run parameters, aggregate parameters, and coefficient analysis |
| **03 · RBF sampling & FID** | Generate evaluation samples with the learned parameters | Preview grids, feature statistics, FID, progress, and result records |

The implementation includes GPU FID evaluation, saved progress for resuming compatible runs, and shared comparison inputs. Model weights and experimental artifacts are stored outside the code directories. Image previews are a subset of the evaluation samples; they are not the complete FID evaluation set.

## Inspect the experiments

The dashboard brings the three samplers into one comparison view and keeps target generation, parameter optimization, and RBF evaluation visible as distinct stages.

| View | What it shows |
|---|---|
| **Comparison** | CFG/NFE selection, FID curves, per-NFE tables, Δ RBF, matched previews, and detailed records |
| **RBF pipeline** | Target, optimization, and sampling progress; learned log γ; coefficient magnitude ratios; signed integration coefficients |
| **GPU status** | Device observations, worker records, and recorded batch policy |
| **Experiment settings** | Model, numerical protocol, FID compatibility checks, and provenance |

**[Read the illustrated dashboard guide →](DASHBOARD_GUIDE.md)**

The dashboard monitors and inspects recorded experiments. It does not start, stop, or reconfigure computation. The image above is a saved snapshot; instructions for starting the observation service are in the [handoff guide](HANDOFF.md).

## Reproduce or continue

Clone the repository together with the pinned SiT and RectifiedFlow source submodules:

```bash
git clone --recurse-submodules https://github.com/scpark20/RBF-Solver.git
cd RBF-Solver
```

| Track | Next step |
|---|---|
| **Paper diffusion benchmarks** | Choose a pipeline in the repository map below and follow its README for model-specific setup and sampling. |
| **SiT / 1-RF additional experiments** | Follow the [handoff guide](HANDOFF.md) for environment setup, external asset restoration, validation, and stage-specific execution. |

For FM experiments, the repository includes a [Dockerfile](handoff/Dockerfile), [Compose configuration](handoff/compose.yaml), [dependency lock](handoff/requirements.lock), and [asset manifest with SHA-256](handoff/assets.json).

Weights, target pairs, learned parameter files, and complete runtime records are external assets. **Cloning the code alone does not restore a completed experiment.** The handoff guide explains how to restore and verify those assets on the destination machine; host paths and GPU capacity must be configured for that environment.

## Repository map

### Core and paper experiment pipelines

| Path | Contents |
|---|---|
| [`rbf_solver.py`](../rbf_solver.py) | Core RBF-Solver implementation and shape-parameter optimization |
| [`score_sde/`](../score_sde/README.md) | CIFAR-10 · Score-SDE experiment pipeline |
| [`edm/`](../edm/README.md) | CIFAR-10 · Elucidated Diffusion Model experiment pipeline |
| [`ddpm_and_guided-diffusion/`](../ddpm_and_guided-diffusion/README.md) | ImageNet 64×64 · Improved-Diffusion (DDPM) experiment pipeline |
| [`guided-diffusion/`](../guided-diffusion/README.md) | ImageNet 128×128 and 256×256 · Guided-Diffusion experiment pipelines |
| [`stable-diffusion/`](../stable-diffusion/README.md) | Stable Diffusion v1.4 · text-to-image experiment pipeline |

Each paper experiment directory retains its own step-by-step reproduction instructions.

### Flow-matching additions and documentation

| Path | Contents |
|---|---|
| [`sit/`](README.md) | SiT-S/2 adapter, target/optimization/sampling stages, GPU FID, and dashboard |
| [`1rf/`](../1rf/README.md) | Non-distilled 1-RF adapter and CIFAR-10 comparison using the same experiment structure |
| [`sit/DASHBOARD_GUIDE.md`](DASHBOARD_GUIDE.md) | Screen-by-screen usage, metric interpretation, and actual screenshots |
| [`sit/HANDOFF.md`](HANDOFF.md) | Portable setup, asset restoration, numerical provenance, and execution boundaries |
| [`sit/handoff/`](handoff/) | Environment files, asset inventory, and published result summaries |

## Citation

If you use RBF-Solver in your research, please cite the [paper](https://arxiv.org/abs/2603.13330):

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
