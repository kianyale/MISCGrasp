# MISCGrasp

**MISCGrasp: Leveraging Multiple Integrated Scales and Contrastive Learning for Enhanced Volumetric Grasping** ([arXiv][1])

This repository provides the research codebase for **MISCGrasp**, a volumetric 6-DoF grasp detection framework that integrates multi-scale feature extraction with contrastive feature enhancement for self-adaptive grasping. ([arXiv][1])

> Note: This project has been developed over a long period of time and the repository may still be **roughly organized** in places. If it provides you with useful insights, please consider starring the repo.

## Links

* Project page: [https://miscgrasp.github.io/](https://miscgrasp.github.io/) ([miscgrasp.github.io][2])
* Paper (arXiv:2507.02672): [https://arxiv.org/abs/2507.02672](https://arxiv.org/abs/2507.02672) ([arXiv][1])
* Video: available via the project page ([miscgrasp.github.io][2])

## Method Overview

Robotic grasping must generalize across objects of diverse shapes and scales. MISCGrasp addresses this by:

* Integrating **multi-scale geometric features** to balance fine-grained local graspability and global structure. ([arXiv][1])
* Using **multi-scale contrastive learning** to enhance feature consistency among positive grasp samples across scales. ([arXiv][1])
* Demonstrating performance in both simulation and real-world tabletop decluttering experiments. ([arXiv][1])

## Repository Structure

> The following is a high-level guide to the visible top-level layout of this repo. ([GitHub][3])

* `src/`
  Core implementation (models, training/inference utilities, etc.).
* `scripts/`
  Experiment / evaluation scripts and helpers.
* `data_generator/`
  Utilities for data generation / preprocessing (e.g., for synthetic scenes or dataset preparation).
* `train.sh`
  Shell entrypoint for training.
* `run_single.sh`
  Shell entrypoint for running a single-case inference / demo.

## Environment Setup

This codebase is primarily Python-based. Please prepare a Python environment that supports typical deep learning / 3D processing workflows (e.g., PyTorch + common scientific stack).

A common setup pattern:

1. Create an isolated environment (Conda or venv).
2. Install your PyTorch version compatible with your CUDA/toolchain.
3. Install remaining dependencies required by `src/` and `scripts/`.

> If you plan to make this repo easier for others to run, consider adding a `requirements.txt` or `environment.yml`.

## Quick Start

### Training

Use the provided training entrypoint:

```bash
bash train.sh
```

If you need to modify training parameters (dataset paths, checkpoints, GPUs, etc.), please edit `train.sh` and/or the underlying Python entrypoints invoked inside it.

### Single-run Inference / Demo

Run a single example via:

```bash
bash run_single.sh
```

Similarly, adjust paths and options inside `run_single.sh` as needed.

## Data

The project introduces/grants access to a dataset rich in both **power** and **pinch** grasps (see project page for details). ([miscgrasp.github.io][2])

> If you host dataset files or download scripts separately, consider adding:
>
> * dataset download instructions
> * expected directory layout
> * preprocessing steps and checksums (optional)

## Results and Evaluation

MISCGrasp is evaluated in simulated settings and physical tabletop decluttering experiments (details on the project page). ([miscgrasp.github.io][2])

If you want this README to be more “reproducibility-friendly”, a good next step is to add:

* exact command-lines for each benchmark split
* checkpoint links
* expected metrics and scripts to reproduce tables/figures

## Citation

If you use this work in your research, please cite:

```bibtex
@article{fan2025miscgrasp,
  title   = {MISCGrasp: Leveraging Multiple Integrated Scales and Contrastive Learning for Enhanced Volumetric Grasping},
  author  = {Fan, Qingyu and Cai, Yinghao and Li, Chao and Jiao, Chunting and Zheng, Xudong and Lu, Tao and Liang, Bin and Wang, Shuo},
  journal = {arXiv preprint arXiv:2507.02672},
  year    = {2025},
}
```

([arXiv][1])

## Acknowledgements

Thanks to the authors of **VGN** for making their work publicly available, and to **OrbitGrasp** for sharing helpful screen recording tips (as noted on the project page). ([miscgrasp.github.io][2])

## Contact

For questions, suggestions, or bug reports, please open an Issue on GitHub.
