# MISCGrasp

**MISCGrasp: Leveraging Multiple Integrated Scales and Contrastive Learning for Enhanced Volumetric Grasping**  
Paper: https://arxiv.org/abs/2507.02672  
Project page: https://miscgrasp.github.io/

This repository provides the research codebase for **MISCGrasp**, a volumetric 6-DoF grasp detection framework that integrates multi-scale feature learning with contrastive feature enhancement for improved generalization across object geometries and scales.

> Note: This project has been developed over a long period of time and the repository may still be roughly organized in places. If it provides you with useful insights, please consider starring the repo.

---

## Links

- Project page: https://miscgrasp.github.io/  
- Paper (arXiv:2507.02672): https://arxiv.org/abs/2507.02672  
- Video: available via the project page

---

## Method Overview

Robotic grasping must generalize across objects of diverse shapes and scales. MISCGrasp addresses this by:

- Integrating **multi-scale geometric features** to balance fine-grained local graspability and global structure.
- Using **multi-scale contrastive learning** to enhance feature consistency among positive grasp samples across scales.
- Demonstrating performance in both simulation and real-world tabletop decluttering experiments (see project page/paper for details).

---

## Repository Structure

High-level layout:

- `src/`  
  Core implementation (models, training/inference utilities, etc.).
- `scripts/`  
  Experiment / evaluation scripts and helpers.
- `data_generator/`  
  Utilities for data generation / preprocessing (e.g., synthetic scene generation, dataset preparation).
- `train.sh`  
  Shell entrypoint for training.
- `run_single.sh`  
  Shell entrypoint for single-case inference / demo.

---

## Environment Setup

This codebase is Python-based. Prepare a Python environment suitable for deep learning and 3D processing (e.g., PyTorch + common scientific stack). A typical setup pattern:

1. Create an isolated environment (Conda or venv).
2. Install PyTorch compatible with your CUDA/toolchain.
3. Install remaining dependencies required by `src/` and `scripts/`.

> Tip: If you want to make this repo easier for others to run, consider adding a `requirements.txt` or `environment.yml`.

---

## Quick Start

### Training

Run training via:

```bash
bash train.sh
````

Edit `train.sh` (and the Python entrypoints it calls) to set dataset paths, checkpoints, GPUs, and other options.

### Single-run Inference / Demo

Run a single example via:

```bash
bash run_single.sh
```

Adjust paths/options inside `run_single.sh` as needed.

---

## Data

The project introduces and evaluates a dataset containing both **power** and **pinch** grasps (see the project page for details). If you distribute dataset assets separately, consider documenting:

* download instructions
* expected directory layout
* preprocessing steps (and optional checksums)

---

## Results and Evaluation

MISCGrasp is evaluated in simulated settings and physical tabletop decluttering experiments. For reproducibility, consider adding:

* exact commands for each benchmark/split
* checkpoint links
* scripts to reproduce paper tables/figures

---

## Citation

If you use this work in academic research, please cite:

```bibtex
@INPROCEEDINGS{11246166,
  author={Fan, Qingyu and Cai, Yinghao and Li, Chao and Jiao, Chunting and Zheng, Xudong and Lu, Tao and Liang, Bin and Wang, Shuo},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={MISCGrasp: Leveraging Multiple Integrated Scales and Contrastive Learning for Enhanced Volumetric Grasping}, 
  year={2025},
  volume={},
  number={},
  pages={11335-11342},
  keywords={Shape;Focusing;Grasping;Contrastive learning;Transformers;Feature extraction;Intelligent robots;Faces},
  doi={10.1109/IROS60139.2025.11246166}
}
```

---

## Acknowledgements

* VGN and related volumetric grasping literature/tools.
* Any third-party libraries, datasets, and upstream repos used in this project (please follow their licenses and citation requirements).

---

## Contact

For questions, suggestions, or bug reports, please open an Issue on GitHub.
