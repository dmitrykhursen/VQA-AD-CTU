# Visual Question Answering for Autonomous Driving (VQA-AD-CTU)

> **Official implementation** of the Master's thesis *Visual Question Answering for Autonomous Driving*
> — Dmytro Khursenko, Czech Technical University in Prague, Faculty of Electrical Engineering, 2026.
> Supervised by Ing. David Hurych, Ph.D. (Valeo) and doc. Georgios Tolias, Ph.D. (CTU FEE).

---

## 📌 Status

> **Code coming soon.**
> This repository is the official codebase for the thesis. The implementation is being prepared for release and will be uploaded progressively. Check back before the thesis defence or watch the repository for updates.

A subset of the codebase involves internal Valeo tooling subject to the company's open-source approval process and will not be included in the public release. The released code will be sufficient to reproduce the core results on the publicly available [DriveLM-nuScenes](https://github.com/OpenDriveLab/DriveLM) dataset.

---

## Overview

This work addresses the data bottleneck limiting Vision-Language Models (VLMs) in autonomous driving. Rather than relying on costly manual annotation, we generate VQA-style pseudo-labels from structured sensor priors — 3D bounding boxes, LiDAR-derived distances, and multi-frame object tracking trajectories — orchestrated by a Large Language Model. The generated data is used to fine-tune a VLM and evaluated on the DriveLM benchmark.

Key findings:
- **Distributional alignment matters more than data volume**: a custom 25k-pair aligned split outperforms the official 300k-pair non-i.i.d. split on DriveLM.
- **Pseudo-label augmentation helps at low mixing ratios** (~10%) but degrades performance at higher ratios without quality filtering.
- **Visual localization is the dominant bottleneck**: an oracle experiment with ground-truth object localizations raises the DriveLM final score from 0.589 to 0.775.

---

## Repository Structure

```
.
├── dataset_analysis/     # DriveLM dataset analysis and custom split construction (Ch. 4)
├── pseudo_labels/        # Pseudo-label generation pipeline (Ch. 5)
│   ├── detection/        # Detection module (GT boxes + YOLO unlabelled mode)
│   ├── spatial/          # LiDAR projection and depth estimation
│   ├── tracking/         # Multi-frame object tracking
│   └── llm_orchestration/# LLM prompt engineering and QA generation
├── finetuning/           # Fine-tuning and evaluation scripts (Ch. 6)
└── README.md
```

> Folders are placeholders — code will be populated before the thesis defence.

---

## Method

### Pseudo-Label Generation Pipeline

Structured sensor priors from the nuScenes / Valeo datasets are extracted and passed to a Large Language Model (Qwen3) which generates natural-language VQA pairs and scene captions grounded in the physical annotations.

| Input | Source |
|---|---|
| 3D bounding boxes | nuScenes GT / YOLO (unlabelled) |
| Per-object metric depth | LiDAR point cloud projection |
| Tracking trajectories | Multi-frame object IDs |
| Ego-vehicle state | nuScenes CAN bus |

### Fine-Tuning

Base model: **InternVL2-2B** with LoRA adaptation.
Evaluated on a custom i.i.d. re-split of DriveLM-nuScenes.

| Model | DriveLM Final Score |
|---|---|
| Pretrained baseline | 0.293 |
| LoRA-25k (aligned split) | 0.560 |
| LoRA-25k + DL-PL 10% | **0.589** |
| Oracle (visual annotation) | 0.775 |

---

## Datasets

- **DriveLM-nuScenes**: [github.com/OpenDriveLab/DriveLM](https://github.com/OpenDriveLab/DriveLM)
- **nuScenes**: [nuscenes.org](https://www.nuscenes.org/)
- **Valeo dataset**: proprietary — not publicly available yet.

---

## Citation

If you find this work useful, please cite:

```bibtex
@mastersthesis{khursenko2026vqa,
  author  = {Khursenko, Dmytro},
  title   = {Visual Question Answering for Autonomous Driving},
  school  = {Czech Technical University in Prague, Faculty of Electrical Engineering},
  year    = {2026},
  supervisor = {Hurych, David and Tolias, Georgios}
}
```

---
