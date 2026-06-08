---
license: cc-by-4.0
task_categories:
  - visual-question-answering
language:
  - en
tags:
  - autonomous-driving
  - nuscenes
  - drivelm
  - pseudo-labels
  - vqa
---

# DriveLM Pseudo-Labels

Part of the Master's thesis:

> **Visual Question Answering for Autonomous Driving**
> Dmytro Khursenko, Czech Technical University in Prague, Faculty of Electrical Engineering, 2026.
> Supervised by Ing. David Hurych, Ph.D. (Valeo) and doc. Georgios Tolias, Ph.D. (CTU FEE).

**[GitHub](https://github.com/dmitrykhursen/VQA-AD-CTU) · [Demo](https://dmitrykhursen.github.io/VQA-AD-CTU/)**

> 📄 Thesis PDF expected by end of June 2026, following successful defense (CTU FEE).

---

~100k VQA pairs generated from nuScenes sensor priors (Boston + Singapore) using Qwen3 with chain-of-thought reasoning, following the DriveLM question-type distribution. Used to augment the custom 25k i.i.d. DriveLM training split in thesis fine-tuning experiments.

**Best result:** mixing 10% of pseudo-labels with the 25k aligned split improved DriveLM Final Score from 0.560 → **0.589** (higher ratios degrade without quality filtering).

---

## Dataset contents

| File | Records | Description |
|---|---:|---|
| `drivelm_pseudo_qas_100k.json` | ~98,700 | Full pseudo-label corpus |
| `train_aug10pct.json` | 35,694 | 25k DriveLM + 10% pseudo-labels — **best mix** |
| `train_aug30pct.json` | 55,431 | 25k DriveLM + 30% pseudo-labels |
| `train_aug50pct.json` | 75,168 | 25k DriveLM + 50% pseudo-labels |
| `train_aug100pct.json` | 124,511 | 25k DriveLM + 100% pseudo-labels |

Each record follows the DriveLM/LLaMA conversation format:

```json
{
  "id": "<scene_token>_<frame_token>_<qa_idx>",
  "image": ["CAM_FRONT.jpg", "CAM_FRONT_LEFT.jpg", ...],
  "conversations": [
    {"from": "human", "value": "<image>\nQuestion text"},
    {"from": "gpt",   "value": "Answer text"}
  ],
  "category": "perception|prediction|planning|behavior",
  "metric_type": "match|chatgpt|accuracy|language"
}
```

---

## Generation pipeline

Sensor priors extracted from nuScenes are passed to **Qwen3** (with extended chain-of-thought thinking) to generate natural-language VQA pairs that follow the DriveLM question-type distribution:

| Input | Source |
|---|---|
| 2D / 3D bounding boxes | nuScenes ground-truth |
| Per-object metric depth | LiDAR point cloud projection |
| Tracking trajectories | Multi-frame object IDs |
| Ego-vehicle state | nuScenes CAN bus |

Question types covered: perception (object class, moving status), prediction (trajectory, yes/no), planning (ego actions, collision/safety), behavior (ego driving state).

Full pipeline code: [github.com/dmitrykhursen/VQA-AD-CTU](https://github.com/dmitrykhursen/VQA-AD-CTU) — Stage 2–3 scripts.

---

## Augmentation results

| Training data | Final | Acc | ChatGPT | Lang | Match | Coord |
|---|---:|---:|---:|---:|---:|---:|
| LoRA-25k (no augmentation) | 0.560 | 0.826 | 0.589 | 0.459 | 0.338 | 0.015 |
| **LoRA-25k + DL-PL 10%** | **0.589** | 0.836 | 0.676 | 0.451 | 0.304 | 0.013 |
| LoRA-25k + DL-PL 30% | 0.548 | 0.832 | 0.605 | 0.434 | 0.264 | 0.008 |
| LoRA-25k + DL-PL 50% | 0.532 | 0.832 | 0.584 | 0.433 | 0.230 | 0.007 |
| LoRA-25k + DL-PL 100% | 0.511 | 0.805 | 0.544 | 0.430 | 0.232 | 0.007 |

All evaluated on the custom i.i.d. DriveLM-nuScenes test split (3,340 QA pairs). Full results: [evaluation/README.md](https://github.com/dmitrykhursen/VQA-AD-CTU/blob/main/evaluation/README.md).

**Score definitions** (all table values normalised to [0, 1]): **Accuracy** — exact-match on MCQ/Yes-No (strict format); **Language** — mean of BLEU-1–4, ROUGE-L, CIDEr/10; **ChatGPT** — GPT-3.5-turbo semantic score (0–100, ÷100 in Final); **Match** — `(F1_coord × 100 + GPT_match) / 2` (0–100, ÷100 in Final); **Coord** — pure coordinate F1 at L1 < 16 px (diagnostic, not in Final); **Final** — `0.4 × (GPT/100) + 0.2 × Language + 0.2 × (Match/100) + 0.2 × Accuracy`

---

## Fine-tuned models

| Model | HF repo | Final |
|---|---|---:|
| LoRA-25k | [dkhursen/InternVL2-2b-LoRA-25k-drivelm](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-25k-drivelm) | 0.560 |
| LoRA-300k | [dkhursen/InternVL2-2b-LoRA-300k-drivelm](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-300k-drivelm) | 0.493 |
| **LoRA-25k + DL-PL 10% ⭐** | [dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct) | **0.589** |
| LoRA-25k + Oracle annotation | [dkhursen/InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag-bkgd](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag-bkgd) | 0.775 |

---

## License

CC BY 4.0. The pseudo-labels are derived from nuScenes annotations; nuScenes is available under the [nuScenes terms of use](https://www.nuscenes.org/terms-of-use).

---

## Citation

```bibtex
@mastersthesis{khursenko2026vqa,
  author     = {Khursenko, Dmytro},
  title      = {Visual Question Answering for Autonomous Driving},
  school     = {Czech Technical University in Prague, Faculty of Electrical Engineering},
  year       = {2026},
  supervisor = {Hurych, David and Tolias, Georgios}
}
```

```bibtex
@inproceedings{caesar2020nuscenes,
  title     = {nuScenes: A Multimodal Dataset for Autonomous Driving},
  author    = {Caesar, Holger and Bankiti, Varun and Lang, Alex H and others},
  booktitle = {CVPR},
  year      = {2020}
}
```
