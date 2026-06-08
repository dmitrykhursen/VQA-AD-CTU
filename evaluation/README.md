# Evaluation Results

Full per-metric results on the **local test split** of DriveLM-nuScenes.

## Setup

The evaluation follows the [DriveLM evaluation protocol](https://github.com/OpenDriveLab/DriveLM/tree/main/challenge).

**1. Java** — required by the `language-evaluation` package (METEOR scorer spawns a JVM on import).

```bash
# HPC (load module):
ml Java          # or: ml Java/11.0.2

# Ubuntu/Debian:
sudo apt-get install default-jre

# Also required:
sudo apt install libxml-parser-perl
```

**2. Install language-evaluation:**

```bash
pip install git+https://github.com/bckim92/language-evaluation.git
python -c "import language_evaluation; language_evaluation.download('coco')"
```

**3. OpenAI API key** — required for the ChatGPT metric:

```bash
export OPENAI_API_KEY="sk-..."
```

## Running evaluation

```bash
bash scripts/06_evaluate.sh <predictions.json> data/drivelm_custom_split/test.jsonl
```

To print the pre-computed results table:
```bash
python evaluation/print_results.py
```

**Score definitions:**
- **Final** — DriveLM weighted composite score
- **Acc** — exact-match accuracy
- **ChatGPT** — GPT-4 semantic similarity (normalised to [0, 1])
- **Lang** — composite language score: average of B1–B4, ROUGE-L, CIDEr
- **B1–B4** — BLEU-1 to BLEU-4
- **RL** — ROUGE-L
- **CIDEr** — CIDEr
- **Match** — (ChatGPT + Coord) / 2
- **Coord** — bounding-box coordinate accuracy (not included in Final score)

---

## Pretrained baselines

| Model | Final | Acc | ChatGPT | Lang | B1 | B2 | B3 | B4 | RL | CIDEr | Match | Coord |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| InternVL2-2B | 0.293 | 0.000 | 0.573 | 0.080 | 0.156 | 0.068 | 0.028 | 0.013 | 0.173 | 0.007 | 0.241 | 0.004 |
| LLaVA-1.6 | 0.280 | 0.003 | 0.552 | 0.059 | 0.127 | 0.049 | 0.019 | 0.006 | 0.152 | 0.000 | 0.224 | 0.000 |
| LLaMA-Adapter-v2 | 0.267 | 0.000 | 0.533 | 0.093 | 0.227 | 0.070 | 0.029 | 0.014 | 0.192 | 0.028 | 0.176 | 0.000 |

---

## Fine-tuned (base: InternVL2-2B + LoRA)

| Model | Final | Acc | ChatGPT | Lang | B1 | B2 | B3 | B4 | RL | CIDEr | Match | Coord |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mini-DA† | 0.606 | 0.898 | 0.668 | 0.416 | 0.596 | 0.564 | 0.533 | 0.503 | 0.651 | 0.470 | 0.381 | 0.000 |
| LoRA-25k | 0.560 | 0.826 | 0.589 | 0.459 | 0.732 | 0.668 | 0.606 | 0.547 | 0.714 | 0.230 | 0.338 | 0.015 |
| **LoRA-25k + DL-PL 10%** | **0.589** | 0.836 | 0.676 | 0.451 | 0.719 | 0.654 | 0.589 | 0.525 | 0.710 | 0.222 | 0.304 | 0.013 |
| LoRA-25k + DL-PL 30% | 0.548 | 0.832 | 0.605 | 0.434 | 0.695 | 0.625 | 0.554 | 0.483 | 0.692 | 0.201 | 0.264 | 0.008 |
| LoRA-25k + DL-PL 50% | 0.532 | 0.832 | 0.584 | 0.433 | 0.691 | 0.622 | 0.551 | 0.481 | 0.699 | 0.171 | 0.230 | 0.007 |
| LoRA-25k + DL-PL 100% | 0.511 | 0.805 | 0.544 | 0.430 | 0.687 | 0.616 | 0.543 | 0.470 | 0.695 | 0.165 | 0.232 | 0.007 |
| LoRA-25k + Valeo | 0.376 | 0.497 | 0.512 | 0.249 | 0.358 | 0.296 | 0.241 | 0.197 | 0.404 | 0.005 | 0.133 | 0.000 |
| LoRA-300k | 0.493 | 0.339 | 0.706 | 0.412 | 0.607 | 0.552 | 0.501 | 0.452 | 0.676 | 0.323 | 0.303 | 0.006 |

† Mini-DA = OpenGVLab/Mini-InternVL2-2B-DA-DriveLM, fine-tuned by OpenGVLab on DriveLM.

---

## Oracle — offline visual annotation

All rows use **InternVL2-2B-LoRA-25k** with pre-rendered visual annotations overlaid on the images at test time (no model change, annotation only).

| Visual Annotation | Final | Acc | ChatGPT | Lang | B1 | B2 | B3 | B4 | RL | CIDEr | Match | Coord |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Red BBox + CTags + Bkgd | 0.772 | 0.848 | 0.726 | 0.827 | 0.914 | 0.885 | 0.856 | 0.826 | 0.895 | 5.884 | 0.778 | 0.740 |
| Red BBox + CTags | 0.655 | 0.832 | 0.684 | 0.589 | 0.748 | 0.699 | 0.648 | 0.598 | 0.743 | 1.002 | 0.568 | 0.393 |
| Red BBox | 0.634 | 0.743 | 0.740 | 0.741 | 0.827 | 0.765 | 0.711 | 0.659 | 0.775 | 0.707 | 0.419 | 0.112 |
| **Red Circle + CTags + Bkgd** | **0.775** | 0.853 | 0.725 | 0.779 | 0.908 | 0.879 | 0.849 | 0.819 | 0.891 | 5.817 | **0.793** | **0.758** |
| Red Circle + CTags | 0.624 | 0.727 | 0.728 | 0.695 | 0.814 | 0.748 | 0.689 | 0.633 | 0.752 | 0.532 | 0.430 | 0.141 |
| Red Circle | 0.646 | 0.793 | 0.729 | 0.761 | 0.853 | 0.787 | 0.730 | 0.675 | 0.778 | 0.741 | 0.443 | 0.158 |
| Red Midpoint + CTags + Bkgd | 0.771 | 0.858 | 0.716 | 0.826 | 0.910 | 0.881 | 0.851 | 0.821 | 0.895 | 5.962 | 0.780 | 0.736 |
| Red Midpoint + CTags | 0.757 | 0.862 | 0.724 | 0.785 | 0.885 | 0.852 | 0.819 | 0.786 | 0.875 | 4.949 | 0.742 | 0.677 |
| Red Midpoint | 0.653 | 0.799 | 0.735 | 0.746 | 0.837 | 0.773 | 0.718 | 0.665 | 0.774 | 0.707 | 0.463 | 0.199 |

**CTags** = class-name text labels overlaid on objects. **Bkgd** = background dimming to highlight annotated objects.
