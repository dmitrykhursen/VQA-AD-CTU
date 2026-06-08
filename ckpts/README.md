# Checkpoints

All checkpoints are LoRA adapters trained on top of [InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B).

## Base Model

| Model | HuggingFace |
|---|---|
| InternVL2-2B | [OpenGVLab/InternVL2-2B](https://huggingface.co/OpenGVLab/InternVL2-2B) |

## Fine-Tuned Checkpoints

| Local dir | Training data | DriveLM Score | HuggingFace |
|---|---|---|---|
| `lora25k/` | Custom 25k split (DriveLM only) | 0.560 | [dkhursen/InternVL2-2b-LoRA-25k-drivelm](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-25k-drivelm) |
| `lora300k/` | Full DriveLM 300k split | — | [dkhursen/InternVL2-2b-LoRA-300k-drivelm](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-300k-drivelm) |
| **`dlpl_aug10pct/`** ⭐ | Custom 25k + 10% pseudo-labels | **0.589** | [dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct) (**recommended**) |
| `offline_lora/` | Custom 25k, offline augmentation | — | [dkhursen/InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag-bkgd](https://huggingface.co/dkhursen/InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag-bkgd) |

Weights are gitignored. Download from HuggingFace and place into the corresponding subdirectory before running inference.

## Loading

See [src/inference/predict_demo.py](../src/inference/predict_demo.py) for a self-contained example.
The inference scripts (`scripts/05_inference.sh`, `scripts/05_inference_all_splits.sh`) load adapters automatically given `--lora_path`.

```python
from src.inference.predict_demo import load_model, predict

model, tokenizer = load_model("OpenGVLab/InternVL2-2B", lora_path="ckpts/dlpl_aug10pct")
answer = predict(model, tokenizer, image_path="image.jpg", question="What is the ego vehicle doing?")
```
