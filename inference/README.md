# Inference & Demo

This document covers running model inference on the DriveLM test split and generating interactive HTML demo pages.

---

## Contents

- [Setup](#setup)
- [Running Inference](#running-inference)
- [Generating a Scene Demo](#generating-a-scene-demo)
- [Regenerating the Full Gallery](#regenerating-the-full-gallery)
- [CLI Reference](#cli-reference)

---

## Setup

```bash
source vqa-ad-ctu-env/bin/activate
```

Inference outputs are stored under `inference/outputs/<model-name>/local_test.json`.
Each file is a flat JSON list of:
```json
{ "id": "<scene>_<frame>_<qa_idx>", "question": "...", "ground_truth": "...", "prediction": "..." }
```

---

## Running Inference

```bash
sbatch scripts/05_inference.sh
```

Or all three splits in parallel (val + test + orig DriveLM test):
```bash
sbatch scripts/05_inference_all_splits.sh
```

---

## Generating a Scene Demo

`scene_demo.py` produces a self-contained HTML page for one scene/frame with:
- Stitched 6-camera collage (embedded as base64)
- All QA pairs with category labels and coordinate token highlighting
- One row per model so long answers never overflow

```bash
python inference/scene_demo.py \
    --scene-frame f9e460f092c94466b1211704b5a8859d_33e36dbd62594a10b783b710350b100f \
    --output docs/my_scene.html
```

Compare a specific subset of models:
```bash
python inference/scene_demo.py \
    --scene-frame f9e460f092c94466b1211704b5a8859d_33e36dbd62594a10b783b710350b100f \
    --models dkhursen__InternVL2-2b-LoRA-25k_plus_DL-PL-10pct OpenGVLab__InternVL2-2B \
    --output docs/my_scene.html
```

By default all models found in `inference/outputs/` are included.

---

## Regenerating the Full Gallery

The `docs/` folder ships with 25 pre-generated demos (one frame per scene, 25 distinct scenes).
To regenerate them:

```bash
python3 - <<'EOF'
import json, random, subprocess, sys
from collections import defaultdict
from pathlib import Path

with open("inference/outputs/dkhursen__InternVL2-2b-LoRA-25k_plus_DL-PL-10pct/local_test.json") as f:
    data = json.load(f)

scene_to_frames = defaultdict(set)
for e in data:
    sf = e["id"].rsplit("_", 1)[0]
    scene_to_frames[sf.split("_", 1)[0]].add(sf)

random.seed(42)
sampled_scenes = random.sample(sorted(scene_to_frames), 25)
sampled = [random.choice(sorted(scene_to_frames[s])) for s in sampled_scenes]

for sf in sampled:
    subprocess.run([sys.executable, "inference/scene_demo.py", "--scene-frame", sf], check=True)
EOF

python inference/make_demo_index.py
```

Then rebuild the gallery index:
```bash
python inference/make_demo_index.py
```

---

## CLI Reference

### `scene_demo.py`

| Argument | Default | Description |
|---|---|---|
| `--scene-frame` | required | `{scene_token}_{frame_token}` |
| `--models` | all in outputs-dir | Space-separated model directory names |
| `--outputs-dir` | `inference/outputs` | Directory containing per-model output folders |
| `--stitched-dir` | cluster path | Directory of pre-stitched 6-camera JPEGs |
| `--test-jsonl` | `data/drivelm_custom_split/test.jsonl` | Source of category labels |
| `--output` | `docs/{scene_frame}_demo.html` | Output HTML path |

### `make_demo_index.py`

| Argument | Default | Description |
|---|---|---|
| `--demo-dir` | `docs` | Directory containing `*_demo.html` files |
| `--stitched-dir` | cluster path | Directory of pre-stitched JPEGs for thumbnails |
| `--output` | `docs/index.html` | Output index HTML path |

---

## Available Models

| Directory | Description |
|---|---|
| `OpenGVLab__InternVL2-2B` | Pretrained baseline |
| `OpenGVLab__Mini-InternVL2-2B-DA-DriveLM` | DriveLM specialist baseline |
| `llava-hf__llava-v1.6-mistral-7b-hf` | LLaVA-1.6 baseline |
| `dkhursen__InternVL2-2b-LoRA-25k-drivelm` | LoRA fine-tuned, 25k aligned split |
| `dkhursen__InternVL2-2b-LoRA-25k_plus_DL-PL-10pct` | LoRA fine-tuned, 25k + 10% pseudo-labels (**best**) |
| `dkhursen__InternVL2-2b-LoRA-300k-drivelm` | LoRA fine-tuned, 300k official split |
| `dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag-bkgd` | Oracle visual annotation variant |
