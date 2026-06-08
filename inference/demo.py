#!/usr/bin/env python3
# Thesis ref: Section 6.3
# Generates a self-contained interactive HTML demo from DriveLM prediction JSONs.
# All images are base64-encoded, all data is JS-embedded, no external dependencies except Tabler icons CDN.
#
# Usage:
#   python inference/demo.py [--output docs/demo.html] [--predictions-dir inference/outputs] [--max-scenes 50]
#
#   --output: path to write the HTML (default: docs/demo.html)
#   --predictions-dir: root directory containing model subdirs with test_predictions.json (default: inference/outputs)
#   --max-scenes: cap scenes per model to keep file size reasonable (default: 50)
#
# If no real predictions are found, generates a demo using realistic mock data.
# Run this after cloning to immediately view a demo: python inference/demo.py && open docs/demo.html

from __future__ import annotations

import argparse
import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class QAPair:
    question_id: str
    question_type: str  # perception, prediction, planning
    question: str
    ground_truth: str
    prediction: str
    metric: str  # match, accuracy, chatgpt, language, coord
    score: float


@dataclass
class SceneScores:
    accuracy: float
    language: float
    chatgpt: float
    match: float
    coord: float
    final: float


@dataclass
class Scene:
    scene_token: str
    location: str  # singapore, boston
    keyframe_token: str
    cameras: dict[str, str | None]  # camera -> base64 or None
    qa_pairs: list[dict]
    scene_scores: dict


def _encode_image(path: str | Path) -> str | None:
    """Encode image as base64 data URI, or return None if not found."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _make_mock_data() -> dict[str, list[dict]]:
    """Generate realistic mock data for demo when no real predictions exist."""
    mock_scenes = [
        {"token": "scene-0553", "location": "singapore", "keyframe": "abc123def"},
        {"token": "scene-0061", "location": "boston", "keyframe": "xyz789uvw"},
        {"token": "scene-0275", "location": "singapore", "keyframe": "ijk456lmn"},
    ]

    qa_template = [
        {
            "question_id": "q_001",
            "question_type": "perception",
            "question": "What is the most important object ahead of the ego vehicle?",
            "ground_truth": "The pedestrian crossing on the left",
            "metric": "match",
        },
        {
            "question_id": "q_002",
            "question_type": "prediction",
            "question": "What will the vehicle on the right do next?",
            "ground_truth": "Turn right at the intersection",
            "metric": "chatgpt",
        },
        {
            "question_id": "q_003",
            "question_type": "planning",
            "question": "What action should the ego vehicle take?",
            "ground_truth": "Slow down and prepare to yield",
            "metric": "language",
        },
        {
            "question_id": "q_004",
            "question_type": "perception",
            "question": "How many vehicles are visible in the scene?",
            "ground_truth": "Three vehicles",
            "metric": "accuracy",
        },
    ]

    model_scores = {
        "lora_25k_dl_pl_10pct": {
            "accuracy": 0.837,
            "language": 0.676,
            "chatgpt": 0.451,
            "match": 0.304,
            "coord": 0.014,
            "final": 0.589,
        },
        "lora_25k": {
            "accuracy": 0.721,
            "language": 0.591,
            "chatgpt": 0.384,
            "match": 0.268,
            "coord": 0.009,
            "final": 0.560,
        },
        "lora_300k": {
            "accuracy": 0.654,
            "language": 0.512,
            "chatgpt": 0.321,
            "match": 0.213,
            "coord": 0.005,
            "final": 0.493,
        },
    }

    pred_scores = {
        "lora_25k_dl_pl_10pct": [0.91, 0.54, 0.83, 0.21],
        "lora_25k": [0.84, 0.48, 0.76, 0.18],
        "lora_300k": [0.75, 0.42, 0.68, 0.14],
    }

    all_data = {}
    for model_name, scores in pred_scores.items():
        scenes = []
        for mock_scene in mock_scenes:
            qa_pairs = []
            for i, qa_template_item in enumerate(qa_template):
                qa = {
                    "question_id": qa_template_item["question_id"],
                    "question_type": qa_template_item["question_type"],
                    "question": qa_template_item["question"],
                    "ground_truth": qa_template_item["ground_truth"],
                    "prediction": qa_template_item["ground_truth"],  # Mock: perfect-ish
                    "metric": qa_template_item["metric"],
                    "score": scores[i],
                }
                qa_pairs.append(qa)

            scene = {
                "scene_token": mock_scene["token"],
                "location": mock_scene["location"],
                "keyframe_token": mock_scene["keyframe"],
                "cameras": {cam: None for cam in [
                    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
                ]},
                "qa_pairs": qa_pairs,
                "scene_scores": model_scores[model_name],
            }
            scenes.append(scene)
        all_data[model_name] = scenes

    return all_data


def _load_predictions(predictions_dir: Path, max_scenes: int) -> dict[str, list[dict]]:
    """Load predictions from discovered models or fall back to mock data."""
    all_data = {}
    predictions_dir = Path(predictions_dir)

    if not predictions_dir.exists():
        print(f"[INFO] {predictions_dir} not found; using mock data")
        return _make_mock_data()

    models_found = False
    for model_dir in sorted(predictions_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        pred_file = model_dir / "test_predictions.json"
        if not pred_file.exists():
            continue

        models_found = True
        model_name = model_dir.name
        print(f"[INFO] Loading {pred_file}...")

        with open(pred_file, "r") as f:
            data = json.load(f)

        scenes = data.get("predictions", [])[:max_scenes]
        all_data[model_name] = scenes
        print(f"      → {len(scenes)} scenes")

    if not models_found:
        print(f"[INFO] No test_predictions.json found in {predictions_dir}; using mock data")
        return _make_mock_data()

    return all_data


def _generate_html(all_data: dict[str, list[dict]], use_mock: bool) -> str:
    """Generate the complete HTML demo file."""
    data_json = json.dumps(all_data, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VQA-AD Demo — CTU Prague 2026</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.44.0/tabler-icons.min.css">
    <style>
        * {{
            box-sizing: border-box;
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --color-background-primary: #1a1a1a;
                --color-background-secondary: #262626;
                --color-background-tertiary: #0f0f0f;
                --color-text-primary: #e5e5e5;
                --color-text-secondary: #a3a3a3;
                --color-text-tertiary: #808080;
                --color-border-tertiary: #333333;
                --color-border-secondary: #404040;
                --color-text-success: #4ade80;
                --color-text-warning: #facc15;
                --color-text-danger: #f87171;
                --color-background-success: #16a34a;
                --color-background-warning: #eab308;
                --color-background-danger: #dc2626;
            }}
        }}

        @media (prefers-color-scheme: light) {{
            :root {{
                --color-background-primary: #ffffff;
                --color-background-secondary: #f9f9f9;
                --color-background-tertiary: #f3f3f3;
                --color-text-primary: #1a1a1a;
                --color-text-secondary: #666666;
                --color-text-tertiary: #999999;
                --color-border-tertiary: #e5e5e5;
                --color-border-secondary: #d9d9d9;
                --color-text-success: #059669;
                --color-text-warning: #b45309;
                --color-text-danger: #dc2626;
                --color-background-success: #d1fae5;
                --color-background-warning: #fef3c7;
                --color-background-danger: #fee2e2;
            }}
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
            background: var(--color-background-tertiary);
            color: var(--color-text-primary);
            margin: 0;
            padding: 0;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 16px;
        }}

        .mock-warning {{
            background: #fff3cd;
            color: #664d03;
            padding: 12px 16px;
            border-radius: 6px;
            margin-top: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}

        .mock-warning::before {{
            content: "⚠";
            font-size: 16px;
        }}

        /* HEADER */
        .header {{
            padding: 24px 0;
            border-bottom: 1px solid var(--color-border-tertiary);
        }}

        .header h1 {{
            margin: 0 0 4px 0;
            font-size: 18px;
            font-weight: 500;
        }}

        .header .subtitle {{
            color: var(--color-text-secondary);
            font-size: 13px;
            margin: 0 0 12px 0;
        }}

        .header-links {{
            display: flex;
            gap: 12px;
        }}

        .badge {{
            display: inline-block;
            padding: 6px 12px;
            background: var(--color-background-secondary);
            color: var(--color-text-secondary);
            border-radius: 4px;
            text-decoration: none;
            font-size: 12px;
            border: 0.5px solid var(--color-border-tertiary);
            transition: all 0.2s;
        }}

        .badge:hover {{
            background: var(--color-background-primary);
            color: var(--color-text-primary);
        }}

        /* CONTROLS */
        .controls {{
            display: flex;
            gap: 16px;
            margin: 20px 0;
            align-items: center;
        }}

        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        select {{
            padding: 6px 8px;
            border: 0.5px solid var(--color-border-tertiary);
            border-radius: 4px;
            background: var(--color-background-primary);
            color: var(--color-text-primary);
            font-size: 12px;
            cursor: pointer;
        }}

        .location-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 500;
        }}

        .location-badge.singapore {{
            background: #EEEDFE;
            color: #3C3489;
        }}

        .location-badge.boston {{
            background: #E6F1FB;
            color: #0C447C;
        }}

        /* MAIN GRID */
        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1.2fr;
            gap: 16px;
            margin: 20px 0;
            padding: 20px;
            background: var(--color-background-primary);
            border: 0.5px solid var(--color-border-tertiary);
            border-radius: 6px;
        }}

        /* CAMERA GRID */
        .camera-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 4px;
            background: var(--color-background-secondary);
            padding: 8px;
            border-radius: 6px;
        }}

        .camera-cell {{
            position: relative;
            aspect-ratio: 16/9;
            background: var(--color-background-secondary);
            border: 0.5px solid var(--color-border-secondary);
            border-radius: 4px;
            overflow: hidden;
        }}

        .camera-cell img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}

        .camera-cell.placeholder {{
            display: flex;
            align-items: center;
            justify-content: center;
            background: #3a3a3a;
            color: #999;
        }}

        .camera-label {{
            position: absolute;
            bottom: 4px;
            left: 4px;
            background: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 10px;
            z-index: 1;
        }}

        /* QA TABLE */
        .qa-table {{
            display: flex;
            flex-direction: column;
        }}

        .qa-table-header {{
            display: grid;
            grid-template-columns: 24px 50px 1fr 1fr 70px;
            gap: 8px;
            padding: 6px 8px;
            border-bottom: 0.5px solid var(--color-border-secondary);
            font-size: 11px;
            font-weight: 500;
            color: var(--color-text-secondary);
            text-align: left;
        }}

        .qa-table-body {{
            overflow-y: auto;
            max-height: 400px;
        }}

        .qa-row {{
            display: grid;
            grid-template-columns: 24px 50px 1fr 1fr 70px;
            gap: 8px;
            padding: 6px 8px;
            border-bottom: 0.5px solid var(--color-border-tertiary);
            align-items: start;
            vertical-align: top;
        }}

        .qa-row:nth-child(even) {{
            background: var(--color-background-secondary);
        }}

        .qa-row-idx {{
            color: var(--color-text-tertiary);
            font-size: 11px;
        }}

        .type-badge {{
            display: inline-block;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 500;
            white-space: nowrap;
        }}

        .type-badge.perception {{
            background: #E1F5EE;
            color: #085041;
        }}

        .type-badge.prediction {{
            background: #EEEDFE;
            color: #3C3489;
        }}

        .type-badge.planning {{
            background: #FAEEDA;
            color: #633806;
        }}

        .qa-content {{
            display: flex;
            flex-direction: column;
            gap: 2px;
        }}

        .qa-question {{
            font-weight: normal;
            line-height: 1.3;
        }}

        .qa-gt {{
            font-size: 11px;
            color: var(--color-text-secondary);
        }}

        .qa-prediction {{
            line-height: 1.3;
            word-wrap: break-word;
        }}

        .qa-prediction.score-hi {{
            color: var(--color-text-success);
        }}

        .qa-prediction.score-md {{
            color: var(--color-text-warning);
        }}

        .qa-prediction.score-lo {{
            color: var(--color-text-danger);
        }}

        .score-display {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 3px;
        }}

        .score-pill {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}

        .score-pill.s-hi {{
            background: #EAF3DE;
            color: #27500A;
        }}

        .score-pill.s-md {{
            background: #FAEEDA;
            color: #633806;
        }}

        .score-pill.s-lo {{
            background: #FCEBEB;
            color: #791F1F;
        }}

        .score-metric {{
            font-size: 11px;
            color: var(--color-text-secondary);
        }}

        /* SCENE SCORE BAR */
        .scene-score-bar {{
            display: grid;
            grid-template-columns: repeat(5, 1fr) 80px;
            gap: 16px;
            padding: 20px;
            background: var(--color-background-primary);
            border: 0.5px solid var(--color-border-tertiary);
            border-radius: 6px;
            margin-top: 20px;
        }}

        .score-stat {{
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 4px;
        }}

        .score-stat-label {{
            font-size: 10px;
            color: var(--color-text-secondary);
            text-align: center;
        }}

        .score-stat-value {{
            font-size: 15px;
            font-weight: 500;
        }}

        .final-score {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 8px;
            border-radius: 6px;
        }}

        .final-score-label {{
            font-size: 10px;
            font-weight: 500;
            margin-bottom: 4px;
        }}

        .final-score-value {{
            font-size: 22px;
            font-weight: 500;
        }}

        .final-score.s-hi {{
            background: #EAF3DE;
            color: #27500A;
        }}

        .final-score.s-md {{
            background: #FAEEDA;
            color: #633806;
        }}

        .final-score.s-lo {{
            background: #FCEBEB;
            color: #791F1F;
        }}

        @media (max-width: 1000px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {f'<div class="mock-warning">Running with mock data — run inference pipeline to populate real predictions</div>' if use_mock else ''}

        <div class="header">
            <h1>VQA for Autonomous Driving — Demo</h1>
            <div class="subtitle">InternVL2-2B fine-tuned on DriveLM · CTU Prague 2026</div>
            <div class="header-links">
                <a href="https://github.com/dmitrykhursen/VQA-AD-CTU" class="badge" target="_blank">GitHub →</a>
                <a href="https://huggingface.co/dmitrykhursen" class="badge" target="_blank">HuggingFace →</a>
            </div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label style="font-size: 12px;">Model:</label>
                <select id="sel-model">
                    <option value="">Select model...</option>
                </select>
            </div>
            <div class="control-group">
                <label style="font-size: 12px;">Scene:</label>
                <select id="sel-scene">
                    <option value="">Select scene...</option>
                </select>
                <span id="location-badge" class="location-badge" style="display: none;"></span>
            </div>
        </div>

        <div class="main-grid">
            <div>
                <div class="camera-grid" id="camera-grid">
                    <!-- 6 camera cells -->
                </div>
            </div>
            <div class="qa-table">
                <div class="qa-table-header">
                    <div>#</div>
                    <div>Type</div>
                    <div>Question & GT</div>
                    <div>Prediction</div>
                    <div>Score</div>
                </div>
                <div class="qa-table-body" id="qa-table-body">
                    <!-- QA rows -->
                </div>
            </div>
        </div>

        <div class="scene-score-bar" id="scene-score-bar">
            <!-- Scene scores -->
        </div>
    </div>

    <script>
const ALL_DATA = {data_json};

const CAMERA_LAYOUT = [
    ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
    ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
];

const CAMERA_LABELS = {{
    'CAM_FRONT': 'Front',
    'CAM_FRONT_LEFT': 'Front Left',
    'CAM_FRONT_RIGHT': 'Front Right',
    'CAM_BACK': 'Back',
    'CAM_BACK_LEFT': 'Back Left',
    'CAM_BACK_RIGHT': 'Back Right'
}};

function formatScore(score) {{
    return score.toFixed(2);
}}

function getScoreClass(score) {{
    if (score >= 0.8) return 's-hi';
    if (score >= 0.5) return 's-md';
    return 's-lo';
}}

function getPredictionClass(score) {{
    if (score >= 0.8) return 'score-hi';
    if (score >= 0.5) return 'score-md';
    return 'score-lo';
}}

function getTypeColor(qtype) {{
    const types = {{
        'perception': 'perception',
        'prediction': 'prediction',
        'planning': 'planning'
    }};
    return types[qtype] || 'perception';
}}

function renderCameraGrid(scene) {{
    const grid = document.getElementById('camera-grid');
    grid.innerHTML = '';

    for (let row of CAMERA_LAYOUT) {{
        for (let cam of row) {{
            const cell = document.createElement('div');
            cell.className = 'camera-cell';

            const imgData = scene.cameras[cam];
            if (imgData) {{
                const img = document.createElement('img');
                img.src = imgData;
                cell.appendChild(img);
            }} else {{
                cell.classList.add('placeholder');
                const label_text = document.createElement('span');
                label_text.textContent = CAMERA_LABELS[cam];
                cell.appendChild(label_text);
            }}

            const label = document.createElement('div');
            label.className = 'camera-label';
            label.textContent = CAMERA_LABELS[cam];
            cell.appendChild(label);

            grid.appendChild(cell);
        }}
    }}
}}

function renderQATable(scene) {{
    const body = document.getElementById('qa-table-body');
    body.innerHTML = '';

    scene.qa_pairs.forEach((qa, idx) => {{
        const row = document.createElement('div');
        row.className = 'qa-row';

        const idx_cell = document.createElement('div');
        idx_cell.className = 'qa-row-idx';
        idx_cell.textContent = idx + 1;
        row.appendChild(idx_cell);

        const type_cell = document.createElement('div');
        const type_badge = document.createElement('span');
        type_badge.className = 'type-badge ' + getTypeColor(qa.question_type);
        type_badge.textContent = qa.question_type;
        type_cell.appendChild(type_badge);
        row.appendChild(type_cell);

        const qa_cell = document.createElement('div');
        const q_line = document.createElement('div');
        q_line.className = 'qa-question';
        q_line.textContent = qa.question;
        const gt_line = document.createElement('div');
        gt_line.className = 'qa-gt';
        gt_line.innerHTML = `GT: <strong>${{qa.ground_truth}}</strong>`;
        qa_cell.appendChild(q_line);
        qa_cell.appendChild(gt_line);
        row.appendChild(qa_cell);

        const pred_cell = document.createElement('div');
        pred_cell.className = 'qa-prediction ' + getPredictionClass(qa.score);
        pred_cell.textContent = qa.prediction;
        row.appendChild(pred_cell);

        const score_cell = document.createElement('div');
        score_cell.className = 'score-display';
        const score_pill = document.createElement('span');
        score_pill.className = 'score-pill ' + getScoreClass(qa.score);
        score_pill.textContent = formatScore(qa.score);
        const metric_label = document.createElement('div');
        metric_label.className = 'score-metric';
        metric_label.textContent = qa.metric;
        score_cell.appendChild(score_pill);
        score_cell.appendChild(metric_label);
        row.appendChild(score_cell);

        body.appendChild(row);
    }});
}}

function renderScoreBar(scores) {{
    const bar = document.getElementById('scene-score-bar');
    bar.innerHTML = '';

    const metrics = ['accuracy', 'language', 'chatgpt', 'match', 'coord'];
    for (let m of metrics) {{
        const stat = document.createElement('div');
        stat.className = 'score-stat';
        const label = document.createElement('div');
        label.className = 'score-stat-label';
        label.textContent = m;
        const value = document.createElement('div');
        value.className = 'score-stat-value';
        value.textContent = formatScore(scores[m]);
        stat.appendChild(label);
        stat.appendChild(value);
        bar.appendChild(stat);
    }}

    const final = document.createElement('div');
    final.className = 'final-score ' + getScoreClass(scores.final);
    const final_label = document.createElement('div');
    final_label.className = 'final-score-label';
    final_label.textContent = 'scene final score';
    const final_value = document.createElement('div');
    final_value.className = 'final-score-value';
    final_value.textContent = formatScore(scores.final);
    final.appendChild(final_label);
    final.appendChild(final_value);
    bar.appendChild(final);
}}

function updateLocation(location) {{
    const badge = document.getElementById('location-badge');
    if (location) {{
        badge.textContent = location.charAt(0).toUpperCase() + location.slice(1);
        badge.className = 'location-badge ' + location.toLowerCase();
        badge.style.display = 'inline-block';
    }} else {{
        badge.style.display = 'none';
    }}
}}

function updateDisplay() {{
    const modelIdx = document.getElementById('sel-model').value;
    const sceneIdx = document.getElementById('sel-scene').value;

    if (!modelIdx || !sceneIdx) return;

    const models = Object.keys(ALL_DATA);
    const model = models[modelIdx];
    const scene = ALL_DATA[model][sceneIdx];

    renderCameraGrid(scene);
    renderQATable(scene);
    renderScoreBar(scene.scene_scores);
    updateLocation(scene.location);
}}

function populateSelects() {{
    const models = Object.keys(ALL_DATA).sort();
    const model_sel = document.getElementById('sel-model');

    models.forEach((m, idx) => {{
        const opt = document.createElement('option');
        opt.value = idx;
        opt.textContent = m + (m.includes('dl_pl_10') ? ' ⭐ recommended' : '');
        model_sel.appendChild(opt);
    }});

    model_sel.addEventListener('change', () => {{
        const model_idx = model_sel.value;
        const scene_sel = document.getElementById('sel-scene');
        scene_sel.innerHTML = '<option value="">Select scene...</option>';

        if (model_idx !== '') {{
            const model = models[model_idx];
            ALL_DATA[model].forEach((scene, idx) => {{
                const opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = scene.scene_token;
                scene_sel.appendChild(opt);
            }});
        }}
    }});

    document.getElementById('sel-scene').addEventListener('change', updateDisplay);
}}

document.addEventListener('DOMContentLoaded', () => {{
    populateSelects();
}});
    </script>
</body>
</html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate self-contained interactive HTML demo from VQA-AD predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output",
        default="docs/demo.html",
        help="Path to write HTML (default: docs/demo.html)",
    )
    parser.add_argument(
        "--predictions-dir",
        default="inference/outputs",
        help="Root directory containing model subdirs (default: inference/outputs)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=50,
        help="Max scenes per model (default: 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("VQA-AD Demo Generator")
    print("=" * 80)

    all_data = _load_predictions(Path(args.predictions_dir), args.max_scenes)
    total_scenes = sum(len(scenes) for scenes in all_data.values())
    use_mock = total_scenes == 0

    if use_mock:
        print(f"[WARN] No predictions found; generating demo with mock data")
        all_data = _make_mock_data()
        total_scenes = sum(len(scenes) for scenes in all_data.values())

    html = _generate_html(all_data, use_mock)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"[OK] Generated {output_path}")
    print(f"     File size: {file_size_mb:.1f} MB")
    print(f"     Models: {len(all_data)}")
    for model_name, scenes in all_data.items():
        print(f"       - {model_name}: {len(scenes)} scenes")
    print(f"     Total scenes: {total_scenes}")
    print(f"\n[TIP] Open in browser: {output_path.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
