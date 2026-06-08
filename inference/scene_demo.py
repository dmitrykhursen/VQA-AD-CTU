"""Generate a self-contained HTML demo for a single scene/frame.

Usage:
    python inference/scene_demo.py \
        --scene-frame f9e460f092c94466b1211704b5a8859d_33e36dbd62594a10b783b710350b100f \
        [--models MODEL_DIR1 MODEL_DIR2 ...] \
        [--outputs-dir inference/outputs] \
        [--stitched-dir /path/to/stitched] \
        [--test-jsonl data/drivelm_custom_split/test.jsonl] \
        [--output data/demo/{scene_frame}_demo.html]
"""

import argparse
import base64
import json
import re
import sys
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "inference" / "outputs"
DEFAULT_STITCHED_DIR = Path("/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched")
DEFAULT_TEST_JSONL = REPO_ROOT / "data" / "drivelm_custom_split" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs"

COORD_TOKEN_RE = re.compile(r"<(c\d+,[A-Z_]+,[\d.]+,[\d.]+)>")

CATEGORY_COLOR = {
    "perception": "#3b82f6",
    "prediction": "#f59e0b",
    "planning":   "#22c55e",
}


def short_name(model_dir: str) -> str:
    if "__" in model_dir:
        return model_dir.split("__", 1)[1]
    return model_dir


def load_categories(test_jsonl: Path) -> dict[str, str]:
    """Map full QA id -> category string."""
    cats: dict[str, str] = {}
    if not test_jsonl.exists():
        return cats
    with open(test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            cats[rec["id"]] = rec.get("category", "")
    return cats


def load_predictions(outputs_dir: Path, model_dirs: list[str], scene_frame: str) -> dict[str, list[dict]]:
    prefix = scene_frame + "_"
    result: dict[str, list[dict]] = {}
    for mdir in model_dirs:
        json_path = outputs_dir / mdir / "local_test.json"
        if not json_path.exists():
            print(f"[warn] {json_path} not found, skipping", file=sys.stderr)
            continue
        with open(json_path) as f:
            data = json.load(f)
        entries = [e for e in data if e["id"].startswith(prefix)]
        entries.sort(key=lambda e: int(e["id"].rsplit("_", 1)[-1]))
        result[mdir] = entries
    return result


def align_rows(predictions: dict[str, list[dict]], categories: dict[str, str]) -> list[dict]:
    """Return list of dicts: {idx, question, ground_truth, category, preds: {model: str}}."""
    if not predictions:
        return []
    canonical_model = max(predictions, key=lambda m: len(predictions[m]))
    rows = []
    for entry in predictions[canonical_model]:
        idx = int(entry["id"].rsplit("_", 1)[-1])
        row = {
            "idx": idx,
            "id": entry["id"],
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
            "category": categories.get(entry["id"], ""),
            "preds": {},
        }
        for model, entries in predictions.items():
            match = next((e for e in entries if e["id"].endswith(f"_{idx}")), None)
            row["preds"][model] = match["prediction"] if match else "—"
        rows.append(row)
    return rows


def collage_b64(stitched_dir: Path, scene_frame: str) -> str | None:
    img_path = stitched_dir / f"{scene_frame}.jpg"
    if not img_path.exists():
        print(f"[warn] stitched image not found: {img_path}", file=sys.stderr)
        return None
    return base64.b64encode(img_path.read_bytes()).decode()


def highlight_coords(text: str) -> str:
    """Escape text and wrap coordinate tokens in a styled span."""
    parts = []
    last = 0
    for m in COORD_TOKEN_RE.finditer(text):
        parts.append(escape(text[last:m.start()]))
        parts.append(f'<span class="coord">&lt;{escape(m.group(1))}&gt;</span>')
        last = m.end()
    parts.append(escape(text[last:]))
    return "".join(parts)



CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1117; color: #e2e8f0; font-family: system-ui, sans-serif; padding: 24px; }
h1 { font-size: 1.1rem; color: #94a3b8; margin-bottom: 4px; }
h2 { font-size: 0.85rem; color: #64748b; margin-bottom: 16px; font-weight: 400; }
.collage { width: 100%; border-radius: 8px; display: block; margin-bottom: 24px; }
table { width: 100%; border-collapse: collapse; font-size: 0.82rem; table-layout: fixed; }
th { background: #1e293b; color: #94a3b8; padding: 8px 10px; text-align: left;
     border-bottom: 2px solid #334155; }
td { padding: 8px 10px; border-bottom: 1px solid #1e293b; vertical-align: top;
     word-break: break-word; }
td.idx  { color: #64748b; font-size: 0.75rem; width: 2.5em; text-align: center; }
td.source { width: 13em; }
td.answer { }
tr.q-row td   { background: #161d2e; color: #94a3b8; font-style: italic;
                border-top: 2px solid #334155; }
tr.gt-row td.answer   { color: #86efac; }
tr.pred-row td.answer { color: #93c5fd; }
.badge { display: inline-block; font-size: 0.7rem; padding: 2px 7px; border-radius: 3px;
         background: #1e293b; color: #64748b; word-break: break-word; }
.cat  { display: inline-block; font-size: 0.68rem; padding: 1px 6px; border-radius: 3px;
        font-style: normal; margin-left: 6px; font-weight: 600; }
.coord { display: inline; background: #292524; color: #fb923c;
         border-radius: 3px; padding: 0 3px; font-size: 0.78em; font-family: monospace; }
"""


def render_html(scene_frame: str, b64: str | None, rows: list[dict], model_dirs: list[str]) -> str:
    scene, frame = scene_frame.split("_", 1)
    model_names = {m: short_name(m) for m in model_dirs}

    parts = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"]
    parts.append(f"<title>Demo: {scene_frame}</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append(f"<h1>Scene: <code>{escape(scene)}</code></h1>")
    parts.append(f"<h2>Frame: <code>{escape(frame)}</code></h2>")

    if b64:
        parts.append(f'<img class="collage" src="data:image/jpeg;base64,{b64}" alt="6-camera collage">')
    else:
        parts.append('<p style="color:#ef4444;margin-bottom:16px;">⚠ Stitched image not found.</p>')

    parts.append("<table><thead><tr>")
    parts.append("<th style='width:2.5em'>#</th><th style='width:13em'>Source</th><th>Answer</th>")
    parts.append("</tr></thead><tbody>")

    for row in rows:
        idx = row["idx"]

        # Category badge
        cat = row["category"]
        cat_color = CATEGORY_COLOR.get(cat, "#64748b")
        cat_html = (f'<span class="cat" style="background:{cat_color}20;color:{cat_color}">'
                    f'{escape(cat)}</span>') if cat else ""

        # Question row
        parts.append(f'<tr class="q-row">'
                     f'<td class="idx">{idx}</td>'
                     f'<td class="source"><span class="badge">Question</span>{cat_html}</td>'
                     f'<td class="answer">{highlight_coords(row["question"])}</td>'
                     f'</tr>')

        # Ground truth row
        parts.append(f'<tr class="gt-row">'
                     f'<td class="idx"></td>'
                     f'<td class="source"><span class="badge">GT Answer</span></td>'
                     f'<td class="answer">{highlight_coords(row["ground_truth"])}</td>'
                     f'</tr>')

        # One row per model
        for m in model_dirs:
            pred = row["preds"].get(m, "—")
            name = escape(model_names[m])
            parts.append(f'<tr class="pred-row">'
                         f'<td class="idx"></td>'
                         f'<td class="source"><span class="badge">{name}</span></td>'
                         f'<td class="answer">{highlight_coords(pred)}</td>'
                         f'</tr>')

    parts.append("</tbody></table></body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-scene HTML comparison demo.")
    parser.add_argument("--scene-frame", required=True,
                        help="scene_token_frame_token")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Model directory names under --outputs-dir (default: all)")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR)
    parser.add_argument("--stitched-dir", type=Path, default=DEFAULT_STITCHED_DIR)
    parser.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST_JSONL)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scene_frame = args.scene_frame
    output: Path = args.output or (DEFAULT_OUTPUT_DIR / f"{scene_frame}_demo.html")

    if args.models:
        model_dirs = args.models
    else:
        model_dirs = sorted(p.name for p in args.outputs_dir.iterdir()
                            if p.is_dir() and (p / "local_test.json").exists())

    if not model_dirs:
        print(f"[error] No model directories found in {args.outputs_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Models: {model_dirs}")

    categories = load_categories(args.test_jsonl)
    b64 = collage_b64(args.stitched_dir, scene_frame)
    predictions = load_predictions(args.outputs_dir, model_dirs, scene_frame)

    if not predictions:
        print(f"[error] No predictions found for '{scene_frame}'", file=sys.stderr)
        sys.exit(1)

    rows = align_rows(predictions, categories)
    print(f"QA pairs found: {len(rows)}")

    html = render_html(scene_frame, b64, rows, model_dirs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
