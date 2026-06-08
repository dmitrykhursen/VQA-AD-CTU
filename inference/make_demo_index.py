"""Generate an index.html gallery from all *_demo.html files in a demo directory.

Usage:
    python inference/make_demo_index.py \
        [--demo-dir data/demo] \
        [--stitched-dir /path/to/stitched] \
        [--output data/demo/index.html]
"""

import argparse
import base64
import io
import sys
from html import escape
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEMO_DIR = REPO_ROOT / "docs"
DEFAULT_STITCHED_DIR = Path("/mnt/proj1/eu-25-10/datasets/DRIVE_LM_zipped/nuscenes/stitched")

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def thumb_b64(stitched_dir: Path, scene_frame: str, width: int = 480) -> str | None:
    img_path = stitched_dir / f"{scene_frame}.jpg"
    if not img_path.exists():
        return None
    if HAS_PIL:
        img = Image.open(img_path)
        h = int(img.height * width / img.width)
        img = img.resize((width, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return base64.b64encode(buf.getvalue()).decode()
    # Fallback: embed full image (larger but no PIL needed)
    return base64.b64encode(img_path.read_bytes()).decode()


CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1117; color: #e2e8f0;
       font-family: system-ui, sans-serif; padding: 32px; }
h1 { font-size: 1.3rem; color: #94a3b8; margin-bottom: 8px; }
p.sub { font-size: 0.82rem; color: #64748b; margin-bottom: 28px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
.card { background: #1e293b; border-radius: 8px; overflow: hidden;
        transition: transform 0.15s; text-decoration: none; color: inherit; display: block; }
.card:hover { transform: translateY(-3px); }
.card img { width: 100%; display: block; }
.card .info { padding: 10px 12px; }
.card .scene { font-size: 0.72rem; color: #64748b; font-family: monospace; margin-bottom: 2px; }
.card .frame { font-size: 0.72rem; color: #475569; font-family: monospace; }
.card .open  { font-size: 0.75rem; color: #3b82f6; margin-top: 6px; }
"""


def render_index(cards: list[dict]) -> str:
    parts = ["<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"]
    parts.append("<title>VQA-AD Demo Gallery</title>")
    parts.append(f"<style>{CSS}</style></head><body>")
    parts.append("<h1>VQA-AD Demo Gallery</h1>")
    parts.append(f'<p class="sub">{len(cards)} scenes — click a card to open the full demo</p>')
    parts.append('<div class="grid">')

    for card in cards:
        scene_frame = card["scene_frame"]
        scene, frame = scene_frame.split("_", 1)
        href = escape(card["href"])
        b64 = card["b64"]

        if b64:
            img_tag = f'<img src="data:image/jpeg;base64,{b64}" alt="{escape(scene_frame)}">'
        else:
            img_tag = '<div style="height:120px;background:#0f1117"></div>'

        parts.append(
            f'<a class="card" href="{href}">'
            f'{img_tag}'
            f'<div class="info">'
            f'<div class="scene">scene: {escape(scene[:20])}…</div>'
            f'<div class="frame">frame: {escape(frame[:20])}…</div>'
            f'<div class="open">Open demo →</div>'
            f'</div></a>'
        )

    parts.append("</div></body></html>")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo gallery index page.")
    parser.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR)
    parser.add_argument("--stitched-dir", type=Path, default=DEFAULT_STITCHED_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    demo_dir: Path = args.demo_dir
    output: Path = args.output or (demo_dir / "index.html")

    html_files = sorted(demo_dir.glob("*_demo.html"))
    if not html_files:
        print(f"[error] No *_demo.html files found in {demo_dir}", file=sys.stderr)
        sys.exit(1)

    if not HAS_PIL:
        print("[warn] Pillow not found — thumbnails will not be resized (larger file size)")

    cards = []
    for html_file in html_files:
        scene_frame = html_file.name.replace("_demo.html", "")
        b64 = thumb_b64(args.stitched_dir, scene_frame)
        cards.append({
            "scene_frame": scene_frame,
            "href": html_file.name,
            "b64": b64,
        })
        status = "ok" if b64 else "no image"
        print(f"  {scene_frame[:24]}… [{status}]")

    html = render_index(cards)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Saved: {output}  ({len(cards)} cards)")


if __name__ == "__main__":
    main()
