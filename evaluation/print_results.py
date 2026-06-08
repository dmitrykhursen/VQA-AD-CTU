#!/usr/bin/env python3
"""
Print a unified DriveLM evaluation table for all models found under evaluation/results/.

Usage:
    python3 print_results.py [--split local_test|local_val] [--results-dir PATH]

Layout:
    1. Pretrained baselines
    2. Fine-tuned models
    3. Finetuned (offline) — extra "Visual Ann." column shows annotation strategy
    ANSI bold highlights the best value in each column (per section).
"""

import argparse
import json
import sys
from pathlib import Path

BOLD  = "\033[1m"
RESET = "\033[0m"

RESULTS_DIR = Path(__file__).parent / "results"

# folder_name → (display_name, group, author, annotation)
# annotation: non-empty only for offline models; becomes "Visual Ann." column value.
# display_name is always the unique internal key.
MODEL_REGISTRY = {
    "OpenGVLab__InternVL2-2B": (
        "InternVL2-2B",              "pretrained", "OpenGVLab", ""),
    "llava-hf__llava-v1.6-mistral-7b-hf": (
        "LLaVA-1.6",                 "pretrained", "llava-hf",  ""),
    "OpenGVLab__llama_adapter_v2_multimodal7b": (
        "LLaMA-Adapter-v2",          "pretrained", "OpenGVLab", ""),
    "OpenGVLab__Mini-InternVL2-2B-DA-DriveLM": (
        "Mini-DA†",                  "finetuned",  "OpenGVLab", ""),
    "dkhursen__InternVL2-2b-LoRA-25k_plus_DL-PL-10pct": (
        "LoRA-25k+DL-PL-10%",        "finetuned",  "dkhursen",  ""),
    "dkhursen__InternVL2-2b-LoRA-25k_plus_DL-PL-30pct": (
        "LoRA-25k+DL-PL-30%",        "finetuned",  "dkhursen",  ""),
    "dkhursen__InternVL2-2b-LoRA-25k_plus_DL-PL-50pct": (
        "LoRA-25k+DL-PL-50%",        "finetuned",  "dkhursen",  ""),
    "dkhursen__InternVL2-2b-LoRA-25k_plus_DL-PL-100pct": (
        "LoRA-25k+DL-PL-100%",       "finetuned",  "dkhursen",  ""),
    "dkhursen__InternVL2-2b-LoRA-25k_plus_Valeo": (
        "LoRA-25k+Valeo",            "finetuned",  "dkhursen",  ""),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm": (
        "LoRA-25k (local trainset)", "finetuned",  "dkhursen",  ""),
    "dkhursen__InternVL2-2b-LoRA-300k-drivelm": (
        "LoRA-300k",                 "finetuned",  "dkhursen",  ""),
    # Offline: annotation = visual strategy shown in "Visual Annotation" column.
    # Model column shows "InternVL2-2B-LoRA-25k" for all offline rows.
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redbbox-ctag-bkgd": (
        "Red BBox + CTags + Bkgd",   "offline",    "dkhursen",  "Red BBox + CTags + Bkgd"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redbbox-ctag": (
        "Red BBox + CTags",          "offline",    "dkhursen",  "Red BBox + CTags"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redbbox": (
        "Red BBox",                  "offline",    "dkhursen",  "Red BBox"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag-bkgd": (
        "Red Circle + CTags + Bkgd", "offline",    "dkhursen",  "Red Circle + CTags + Bkgd"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redcircle-ctag": (
        "Red Circle + CTags",        "offline",    "dkhursen",  "Red Circle + CTags"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redcircle": (
        "Red Circle",                "offline",    "dkhursen",  "Red Circle"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redmidpoint-ctag-bkgd": (
        "Red Midpoint + CTags + Bkgd","offline",   "dkhursen",  "Red Midpoint + CTags + Bkgd"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redmidpoint-ctag": (
        "Red Midpoint + CTags",      "offline",    "dkhursen",  "Red Midpoint + CTags"),
    "dkhursen__InternVL2-2b-LoRA-25k-drivelm-offline-redmidpoint": (
        "Red Midpoint",              "offline",    "dkhursen",  "Red Midpoint"),
}

PRETRAINED_ORDER = ["InternVL2-2B", "LLaVA-1.6", "LLaMA-Adapter-v2"]
FINETUNED_ORDER  = [
    "Mini-DA†",
    "LoRA-25k+DL-PL-10%",
    "LoRA-25k+DL-PL-30%",
    "LoRA-25k+DL-PL-50%",
    "LoRA-25k+DL-PL-100%",
    "LoRA-25k+Valeo",
    "LoRA-25k (local trainset)",
    "LoRA-300k",
]
OFFLINE_ORDER = [
    "Red BBox + CTags + Bkgd",
    "Red BBox + CTags",
    "Red BBox",
    "Red Circle + CTags + Bkgd",
    "Red Circle + CTags",
    "Red Circle",
    "Red Midpoint + CTags + Bkgd",
    "Red Midpoint + CTags",
    "Red Midpoint",
]

# (header, json_key, divisor)
METRICS = [
    ("Final",   "eval_drivelm_final_score",    1.0),
    ("Acc",     "eval_accuracy",               1.0),
    ("ChatGPT", "eval_chatgpt",              100.0),
    ("Lang",    "eval_lang_score",             1.0),
    ("B1",      "eval_lang_val/Bleu_1",        1.0),
    ("B2",      "eval_lang_val/Bleu_2",        1.0),
    ("B3",      "eval_lang_val/Bleu_3",        1.0),
    ("B4",      "eval_lang_val/Bleu_4",        1.0),
    ("RL",      "eval_lang_val/ROUGE_L",       1.0),
    ("CIDEr",   "eval_lang_val/CIDEr",         1.0),
    ("Match",   "eval_match",                100.0),
    ("Coord",   "eval_pure_coordinate_match",  1.0),
]

OFFLINE_MODEL_NAME = "InternVL2-2B-LoRA-25k"


def _lang_score_fallback(gm):
    """Approximate lang score from individual NLP metrics when not stored.
    CIDEr may be stored raw in [0,10] for offline models; normalise if so."""
    keys = [
        "eval_lang_val/Bleu_1", "eval_lang_val/Bleu_2",
        "eval_lang_val/Bleu_3", "eval_lang_val/Bleu_4",
        "eval_lang_val/ROUGE_L", "eval_lang_val/CIDEr",
    ]
    vals = []
    for k in keys:
        v = gm.get(k)
        if v is None:
            continue
        if k == "eval_lang_val/CIDEr" and v > 1.0:
            v /= 10.0
        vals.append(v)
    return sum(vals) / len(vals) if vals else None


def load_results(results_dir, split="local_test"):
    rows = {}
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir():
            continue
        json_path = folder / f"{split}.json"
        if not json_path.exists():
            continue
        try:
            data = json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        gm = data.get("global_metrics", {})
        if not gm:
            continue

        entry = MODEL_REGISTRY.get(folder.name)
        if entry:
            display, group, author, annot = entry
        else:
            display, group, author, annot = folder.name, "finetuned", "unknown", ""

        if display in rows:
            continue

        model_col = OFFLINE_MODEL_NAME if annot else display
        annot_col = annot if annot else "-"

        values = {}
        for _, key, div in METRICS:
            raw = gm.get(key)
            if raw is not None:
                values[key] = raw / div
            elif key == "eval_lang_score":
                values[key] = _lang_score_fallback(gm)
            else:
                values[key] = None

        rows[display] = {
            "group": group, "author": author,
            "model_col": model_col, "annot_col": annot_col,
            "values": values,
        }
    return rows


def _fmt(val):
    if val is None:
        return "  N/A"
    return f"{val:.3f}"


def print_table(rows, split="local_test"):
    if not rows:
        print(f"No results found for split='{split}'.", file=sys.stderr)
        return

    author_w = max(len(r["author"])    for r in rows.values()) + 1
    model_w  = max(len(r["model_col"]) for r in rows.values()) + 1
    annot_w  = max(len(r["annot_col"]) for r in rows.values()) + 1
    col_w    = 7

    headers = [m[0] for m in METRICS]
    col_ws  = [max(len(h) + 1, col_w) for h in headers]

    def _best_for(names):
        best = {}
        for _, key, _ in METRICS:
            vals = [
                rows[n]["values"][key]
                for n in names
                if n in rows and rows[n]["values"][key] is not None
            ]
            best[key] = max(vals) if vals else None
        return best

    hdr_metric = "|".join(h.rjust(col_ws[i]) for i, h in enumerate(headers))
    hdr_line   = (
        f"| {'Author':<{author_w}} |"
        f" {'Model':<{model_w}} |"
        f" {'Visual Annotation':<{annot_w}} |"
        + hdr_metric
        + "|"
    )

    def _hline(char="-"):
        return (
            "+"
            + char * (author_w + 2)
            + "+"
            + char * (model_w + 2)
            + "+"
            + char * (annot_w + 2)
            + "+"
            + "+".join(char * w for w in col_ws)
            + "+"
        )

    def _row(name, values, best):
        cells = []
        for i, (_, key, _) in enumerate(METRICS):
            v = values[key]
            s = _fmt(v).rjust(col_ws[i])
            if v is not None and best[key] is not None and abs(v - best[key]) < 1e-9:
                s = BOLD + s + RESET
            cells.append(s)
        return (
            f"| {rows[name]['author']:<{author_w}} |"
            f" {rows[name]['model_col']:<{model_w}} |"
            f" {rows[name]['annot_col']:<{annot_w}} |"
            + "|".join(cells)
            + "|"
        )

    def _section(title, names_order, group_key):
        present   = [n for n in names_order if n in rows]
        extras    = [n for n, r in rows.items()
                     if r["group"] == group_key and n not in names_order]
        all_names = present + extras
        if not all_names:
            return set()
        best    = _best_for(all_names)
        total_w = author_w + model_w + annot_w + 8
        print(_hline("="))
        print(f"|{title.center(total_w)}|" + "|".join(" " * w for w in col_ws) + "|")
        print(_hline("-"))
        print(hdr_line)
        print(_hline("-"))
        for name in all_names:
            print(_row(name, rows[name]["values"], best))
        return set(all_names)

    print()
    print(
        f"  DriveLM Evaluation Results  —  split: {split}"
        f"  |  {BOLD}bold{RESET} = best per column within each section"
    )

    _section(" Pretrained ", PRETRAINED_ORDER, "pretrained")
    _section(" Finetuned (base model: InternVL2-2B)", FINETUNED_ORDER, "finetuned")
    _section(" Finetuned (offline) ", OFFLINE_ORDER, "offline")

    print(_hline("="))
    print("  Lang    = composite language score (avg of B1–B4, RL, CIDEr)")
    print("  ChatGPT, Match normalised to [0,1]  (raw value ÷ 100)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Print DriveLM evaluation results table.")
    parser.add_argument("--split",       default="local_test",     help="Split file to read (default: local_test)")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Override results directory")
    args = parser.parse_args()

    rows = load_results(Path(args.results_dir), args.split)
    print_table(rows, args.split)


if __name__ == "__main__":
    main()
