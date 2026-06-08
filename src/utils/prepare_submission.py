# Modified from https://github.com/OpenDriveLab/DriveLM/blob/main/challenge/prepare_submission.py
#
# Wraps inference results into the DriveLM challenge submission envelope.
#
# Usage
# -----
# 1. Via inference script (automatic) — pass --submission to 06_inference.sh when
#    ANNOTATIONS points to the official val/test split (v1_1_val_nus_q_only*).
#    submission.json is written next to the inference output automatically:
#
#       bash scripts/06_inference.sh --submission
#
# 2. Manually on an existing inference output:
#
#       python -m src.utils.prepare_submission \
#           --input_json  inference/outputs/<model>/v1_1_val_nus_q_only_converted_llama.json \
#           --output_folder inference/outputs/<model> \
#           --method <model>
#
#    Accepts two input formats:
#      a. This repo's inference output — dict {"args": {...}, "predictions": [...]}
#      b. Original DriveLM format     — list  [{"id", "question", "answer", ...}]

import argparse
import json
import os

TEAM_INFO = {
    "team": "TODO_TEAM_NAME",
    "authors": ["TODO_AUTHOR"],
    "email": "TODO_EMAIL",
    "institution": "TODO_INSTITUTION",
    "country": "TODO_COUNTRY",
}


def _normalise(data) -> list:
    """Return a flat list in the original DriveLM schema regardless of input format."""
    if isinstance(data, list):
        # keep only the three fields the submission schema requires
        return [{"id": item["id"], "question": item["question"], "answer": item["answer"]} for item in data]

    # repo inference format: {"args": ..., "predictions": [...]}
    if isinstance(data, dict) and "predictions" in data:
        return [
            {
                "id": item["id"],
                "question": item.get("question", ""),
                "answer": item.get("prediction", ""),
            }
            for item in data["predictions"]
        ]

    raise ValueError(
        "Unrecognised input format: expected a list or a dict with a 'predictions' key."
    )


def prepare_submission(input_json: str, output_folder: str, method: str, output_name: str = "submission") -> str:
    """Wrap *input_json* results into a submission envelope and write <output_name>.json.

    Returns the path to the written file.
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    submission = {"method": method, **TEAM_INFO, "results": _normalise(data)}

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{output_name}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, indent=4, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Format inference results for DriveLM submission.")
    parser.add_argument("--input_json", required=True, help="Path to inference output JSON")
    parser.add_argument("--output_folder", default=".", help="Directory for the output file")
    parser.add_argument("--method", required=True, help="Method name (used as submission identifier)")
    parser.add_argument("--output_name", default="submission", help="Output filename stem (default: submission)")
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input file not found: {args.input_json}")

    out = prepare_submission(args.input_json, args.output_folder, args.method, args.output_name)
    print(f"Submission written to: {out}")


if __name__ == "__main__":
    main()
