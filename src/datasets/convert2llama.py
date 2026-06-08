"""Step 3 of the custom DriveLM split pipeline.

Flattens the nested scene→frame→QA structure into a flat list of records
in LLaVA/LLaMA conversation format, adding `category` and `metric_type`
fields needed by downstream evaluation.

Usage:
    python src/datasets/convert2llama.py \
        --input  data/drivelm_custom_split/intermediate/02_converted.json \
        --output data/drivelm_custom_split/intermediate/03_llama.json
"""

import argparse
import json
import os

# Maps DriveLM integer tag to the evaluation metric name.
_TAG_TO_METRIC: dict[int, str] = {
    0: "match",    # multiple-choice / yes-no → exact match
    1: "chatgpt",  # planning free-form → GPT eval
    2: "accuracy", # class/description → accuracy
    3: "language", # graph/location → language similarity
}


def convert2llama(root: str, dst: str) -> None:
    with open(root) as f:
        data = json.load(f)

    records = []
    for scene_id, scene in data.items():
        for frame_id, frame in scene["key_frames"].items():
            image_paths = [
                frame["image_paths"][k].replace("..", "data")
                for k in frame["image_paths"]
            ]
            idx = 0
            for category in ("perception", "prediction", "planning", "behavior"):
                for qa in frame["QA"][category]:
                    tag = qa.get("tag", [])
                    metric_type = _TAG_TO_METRIC.get(tag[0], "unknown") if tag else "unknown"
                    records.append({
                        "id": f"{scene_id}_{frame_id}_{idx}",
                        "image": image_paths,
                        "conversations": [
                            {"from": "human", "value": "<image>\n" + qa["Q"]},
                            {"from": "gpt",   "value": qa["A"]},
                        ],
                        "tag":         tag,
                        "category":    category,
                        "metric_type": metric_type,
                    })
                    idx += 1

    with open(dst, "w") as f:
        json.dump(records, f, indent=4)

    print(f"Flattened {len(records)} QA records saved to: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Flatten DriveLM JSON to LLaMA conversation format.")
    parser.add_argument("--input",  required=True, help="Converted QA JSON (output of convert_data.py).")
    parser.add_argument("--output", required=True, help="Destination flat JSON path.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    convert2llama(args.input, args.output)


if __name__ == "__main__":
    main()
