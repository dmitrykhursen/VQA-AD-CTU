"""Step 1 of the custom DriveLM split pipeline.

Extracts a fixed subset of QAs (with evaluation tags) from the raw DriveLM
train JSON, mirroring the test-distribution question types used in the
DriveLM challenge.

Usage:
    python src/datasets/extract_data.py \
        --input  /path/to/v1_1_train_nus_with_all_metainfo.json \
        --output data/drivelm_custom_split/intermediate/01_extracted.json
"""

import argparse
import json


def extract_data(root_path: str, save_path: str) -> None:
    with open(root_path) as f:
        train_file = json.load(f)

    test_data: dict = {}

    for scene_id in train_file:
        scene_data = train_file[scene_id]["key_frames"]

        test_data[scene_id] = {"key_frames": {}}

        for frame_id in scene_data:
            frame_data_infos = scene_data[frame_id]["key_object_infos"]
            frame_data_qa   = scene_data[frame_id]["QA"]
            image_paths     = scene_data[frame_id]["image_paths"]

            test_data[scene_id]["key_frames"][frame_id] = {
                "image_paths": image_paths,
                "QA": {cat: [] for cat in ("perception", "prediction", "planning", "behavior")},
            }
            fd = test_data[scene_id]["key_frames"][frame_id]["QA"]

            classes   = [v["Visual_description"].split(".")[0] for v in frame_data_infos.values()]
            locations = list(frame_data_infos.keys())

            # perception – class-description question (tag 2)
            for qa in frame_data_qa["perception"]:
                if all(cl.lower() in qa["A"].lower() for cl in classes):
                    fd["perception"].append({**qa, "tag": [2]})
                    break

            # perception – moving-status multiple-choice (tag 0)
            for qa in frame_data_qa["perception"]:
                if "what is the moving status of object" in qa["Q"].lower():
                    fd["perception"].append({**qa, "tag": [0]})
                    break

            # prediction – location/graph question (tag 3)
            for qa in frame_data_qa["prediction"]:
                if all(loc.lower() in qa["A"].lower() for loc in locations):
                    fd["prediction"].append({**qa, "tag": [3]})
                    break

            # prediction – yes/no question (tag 0)
            for qa in frame_data_qa["prediction"]:
                if "yes" in qa["A"].lower() or "no" in qa["A"].lower():
                    fd["prediction"].append({**qa, "tag": [0]})
                    break

            # planning – three question types (tag 1 each)
            added = {"ego": False, "collision": False, "safe": False}
            for qa in frame_data_qa["planning"]:
                q = qa["Q"].lower()
                if "what actions could the ego vehicle take" in q and not added["ego"]:
                    fd["planning"].append({**qa, "tag": [1]})
                    added["ego"] = True
                elif "lead to a collision" in q and not added["collision"]:
                    fd["planning"].append({**qa, "tag": [1]})
                    added["collision"] = True
                elif "safe actions" in q and not added["safe"]:
                    fd["planning"].append({**qa, "tag": [1]})
                    added["safe"] = True
                if all(added.values()):
                    break

            # behavior – all (tag 0)
            for qa in frame_data_qa["behavior"]:
                fd["behavior"].append({**qa, "tag": [0]})

    with open(save_path, "w") as f:
        json.dump(test_data, f, indent=4)

    print(f"Extracted data saved to: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract DriveLM QAs following test distribution.")
    parser.add_argument("--input",  required=True, help="Raw DriveLM train JSON (v1_1_train_nus_with_all_metainfo.json).")
    parser.add_argument("--output", required=True, help="Destination JSON path for extracted QAs.")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    extract_data(args.input, args.output)


if __name__ == "__main__":
    main()
