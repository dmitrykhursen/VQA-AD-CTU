"""Step 2 of the custom DriveLM split pipeline.

Augments the extracted QAs: injects multiple-choice answer options for
perception (moving-status) and behavior questions using fixed rule sets,
matching the DriveLM challenge evaluation format.

Usage:
    python src/datasets/convert_data.py \
        --input  data/drivelm_custom_split/intermediate/01_extracted.json \
        --output data/drivelm_custom_split/intermediate/02_converted.json
"""

import argparse
import json
import random


_ACTIONS = [
    "Going ahead.", "Turn right.", "Turn left.", "Stopped.",
    "Back up.", "Reverse parking.", "Drive backward.",
]

_BEHAVIORS = [
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving very fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving with normal speed.",
    "The ego vehicle is steering to the left. The ego vehicle is driving fast.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving fast.",
    "The ego vehicle is going straight. The ego vehicle is driving slowly.",
    "The ego vehicle is going straight. The ego vehicle is driving with normal speed.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving with normal speed.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving slowly.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving slowly.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving very fast.",
    "The ego vehicle is steering to the right. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the right. The ego vehicle is driving very fast.",
    "The ego vehicle is slightly steering to the left. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving very fast.",
    "The ego vehicle is going straight. The ego vehicle is not moving.",
    "The ego vehicle is slightly steering to the right. The ego vehicle is driving with normal speed.",
    "The ego vehicle is steering to the right. The ego vehicle is driving slowly.",
    "The ego vehicle is steering to the right. The ego vehicle is driving with normal speed.",
    "The ego vehicle is going straight. The ego vehicle is driving very fast.",
    "The ego vehicle is going straight. The ego vehicle is driving fast.",
    "The ego vehicle is steering to the left. The ego vehicle is driving slowly.",
]

_LETTERS = {0: "A", 1: "B", 2: "C", 3: "D"}


def _make_mc(question: str, answer: str, pool: list[str]) -> dict:
    distractors = [x for x in pool if x != answer]
    choices = random.sample(distractors, 3) + [answer]
    random.shuffle(choices)
    suffix = " Please select the correct answer from the following options: " + " ".join(
        f"{_LETTERS[i]}. {choices[i]}" for i in range(4)
    )
    return {"Q": question + suffix, "A": _LETTERS[choices.index(answer)]}


def convert_data(root: str, dst: str) -> None:
    with open(root) as f:
        data = json.load(f)

    for scene_id in data:
        for frame_id in data[scene_id]["key_frames"]:
            qa_dict = data[scene_id]["key_frames"][frame_id]["QA"]

            for qa in qa_dict["perception"]:
                if "what is the moving status of object" in qa["Q"].lower():
                    mc = _make_mc(qa["Q"], qa["A"], _ACTIONS)
                    qa["Q"] = mc["Q"]
                    qa["A"] = mc["A"]

            for qa in qa_dict["behavior"]:
                mc = _make_mc(qa["Q"], qa["A"], _BEHAVIORS)
                qa["Q"] = mc["Q"]
                qa["A"] = mc["A"]

    with open(dst, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Converted data saved to: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add multiple-choice options to DriveLM QAs.")
    parser.add_argument("--input",  required=True, help="Extracted QA JSON (output of extract_data.py).")
    parser.add_argument("--output", required=True, help="Destination JSON path.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for MC distractor sampling (default: 0).")
    args = parser.parse_args()

    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    random.seed(args.seed)
    convert_data(args.input, args.output)


if __name__ == "__main__":
    main()
