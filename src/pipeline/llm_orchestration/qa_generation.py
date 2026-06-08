import argparse
import json
import math
import os
import random
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import regex as re
import torch
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

"""
Typical usage (via the project shell script):
  bash scripts/03_run_pipeline_nuscenes.sh

Or directly for a single scene / camera:
  python src/pipeline/llm_orchestration/qa_generation.py \
      --model Qwen/Qwen3-14B \
      --yolo_path data/nuscenes-drivelm_metadata/object_annotations/CAM_FRONT/SCENE_NAME \
      --tracks_path data/nuscenes-drivelm_metadata/object_tracks/SCENE_NAME/tracks.json \
      --prompts_config configs/pipeline/llm_prompt_config.yaml \
      --qas_ratios configs/pipeline/drivelm_qas_ratios_to_gen.json \
      --output_folder data/drivelm_aug_pseudo_labels \
      --file_name drivelm_pseudo_qas \
      --use_tracks --thinking --answer_formatting
"""

INIT_QUESTION = "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."

INIT_NUM_OBJ_DIST = {3: 0.3352, 4: 0.2731, 5: 0.2149, 6: 0.1768}

YES_NO_QUESTIONS = set(
    [
        "Would <obj> be in the moving direction of the ego vehicle?",
        "Will <obj> be in the moving direction of <obj>?",
        "Will <obj> change its motion state based on <obj>?",
        "Would <obj> take <obj> into account?",
        "Is <obj> a traffic sign or a road barrier?",
    ]
)

OPTIONS_QUESTIONS = set(
    [
        "Predict the behavior of the ego vehicle. Please select the correct answer from the following options:",
        "What is the moving status of object <obj>? Please select the correct answer from the following options:",
        "Predict the behavior of the ego vehicle. Please select the correct answer from the following options:",
    ]
)

MULTIPLE_PARTS_QUESTIONS = (
    set(
        [
            "What actions could the ego vehicle take based on <obj>? Why take this action and what's the probability?",
            "What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second when the ego vehicle is getting to the next possible location? What is the state of the object perceived by the ego vehicle as second and what action should the ego vehicle take? What object should the ego vehicle notice third? What is the state of the object perceived by the ego vehicle as third and what action should the ego vehicle take?",
        ]
    )
    | OPTIONS_QUESTIONS
)

EGO_ACTIONS = set(
    [
        "In this scenario, what are dangerous actions to take for the ego vehicle?",
        "In this scenario, what are safe actions to take for the ego vehicle?",
    ]
)

NO_ADDITIONAL_INFO = (
    set(
        [
            "What actions taken by the ego vehicle can lead to a collision with <obj>?",
            "Based on <obj> in this scene, what is the most possible action of the ego vehicle?",
            "Based on the observation of <obj>, what actions may <obj> take?",
            "What is the priority of the objects that the ego vehicle should consider?(in descending order)",
        ]
    )
    | YES_NO_QUESTIONS
    | OPTIONS_QUESTIONS
    | EGO_ACTIONS
)

QUESTION_RULES = [
    (YES_NO_QUESTIONS, ["yes_no_questions"]),
    (OPTIONS_QUESTIONS, ["multiple_choice_questions"]),
    # (MULTIPLE_PARTS_QUESTIONS, ["multiple_parts"]), # add manually at the end
    # (NO_ADDITIONAL_INFO, ["no_additional_info"]), # add manually at the end
    (EGO_ACTIONS, ["actions_for_ego"]),
    (
        {INIT_QUESTION},
        ["init_q_main"],
    ),
    (
        {"What actions could the ego vehicle take based on <obj>? Why take this action and what's the probability?"},
        ["actions_probability"],
    ),
    (
        {"What actions taken by the ego vehicle can lead to a collision with <obj>?"},
        ["collison"],
    ),
    (
        {
            "What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second when the ego vehicle is getting to the next possible location? What is the state of the object perceived by the ego vehicle as second and what action should the ego vehicle take? What object should the ego vehicle notice third? What is the state of the object perceived by the ego vehicle as third and what action should the ego vehicle take?"
        },
        ["notice_three_obj"],
    ),
    (
        {
            "Based on <obj> in this scene, what is the most possible action of the ego vehicle?",
            "Based on the observation of <obj>, what actions may <obj> take?",
        },
        ["most_possible_action"],
    ),
    (
        {"What is the priority of the objects that the ego vehicle should consider?(in descending order)"},
        ["priority_of_objects"],
    ),
    (
        {"Please describe the current scene."},
        ["describe_scene"],
    ),
    (
        {"What is the future state of <obj>?"},
        ["future_state"],
    ),
    (
        {"What does <obj> mean?"},
        ["object_meaning"],
    ),
    (
        {"What object would consider <obj> to be most relevant to its decision?"},
        ["most_relevant_object"],
    ),
    (
        {"Except for the ego vehicle, what object would consider <obj> to be most relevant to its decision?"},
        ["relevant_object_except_ego"],
    ),
    (
        {"What are objects to the <position>"},
        ["objects_to_position"],
    ),
    (
        {"In this scenario, what object is most likely to consider <obj>?"},
        ["likely_to_consider"],
    ),
    (
        {
            "Which object is most likely to be occluded by <obj>? Would this object affect the ego vehicle? Based on this object, what action of the ego vehicle is dangerous?"
        },
        ["occlusion_and_danger"],
    ),
    (
        {
            "Identify all the traffic elements in the front view, categorize them, determine their status, and predict the bounding box around each one. The output should be a list formatted as (c, s, x1, y1, x2, y2), where c represents the category, s denotes the status, and x1, y1, x2, y2 are the offsets of the top-left and bottom-right corners of the box relative to the center point."
        },
        ["traffic_elements_bounding_box"],
    ),
]


def get_prefixed_permutation(options):
    permuted = options.copy()
    random.shuffle(permuted)
    letters = ["A. ", "B. ", "C. ", "D. "]
    return [f"{letters[i]}{item}" for i, item in enumerate(permuted)]


def random_permutation(q_distribution: Dict[str, int], min_length: int, max_length: int, rng: random.Random) -> List[str]:
    available_keys = [k for k, v in q_distribution.items() if v > 0]
    if not available_keys:
        return []

    target_length = min(rng.randint(min_length, max_length), sum(q_distribution.values()))
    chosen = []

    rng.shuffle(available_keys)
    for q in available_keys:
        if len(chosen) >= target_length:
            break
        chosen.append(q)
        q_distribution[q] -= 1

    while len(chosen) < target_length:
        active_keys = [k for k, v in q_distribution.items() if v > 0]
        if not active_keys:
            break
        q = rng.choice(active_keys)
        chosen.append(q)
        q_distribution[q] -= 1

    return chosen


def get_test_distribution(ratio_data: List[Dict[str, Any]], no_dir_questions: Optional[List[str]] = None) -> Dict[str, float]:
    test_distribution = {}
    for entry in ratio_data:
        if entry["ratio_test"] > 0:
            if no_dir_questions is not None and entry["question"] not in no_dir_questions:
                continue
            question = entry["question"]
            test_distribution[question] = entry["ratio_test"]
    return test_distribution


def distribute_by_ratio(question_to_ratio: Dict[str, float], total_count: int) -> Dict[str, int]:
    ratio_sum = sum(question_to_ratio.values())
    if ratio_sum == 0:
        return {}
    normalized = {q: r / ratio_sum for q, r in question_to_ratio.items()}
    counts = {q: max(1, round(normalized[q] * total_count)) for q in question_to_ratio}
    return counts


def get_past(tracks: List[Dict[str, Any]], frames: List[int], current_frame: int, frames_back: int):
    start_f = current_frame - frames_back
    end_f = current_frame + 1
    i0 = bisect_left(frames, start_f)
    i1 = bisect_left(frames, end_f)
    window = tracks[i0:i1]
    expected = set(range(start_f, end_f))
    present = {d["frame"] for d in window}
    missing = expected - present
    return window, missing


def detections_to_text(
    data: Dict[str, Any],
    tracks_by_object_id: Optional[Dict[str, Any]] = None,
    frame: Optional[int] = None,
    frames_back: int = 5,
    track_type: str = "m",
    include_tracks: bool = False,
) -> str:
    """Converts bbox detected data into a JSON-formatted string representing the objects."""
    if include_tracks:
        if tracks_by_object_id is None or frame is None:
            raise ValueError("Tracks and frame must be provided when include_tracks is True.")
        if frame > 1426:  # manual fix for bad data
            return "[\n]"

    if data.get("categories") is None:
        return "[\n]"

    objects_list = []

    for category, cat_objs in data["categories"].items():
        for idx, obj in enumerate(cat_objs["objects"]):
            obj_dict = {
                "id": f"{category}_{idx}",
                "bbox": obj["bbox"],
                "center": obj["mid_point"],
            }

            if include_tracks and tracks_by_object_id:
                # Use the object's original ID to fetch tracks
                track_id = obj.get("id")
                tracks = tracks_by_object_id.get(track_id)

                if tracks is not None:
                    track_data = tracks["track"]

                    frames = [d["frame"] for d in track_data]

                    window, missing = get_past(
                        track_data,
                        frames,
                        frame,
                        frames_back,
                    )

                    if window:
                        # Determine loop sequence for history matching
                        if frame in missing:
                            time_range = range(len(window), 0, -1)
                        else:
                            time_range = range(len(window) - 1, -1, -1)

                        # Determine value formatting based on track type
                        if track_type == "m":
                            fields = ["x", "y", "z"]

                            def get_value(d):
                                return [round(d[field], 3) for field in fields]
                        elif track_type == "px":

                            def get_value(d):
                                return d["center_2d_px"]
                        else:
                            raise ValueError(f"Invalid track type: {track_type}")

                        obj_dict["pos_history"] = [get_value(d_item) for _, d_item in zip(time_range, window)]

            objects_list.append(obj_dict)

    # Format into the requested JSON string layout
    if not objects_list:
        return "[\n]"

    object_lines = [json.dumps(obj, separators=(",", ":")) for obj in objects_list]
    json_data = "[\n" + ",\n".join(object_lines) + "\n]"

    return json_data


def add_formatting(parts: list, answer_formatting: Dict[str, str], q: str) -> list[str]:
    for question_set, formatting_keys in QUESTION_RULES:
        if q in question_set:
            for key in formatting_keys:
                parts.append("\n" + answer_formatting[key])
                if q == INIT_QUESTION and key == "init_q_main":
                    selected_number = str(
                        np.random.choice(
                            a=list(INIT_NUM_OBJ_DIST.keys()),
                            p=list(INIT_NUM_OBJ_DIST.values()),
                        )
                    )
                    parts.append(answer_formatting["init_q_num_obj_" + selected_number])

    if q in MULTIPLE_PARTS_QUESTIONS:
        parts.append(answer_formatting["multiple_parts"])
    if q in NO_ADDITIONAL_INFO:
        parts.append(answer_formatting["no_additional_info"])

    return parts


def generate_prompt(
    q: str,
    description: Optional[str],
    objects: str,
    config_prompts: Dict[str, str],
    config_prompts_fewshot: Dict[str, str],
    config_prompts_answer_formatting: Dict[str, str],
    enforce_formatting: bool,
    directions: List[str],
    tracks: bool,
    track_type: str = "m",
    dataset_name: str = "nuscenes",
) -> str:
    parts = [
        config_prompts["context"],
        config_prompts["answer_rules"],
    ]

    parts.append(config_prompts["coordinate_system"])

    if tracks:
        parts.append(f"\n{config_prompts[dataset_name + '_tracks_coords_' + track_type]}")

    parts.append(f"\n{config_prompts['detected_objects' + ('_tracks' if tracks else '') + ('_' + dataset_name if tracks else '')]}")

    parts.append(objects)

    car_in_view_key = f"{dataset_name}_dataset_car_in_view"
    if car_in_view_key in config_prompts:
        parts.append(f"\n{config_prompts[car_in_view_key]}")

    if "<position>" in q:
        parts.append("\n" + config_prompts["position"])

    if any(x in q for x in ["important objects", "priority"]):
        parts.append("\n" + config_prompts["important_objects"])

    if description:
        parts.append(f"\nThe scene description is:\n {description}")

    if "<obj>" in q or q in MULTIPLE_PARTS_QUESTIONS or q in INIT_QUESTION:
        parts.append(config_prompts["obj"])
        parts.append(config_prompts_fewshot["obj"])
    else:
        parts.append(config_prompts["no_obj"])

    if "following options:" in q:
        parts.append(f"\nThe question to use is:\n {q + ' ' + ' '.join(get_prefixed_permutation(directions))}\n")
    else:
        parts.append(f"\nThe question to use is:\n {q}\n")

    if enforce_formatting:
        parts.append("Formating rules for the answer:")
        parts = add_formatting(parts, config_prompts_answer_formatting, q)

    return "".join(parts)


def generate_experiment_name(args: argparse.Namespace) -> str:
    model_short = args.model.split("/")[-1]
    think_tag = "think" if args.thinking else "no-think"
    track_tag = f"tracks-{args.track_type}" if args.tracks_path is not None else "no-tracks"
    return f"{model_short}_{think_tag}_{track_tag}_q{args.number_of_questions}"


def append_json_line(path: str, obj: Any):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


class ModelRespondervllm:
    def __init__(self, model_name: str, thinking: bool = True):
        self.thinking = thinking
        self.model_name = model_name

        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",  # local problem
            gpu_memory_utilization=0.95,
            enable_chunked_prefill=True,
            max_model_len=32768,
            enable_prefix_caching=True,
            # attention_backend="TRITON_ATTN",  # local problem
        )
        self.tokenizer = self.llm.get_tokenizer()

        # Pre-compile sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.6 if thinking else 0.7,
            top_p=0.95 if thinking else 0.8,
            top_k=20,
            max_tokens=32768,
        )

    def generate(self, messages: List[Dict[str, str]]):
        """Accepts a list of conversations and processes them in parallel."""
        outputs = self.llm.chat(
            messages,  # type: ignore
            self.sampling_params,
            use_tqdm=True,
            chat_template_kwargs={"enable_thinking": self.thinking},
        )

        text = outputs[0].outputs[0].text

        if "</think>" in text:
            think, answer = text.split("</think>", 1)
            think = think.replace("<think>", "").strip()
        else:
            think, answer = "", text.strip()

        return think, answer.strip()

    def generate_batch(self, batch_messages: List[List[Dict[str, str]]], chunk_size: int = 2):
        """Processes massive batches by chunking them to avoid OOM spikes, preserving original order."""
        all_results = []

        for i in range(0, len(batch_messages), chunk_size):
            chunk = batch_messages[i : i + chunk_size]

            outputs = self.llm.chat(
                chunk,  # type: ignore
                self.sampling_params,
                use_tqdm=True,
                chat_template_kwargs={"enable_thinking": self.thinking},
            )

            for out in outputs:
                text = out.outputs[0].text
                if "</think>" in text:
                    think, answer = text.split("</think>", 1)
                    think = think.replace("<think>", "").strip()
                else:
                    think, answer = "", text.strip()

                all_results.append((think, answer.strip()))

        return all_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B", help="HF model to use")
    parser.add_argument("--thinking", action="store_true", help="Enable Qwen thinking mode")
    parser.add_argument("--use_tracks", action="store_true", help="Enable Qwen thinking mode")
    parser.add_argument("--output_folder", type=str, default="data/drivelm_aug_pseudo_labels", help="Folder to save generated results")
    parser.add_argument("--file_name", type=str, default="drivelm_pseudo_qas", help="Base name for output JSONL files")
    parser.add_argument("--track_type", choices=["m", "px"], default="m")
    parser.add_argument("--frames_back", type=int, default=5, help="How many frames back to include in the track information.")
    parser.add_argument(
        "--yolo_path",
        type=str,
        default=None,
        help="Path to annotation directory containing merged_processed.json (output of Stage 2)",
    )
    parser.add_argument("--number_of_questions", type=int, default=15)
    parser.add_argument("--camera", type=str, default="CAM_FRONT", help="Camera name used in annotation data")
    parser.add_argument("--tracks_path", type=str, default=None, help="Path to tracks JSON file (output of Stage 2)")
    parser.add_argument("--answer_formatting", action="store_true", help="Try to enforce DriveLM answer formatting")
    parser.add_argument(
        "--test", action="store_true", help="Test mode: generate 5 of each q with test_ratio > 0.5 uniformly in one folder, print prompts."
    )
    parser.add_argument("--qas_ratios", type=str, default="configs/pipeline/drivelm_qas_ratios_to_gen.json")
    parser.add_argument("--prompts_config", type=str, default="configs/pipeline/llm_prompt_config.yaml")
    parser.add_argument("--dataset_name", type=str, default="nuscenes", help="Dataset name; selects dataset-specific prompts from the config")

    # --- Chunking Arguments for SLURM Arrays ---
    parser.add_argument("--chunk_id", type=int, default=0, help="Which chunk this worker processes (0-indexed)")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of workers/chunks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args)
    exp_name = generate_experiment_name(args)
    # =============================================
    #               Data Loading
    # =============================================
    if not os.path.exists(os.path.join(args.yolo_path, "merged_processed.json")):
        from parse_yolo import run_yolo_processing

        data = run_yolo_processing(
            input_path=args.yolo_path,
            output_path=args.yolo_path,
            camera_id=args.camera,
            output_name="merged_processed.json",
        )
    else:
        print("Loading processed YOLO data...")
        with open(os.path.join(args.yolo_path, "merged_processed.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

    with open(args.qas_ratios, "r", encoding="utf-8") as f:
        ratio_data = json.load(f)
    with open(args.prompts_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_question_pool = []
    if args.test:
        target_questions = []
        test_question_pool = target_questions * 15
        rng_test = random.Random(42)  # fixed seed to ensure uniformity
        rng_test.shuffle(test_question_pool)
        print(f"\n[TEST MODE ENABLED] Selected {len(target_questions)} question types. Pool size: {len(test_question_pool)}.")

    q_ratios = get_test_distribution(ratio_data)
    config_prompts = config["prompts"]
    config_prompts_fewshot = config["fewshot"]
    config_prompts_answer_formatting = config["answer_formatting"]
    directions = ["Turn right.", "Drive backward.", "Going ahead.", "Turn left."]

    q_ratios = get_test_distribution(ratio_data)
    num_of_scenes = len(data)
    q_dist = distribute_by_ratio(q_ratios, num_of_scenes * args.number_of_questions)
    tracks_by_object_id = None
    if args.tracks_path:
        with open(args.tracks_path, "r", encoding="utf-8") as f:
            tracks = json.load(f)
        tracks_by_object_id = {d["object_id"]: d for d in tracks["tracks"]}

    scene_objects = {
        int(img): detections_to_text(
            obj_info,
            tracks_by_object_id=tracks_by_object_id,
            frame=int(img),
            frames_back=args.frames_back,
            track_type=args.track_type,
            include_tracks=args.use_tracks,
        )
        for img, obj_info in data.items()
    }
    output_file = Path(args.output_folder) / f"{args.file_name}_{args.chunk_id}.jsonl"
    responder = ModelRespondervllm(args.model, thinking=args.thinking)
    pattern = r"\b\S*\.\w+_\d+\b"
    chronological_scenes = sorted(scene_objects.items())
    rng = random.Random(42)
    total_scenes = len(chronological_scenes)
    chunk_size = math.ceil(total_scenes / args.num_chunks)
    start_idx = args.chunk_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_scenes)

    # fast forward to the correct rng state fot the chunk
    for id, objects in chronological_scenes[:start_idx]:
        if args.tracks_path is not None and (id < args.frames_back + 1 or (args.test and id < 15)):
            continue
        _ = random_permutation(q_dist, args.number_of_questions, args.number_of_questions, rng)

    for id, objects in tqdm(chronological_scenes[start_idx:end_idx], desc="Processing scenes"):
        if args.tracks_path is not None and (id < args.frames_back + 1 or (args.test and id < 15)):
            continue

        questions = random_permutation(q_dist, args.number_of_questions, args.number_of_questions, rng)
        scene_results = {}
        used_objs = set()
        with torch.inference_mode():
            for i, q in enumerate(questions):
                prompt = generate_prompt(
                    q=q,
                    description=None,
                    objects=objects,
                    config_prompts=config_prompts,
                    config_prompts_fewshot=config_prompts_fewshot,
                    config_prompts_answer_formatting=config_prompts_answer_formatting,
                    enforce_formatting=args.answer_formatting,
                    directions=directions,
                    tracks=args.tracks_path is not None,
                    track_type=args.track_type,
                )
                # print(80*'-')
                # print(prompt)
                # print(80*'-')
                if used_objs and "<obj>" in prompt:
                    prompt += f"\nAvoid reusing: {', '.join(sorted(used_objs))}"

                    # --- Print Prompt in Test Mode ---
                if args.test:
                    print(f"\n\n{'=' * 70}\n[TEST MODE] Prompt for: {q}\n{'-' * 70}\n{prompt}\n{'=' * 70}\n")
                messages = [{"role": "user", "content": prompt}]
                _, content = responder.generate(messages)
                # print(f"Model response:\n{content}\n{'=' * 70}\n")
                try:
                    clean_content = content.strip()
                    clean_content = "{" + f'"Question_{i}": ' + clean_content + "}"
                    parsed = json.loads(clean_content)
                    scene_results.update(parsed)
                    question_text = parsed[f"Question_{i}"]["question"]
                    used_objs.update(re.findall(pattern, question_text))
                    if args.test:
                        print("Next <obj> prompt will have:")
                        print(f"\nAvoid reusing: {', '.join(sorted(used_objs))}")
                except json.JSONDecodeError as e:
                    print(f"[Warning] JSON decode error for question {i}: {e}\n{content}")

            append_json_line(str(output_file), {id: scene_results})

            # --- Early Exit for Test Mode ---
            if args.test:
                break
    print("Ran out of scenes.")
