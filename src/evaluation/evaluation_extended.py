import re
import argparse
import json
import numpy as np
import torch.nn as nn
import language_evaluation

from src.evaluation.gpt_eval import GPTEvaluation


class evaluation_suit():
    def __init__(self):
        self.language_eval = language_evaluation.CocoEvaluator(coco_types=["BLEU", "ROUGE_L", "CIDEr"])
        self.chatgpt_eval = GPTEvaluation()
        self.GPT = []
        self.accuracy = {"answer": [], "GT": []}
        self.language = {"answer": [], "GT": []}
        self.str_acc = {"answer": [], "GT": []}
        self.match = {"match": {"answer": [], "GT": []}, "GPT": []}

    def eval_string_acc(self):
        scores = []
        for i in range(len(self.str_acc["answer"])):
            answer = self.str_acc["answer"][i]
            GT = self.str_acc["GT"][i]

            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        # print(f"sum string_acc scores: {sum(scores)}")
        # print(f"len string_acc scores: {len(scores)}")
        if len(scores) == 0:
            # print("len string_acc scores == 0!!! RETURNING 0 string_acc !")
            return 0
        scores = sum(scores) / len(scores)

        return scores

    def eval_acc(self):
        if len(self.accuracy["answer"]) == 0:
            # print("length of self.accuracy['answer']) == 0!!! RETURNING 0 acc !")
            return 0.0

        scores = []
        for i in range(len(self.accuracy["answer"])):
            answer = self.accuracy["answer"][i]
            GT = self.accuracy["GT"][i]
            if answer == GT:
                scores.append(1.0)
            else:
                scores.append(0.0)

        scores = sum(scores) / len(scores)
        return scores

    def eval_chatGPT(self, data):
        if len(data) == 0:
            # print("len(data) == 0!!! RETURNING 0 gptscore !")
            return 0.0

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(32, len(data))) as pool:
            raw_scores = list(pool.map(self.chatgpt_eval.forward, data))

        # print(f"Raw GPT scores: {raw_scores}")

        parsed_scores = []
        for s in raw_scores:
            match = re.search(r'\d+\.?\d*', str(s))
            if match:
                parsed_scores.append(float(match.group()))
            else:
                print(f"Warning: Could not parse a number from GPT response: '{s}'")
                parsed_scores.append(0.0)

        scores = sum(parsed_scores) / len(parsed_scores)
        return scores

    def eval_language(self):
        if len(self.language["answer"]) == 0:
            # print("self.language['answer']) == 0!!! RETURNING 0 language score total !")
            return {"val/BLEU": 0.0, "val/ROUGE_L": 0.0, "val/CIDEr": 0.0}

        answer = self.language["answer"]
        GT = self.language["GT"]
        results_gen = self.language_eval.run_evaluation(answer, GT)
        results_gen_dict = {
            f"val/{k}": v for k, v in results_gen.items()
        }
        return results_gen_dict

    def eval_match(self):
        if len(self.match["match"]["answer"]) == 0:
            # print("self.match[match][answer]) == 0!!! RETURNING 0 match !")
            return 0.0, 0.0

        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            _, F1_score = self.match_result(answer, GT)
            outs1.append(F1_score * 100)

        outs1 = sum(outs1) / len(outs1)
        outs2 = self.eval_chatGPT(self.match["GPT"])

        # print(f"match without GPT score: {outs1}")
        # print(f"GPT score inside match computation: {outs2}")

        scores = (outs1 + outs2) / 2.0
        pure_coordinate_match_score = outs1
        return pure_coordinate_match_score, scores

    def eval_graph(self, question):
        question_nums = re.findall(r'\d+\.\d+', question)
        question_nums = np.array([list(map(float, x.split()))[0] for x in question_nums]).reshape(-1, 2)
        question_nums = [list(i) for i in question_nums]
        for q in question_nums:
            if q not in self.graph:
                return False
        return True

    def match_result(self, answer, GT):
        answer_nums = re.findall(r'\d+\.\d+', answer)
        GT_nums = re.findall(r'\d+\.\d+', GT)
        if len(answer_nums) % 2 != 0:
            answer_nums = answer_nums[:-1]
        answer_nums = np.array([list(map(float, x.split()))[0] for x in answer_nums]).reshape(-1, 2)
        GT_nums = np.array([list(map(float, x.split()))[0] for x in GT_nums]).reshape(-1, 2)
        length = len(GT_nums)

        matched_out = []
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for pred in answer_nums:
            closest_distance = float('inf')
            closest_gt = None
            closest_id = None
            for i, gt in enumerate(GT_nums):
                distance = np.sum(np.abs(pred - gt))
                if distance < closest_distance:
                    closest_distance = distance
                    closest_gt = gt
                    closest_id = i

            if closest_distance < 16:
                true_positives += 1
                matched_out.append(closest_gt)
                GT_nums = np.delete(GT_nums, closest_id, axis=0)
            else:
                false_positives += 1

        false_negatives = length - true_positives
        precision = true_positives / (true_positives + false_positives + 1e-8)
        recall = true_positives / (true_positives + false_negatives + 1e-8)
        F1 = 2 * precision * recall / (precision + recall + 1e-8)

        return matched_out, F1

    def set_graph(self, answer, GT):
        self.graph, _ = self.match_result(answer, GT)
        self.graph = [list(i) for i in self.graph]

    def forward(self, tag, answer, GT):
        self.str_acc["answer"].append(answer)
        self.str_acc["GT"].append(GT)

        if 0 in tag:
            self.accuracy["answer"].append(answer)
            self.accuracy["GT"].append(GT)
        if 1 in tag:
            self.GPT.append((answer, GT))
        if 2 in tag:
            self.language["GT"].append(GT)
            self.language["answer"].append(answer)
        if 3 in tag:
            self.match["match"]["GT"].append(GT)
            self.match["match"]["answer"].append(answer)
            self.match["GPT"].append((answer, GT))

    def evaluation(self):
        print("evaluation start!")
        scores = {}
        scores["accuracy"] = self.eval_acc()
        scores["chatgpt"] = self.eval_chatGPT(self.GPT)
        scores["language"] = self.eval_language()
        scores["pure_coordinates_match"], scores["match"] = self.eval_match()
        scores["string_accuracy"] = self.eval_string_acc()
        return scores


# ---------------------------------------------------------------------------
# Extended analysis helpers
# ---------------------------------------------------------------------------

# Canonical question templates as they appear verbatim in the GT json field
# "question_template". Anything not in this set is reported as "unknown".
KNOWN_TEMPLATES = [
    "In this scenario, what are safe actions to take for the ego vehicle?",
    "Predict the behavior of the ego vehicle.",
    "Is <obj> a traffic sign or a road barrier?",
    "What actions could the ego vehicle take based on <obj>? Why take this action and what's the probability?",
    "What actions taken by the ego vehicle can lead to a collision with <obj>?",
    "What are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
    "What is the moving status of object <obj>?",
    "What object should the ego vehicle notice first when the ego vehicle is getting to the next possible location? What is the state of the object that is first noticed by the ego vehicle and what action should the ego vehicle take? What object should the ego vehicle notice second when the ego vehicle is getting to the next possible location? What is the state of the object perceived by the ego vehicle as second and what action should the ego vehicle take? What object should the ego vehicle notice third? What is the state of the object perceived by the ego vehicle as third and what action should the ego vehicle take?",
    "In this scenario, what object is most likely to consider <obj>?",
    "What are objects <position_to_ego_car>?",
    "Which object is most likely to be occluded by <obj>? Would this object affect the ego vehicle? Based on this object, what action of the ego vehicle is dangerous?",
    "Identify all the traffic elements in the front view, categorize them, determine their status, and predict the bounding box around each one. The output should be a list formatted as (c, s, x1, y1, x2, y2), where c represents the category, s denotes the status, and x1, y1, x2, y2 are the offsets of the top-left and bottom-right corners of the box relative to the center point.",
    "Except for the ego vehicle, what object would consider <obj> to be most relevant to its decision?",
    "What object would consider <obj> to be most relevant to its decision?",
    "What is the future state of <obj>?",
    "What does <obj> mean?",
    "What kind of traffic sign is <obj>?",
    "Please describe the current scene.",
]
_KNOWN_TEMPLATES_SET = set(KNOWN_TEMPLATES)
_UNKNOWN_KEY = "unknown"


def length_stats(lengths):
    if not lengths:
        return {"avg": 0, "min": 0, "max": 0, "median": 0, "count": 0}
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    mid = n // 2
    median = sorted_lens[mid] if n % 2 == 1 else (sorted_lens[mid - 1] + sorted_lens[mid]) / 2
    return {
        "avg": round(sum(lengths) / n, 2),
        "min": min(lengths),
        "max": max(lengths),
        "median": median,
        "count": n,
    }


def _forward_template(template_evals, template_answer_lengths, all_answer_lengths,
                      question_template, tag, predict, GT):
    """Record item into per-template suite and length trackers, bucketing unknowns."""
    key = question_template if question_template in _KNOWN_TEMPLATES_SET else _UNKNOWN_KEY
    if key not in template_evals:
        template_evals[key] = evaluation_suit()
    template_evals[key].forward([tag], predict, GT)

    length = len(predict.split())
    all_answer_lengths.append(length)
    if key not in template_answer_lengths:
        template_answer_lengths[key] = []
    template_answer_lengths[key].append(length)


def _print_template_block(label, res, lens):
    lstats = length_stats(lens)
    n = lstats["count"]
    print(f"\n  [{label}]  ({n} samples)")
    if res is None:
        print("    None")
        return
    if "error" in res:
        print(f"    ERROR: {res['error']}")
        return
    acc = res.get("accuracy", 0.0)
    chatgpt = res.get("chatgpt", 0.0)
    match = res.get("match", 0.0)
    pure_match = res.get("pure_coordinates_match", 0.0)
    str_acc = res.get("string_accuracy", 0.0)
    lang = res.get("language", {})
    bleu1 = lang.get("val/Bleu_1", lang.get("val/BLEU", 0.0))
    rouge = lang.get("val/ROUGE_L", 0.0)
    cider = lang.get("val/CIDEr", 0.0)
    print(f"    accuracy:      {acc * 100:.2f}%")
    print(f"    string_acc:    {str_acc * 100:.2f}%")
    print(f"    chatgpt:       {chatgpt:.2f}")
    print(f"    match (w/GPT): {match:.2f}  pure_coord: {pure_match:.2f}")
    print(f"    BLEU-1: {bleu1:.4f}  ROUGE_L: {rouge:.4f}  CIDEr: {cider:.4f}")
    print(f"    answer_length: avg={lstats['avg']}  min={lstats['min']}  "
          f"max={lstats['max']}  median={lstats['median']}")


def print_extended_report(template_results, template_answer_lengths, all_answer_lengths):
    print("\n" + "=" * 70)
    print("=== EXTENDED ANALYSIS ===")
    print("=" * 70)

    stats = length_stats(all_answer_lengths)
    print(f"\n--- Global Answer Length Stats (words) ---")
    print(f"  count: {stats['count']}  avg: {stats['avg']}  min: {stats['min']}  "
          f"max: {stats['max']}  median: {stats['median']}")

    print(f"\n--- Per Question-Template Results (known templates) ---")
    for tmpl in KNOWN_TEMPLATES:
        res = template_results.get(tmpl, None)
        lens = template_answer_lengths.get(tmpl, [])
        _print_template_block(tmpl, res, lens)

    print(f"\n--- Unknown Templates ---")
    unknown_res = template_results.get(_UNKNOWN_KEY, None)
    unknown_lens = template_answer_lengths.get(_UNKNOWN_KEY, [])
    if unknown_res is None and not unknown_lens:
        print("  None")
    else:
        _print_template_block(_UNKNOWN_KEY, unknown_res, unknown_lens)

    print("\n" + "=" * 70)


if __name__ == '__main__':
    import wandb

    parser = argparse.ArgumentParser(description='Evaluation (extended)')
    parser.add_argument('--root_path1', type=str, required=True, help='path to prediction file')
    parser.add_argument('--root_path2', type=str, default=None, help='path to GT test file (optional if root_path1 contains gt_answers)')
    parser.add_argument('--llama_format', action='store_true', help='treat root_path2 as llama-format GT (list of {id, conversations, tag})')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--model_name', type=str, default="Ours")
    parser.add_argument('--latex_out', type=str, default="evaluation_results_latex.txt")
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None, help='directory for output files; defaults to the directory of root_path1')
    parser.add_argument('--eval_all', action='store_true', help='evaluate all questions regardless of graph dependency')
    args = parser.parse_args()

    if args.wandb_project and args.wandb_run_name:
        wandb.init(project=args.wandb_project, id=args.wandb_run_name, resume="allow")

    def load_json_or_jsonl(path):
        """Load a .json (array or dict) or .jsonl (one object per line) file."""
        with open(path, 'r') as f:
            if path.endswith('.jsonl'):
                return [json.loads(line) for line in f if line.strip()]
            return json.load(f)

    pred_file_raw = load_json_or_jsonl(args.root_path1)

    def get_answer(item):
        """Accept 'answer' (eval format) or 'prediction' (inference output format)."""
        return item.get("answer") or item.get("prediction", "")

    def get_gt(item):
        """Accept 'gt_answers' (eval format) or 'ground_truth' (inference output format)."""
        return item.get("gt_answers") or item.get("ground_truth", "")

    single_file_mode = args.root_path2 is None

    if single_file_mode:
        missing = [i for i, item in enumerate(pred_file_raw)
                   if not item.get("gt_answers") and not item.get("ground_truth")]
        if missing:
            raise ValueError(f"'gt_answers'/'ground_truth' missing in {len(missing)} items. Provide --root_path2 or ensure all items contain the GT answer.")
        test_file = pred_file_raw
        pred_file = {item["id"]: item for item in pred_file_raw}
    else:
        pred_file = {item["id"]: item for item in pred_file_raw}
        test_file = load_json_or_jsonl(args.root_path2)

    evaluation = evaluation_suit()

    template_evals = {}
    template_answer_lengths = {}
    all_answer_lengths = []

    current_frame_id = None
    first_flag = True

    if single_file_mode:
        for item in test_file:
            idx = item['id']
            question = item.get('question', '').replace("<image>\n", "").strip()
            GT = get_gt(item)
            tags = item.get('tag', [2])
            tag = tags[0] if isinstance(tags, list) and len(tags) > 0 else tags
            predict = get_answer(item)
            question_template = item.get('question_template') or question[:80]
            parts = idx.split('_')
            frame_id = parts[1] if len(parts) >= 2 else "unknown"
            if frame_id != current_frame_id:
                current_frame_id = frame_id
                first_flag = True
            if first_flag:
                first_flag = False
                evaluation.set_graph(predict, GT)
                evaluation.forward([tag], predict, GT)
                _forward_template(template_evals, template_answer_lengths, all_answer_lengths,
                                  question_template, tag, predict, GT)
            else:
                if args.eval_all or evaluation.eval_graph(question):
                    evaluation.forward([tag], predict, GT)
                    _forward_template(template_evals, template_answer_lengths, all_answer_lengths,
                                      question_template, tag, predict, GT)

    elif args.llama_format or (not isinstance(test_file, dict)):
        print(f"len of test_file: {len(test_file)}")
        for item in test_file:
            idx = item['id']
            question = ""
            GT = ""
            for msg in item.get('conversations', []):
                if msg.get('from') == 'human':
                    question = msg.get('value', '').replace("<image>\n", "").strip()
                elif msg.get('from') == 'gpt':
                    GT = msg.get('value', '').strip()
            tags = item.get('tag', [])
            tag = tags[0] if len(tags) > 0 else -1
            question_template = item.get('question_template') or question[:80]
            predict = get_answer(pred_file[idx])
            parts = idx.split('_')
            frame_id = parts[1] if len(parts) >= 2 else "unknown"
            if frame_id != current_frame_id:
                current_frame_id = frame_id
                first_flag = True
            if first_flag:
                first_flag = False
                evaluation.set_graph(predict, GT)
                evaluation.forward([tag], predict, GT)
                _forward_template(template_evals, template_answer_lengths, all_answer_lengths,
                                  question_template, tag, predict, GT)
            else:
                if args.eval_all or evaluation.eval_graph(question):
                    evaluation.forward([tag], predict, GT)
                    _forward_template(template_evals, template_answer_lengths, all_answer_lengths,
                                      question_template, tag, predict, GT)

    elif isinstance(test_file, dict):
        for scene_id in test_file.keys():
            scene_data = test_file[scene_id]['key_frames']
            for frame_id in scene_data.keys():
                frame_data_qa = scene_data[frame_id]['QA']
                qa_list = (
                    frame_data_qa.get("perception", [])
                    + frame_data_qa.get("prediction", [])
                    + frame_data_qa.get("planning", [])
                    + frame_data_qa.get("behavior", [])
                )
                for i, qa in enumerate(qa_list):
                    question = qa['Q']
                    GT = qa['A']
                    tag = qa['tag']
                    question_template = question[:80]
                    idx = scene_id + "_" + frame_id + "_" + str(i)
                    try:
                        predict = get_answer(pred_file[idx])
                    except KeyError:
                        print(f"skip: idx: {idx}")
                        continue
                    if frame_id != current_frame_id:
                        current_frame_id = frame_id
                        first_flag = True
                    if first_flag:
                        first_flag = False
                        evaluation.set_graph(predict, GT)
                        evaluation.forward(tag, predict, GT)
                        _forward_template(template_evals, template_answer_lengths, all_answer_lengths,
                                          question_template, tag[0] if isinstance(tag, list) else tag,
                                          predict, GT)
                    else:
                        if args.eval_all or evaluation.eval_graph(question):
                            evaluation.forward(tag, predict, GT)
                            _forward_template(template_evals, template_answer_lengths, all_answer_lengths,
                                              question_template, tag[0] if isinstance(tag, list) else tag,
                                              predict, GT)

    output = evaluation.evaluation()
    print("string accuracy score: ", output["string_accuracy"])
    print("accuracy score: ", output["accuracy"])
    print("chatgpt score: ", output["chatgpt"])
    print("match score(w/ GPT score): ", output["match"])
    print("match score(without GPT score): ", output["pure_coordinates_match"])
    print("language score: ", output["language"])

    scores = []
    weights = [0.4, 0.2, 0.2, 0.2]
    scores.append(output["chatgpt"] / 100.)
    lang_score = 0
    lang_results = output["language"]
    for idx, key in enumerate(lang_results.keys()):
        if idx < 4:
            lang_score += lang_results[key] / 4. / 3.
        elif idx == 4:
            lang_score += lang_results[key] / 3.
        else:
            lang_score += lang_results[key] / 10. / 3.
    scores.append(lang_score)
    scores.append(output["match"] / 100.)
    scores.append(output["accuracy"])

    final_score = sum([x * y for x, y in zip(scores, weights)])
    pure_coordinate_match_score = output["pure_coordinates_match"] / 100.0

    prefix = "eval"
    metrics_dict = {
        "step": args.step,
        f"{prefix}_drivelm_final_score": final_score,
        f"{prefix}_accuracy": output["accuracy"],
        f"{prefix}_chatgpt": output["chatgpt"],
        f"{prefix}_match": output["match"],
        f"{prefix}_pure_coordinate_match": pure_coordinate_match_score,
        f"{prefix}_lang_score": lang_score,
        **{f"{prefix}_lang_{k}": v for k, v in lang_results.items()}
    }

    if args.wandb_project:
        wandb.log(metrics_dict, step=args.step)
        print(f"Logged metrics to WandB for step {args.step}")

    print(f"Metrics dict for step {args.step}: ")
    print(json.dumps(metrics_dict, indent=4))

    # --- Extended: compute per-template metrics ---
    print("\nComputing per-template metrics...")
    template_results = {}
    for tmpl, suite in template_evals.items():
        try:
            template_results[tmpl] = suite.evaluation()
        except Exception as e:
            template_results[tmpl] = {"error": str(e)}

    print_extended_report(template_results, template_answer_lengths, all_answer_lengths)

    def _to_python(obj):
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        return obj

    extended_output = {
        "global_metrics": metrics_dict,
        "global_answer_length": length_stats(all_answer_lengths),
        "per_template": {},
        "unknown_templates": None,
    }
    for tmpl in KNOWN_TEMPLATES:
        res = template_results.get(tmpl, None)
        extended_output["per_template"][tmpl] = {
            "metrics": _to_python(res) if res is not None else None,
            "answer_length": length_stats(template_answer_lengths.get(tmpl, [])),
        }
    if _UNKNOWN_KEY in template_results or _UNKNOWN_KEY in template_answer_lengths:
        extended_output["unknown_templates"] = {
            "metrics": _to_python(template_results.get(_UNKNOWN_KEY, None)),
            "answer_length": length_stats(template_answer_lengths.get(_UNKNOWN_KEY, [])),
        }

    import os
    out_dir = args.output_dir if args.output_dir else os.path.dirname(os.path.abspath(args.root_path1))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.root_path1))[0]
    ext_out_path = os.path.join(out_dir, f"{stem}_extended.json")
    with open(ext_out_path, "w") as f:
        json.dump(extended_output, f, indent=2)
    print(f"Extended saved to: {ext_out_path}")
    print("Finished.")
