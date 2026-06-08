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

        print(f"sum string_acc scores: {sum(scores)}")
        print(f"len string_acc scores: {len(scores)}")
        if len(scores) == 0:
            print("len string_acc scores == 0!!! RETURNING 0 string_acc !")
            return 0
        scores = sum(scores) / len(scores)
        return scores

    def eval_acc(self):
        if len(self.accuracy["answer"]) == 0:
            print("length of self.accuracy['answer']) == 0!!! RETURNING 0 acc !")
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
            print("len(data) == 0!!! RETURNING 0 gptscore !")
            return 0.0

        # ThreadPoolExecutor for concurrent API calls (avoids multiprocessing pickling issues)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(8, len(data))) as pool:
            raw_scores = list(pool.map(self.chatgpt_eval.forward, data))

        print(f"Raw GPT scores: {raw_scores}")

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
            print("self.language['answer']) == 0!!! RETURNING 0 language score total !")
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
            print("self.match[match][answer]) == 0!!! RETURNING 0 match !")
            return 0.0, 0.0

        outs1 = []
        for i in range(len(self.match["match"]["answer"])):
            answer = self.match["match"]["answer"][i]
            GT = self.match["match"]["GT"][i]
            _, F1_score = self.match_result(answer, GT)
            outs1.append(F1_score * 100)

        mean_outs1 = sum(outs1) / len(outs1)
        print(f"sum of f1-scores in match: {sum(outs1)}")
        print(f"len of f1-scores in match: {len(outs1)}")
        outs1 = mean_outs1
        outs2 = self.eval_chatGPT(self.match["GPT"])

        print(f"match without GPT score: {outs1}")
        print(f"GPT score inside match computation: {outs2}")

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


if __name__ == '__main__':
    import wandb

    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--root_path1', type=str, required=True, help='path to prediction file')
    parser.add_argument('--root_path2', type=str, default=None, help='path to GT test file (optional if root_path1 contains gt_answers)')
    parser.add_argument('--llama_format', action='store_true', help='treat root_path2 as llama-format GT (list of {id, conversations, tag})')
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--model_name', type=str, default="Ours")
    parser.add_argument('--latex_out', type=str, default="evaluation_results_latex.txt")
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
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

    single_file_mode = args.root_path2 is None

    if single_file_mode:
        missing = [i for i, item in enumerate(pred_file_raw) if "gt_answers" not in item]
        if missing:
            raise ValueError(f"'gt_answers' missing in {len(missing)} items. Provide --root_path2 or ensure all items contain 'gt_answers'.")
        test_file = pred_file_raw
        pred_file = {item["id"]: item for item in pred_file_raw}
    else:
        pred_file = {item["id"]: item for item in pred_file_raw}
        test_file = load_json_or_jsonl(args.root_path2)

    evaluation = evaluation_suit()

    current_frame_id = None
    first_flag = True

    if single_file_mode:
        for item in test_file:
            idx = item['id']
            question = item.get('question', '').replace("<image>\n", "").strip()
            GT = item['gt_answers']
            tags = item.get('tag', [2])
            tag = tags[0] if isinstance(tags, list) and len(tags) > 0 else tags
            predict = item['answer']
            parts = idx.split('_')
            frame_id = parts[1] if len(parts) >= 2 else "unknown"
            if frame_id != current_frame_id:
                current_frame_id = frame_id
                first_flag = True
            if first_flag:
                first_flag = False
                evaluation.set_graph(predict, GT)
                evaluation.forward([tag], predict, GT)
            else:
                if evaluation.eval_graph(question):
                    evaluation.forward([tag], predict, GT)

    elif args.llama_format or (not isinstance(test_file, dict)):
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
            try:
                predict = pred_file[idx]["answer"]
            except KeyError:
                print(f"skip: idx: {idx}")
                continue
            parts = idx.split('_')
            frame_id = parts[1] if len(parts) >= 2 else "unknown"
            if frame_id != current_frame_id:
                current_frame_id = frame_id
                first_flag = True
            if first_flag:
                first_flag = False
                evaluation.set_graph(predict, GT)
                evaluation.forward([tag], predict, GT)
            else:
                if evaluation.eval_graph(question):
                    evaluation.forward([tag], predict, GT)

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
                    idx = scene_id + "_" + frame_id + "_" + str(i)
                    try:
                        predict = pred_file[idx]["answer"]
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
                    else:
                        if evaluation.eval_graph(question):
                            evaluation.forward(tag, predict, GT)

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

    metrics_dict = {
        "step": args.step,
        "eval_drivelm_final_score": final_score,
        "eval_accuracy": output["accuracy"],
        "eval_chatgpt": output["chatgpt"],
        "eval_match": output["match"],
        "eval_pure_coordinate_match": pure_coordinate_match_score,
        "eval_lang_score": lang_score,
        **{f"eval_lang_{k}": v for k, v in lang_results.items()}
    }

    if args.wandb_project:
        wandb.log(metrics_dict, step=args.step)

    print(json.dumps(metrics_dict, indent=4))

    b1 = lang_results.get("val/Bleu_1", -1)
    b2 = lang_results.get("val/Bleu_2", -1)
    b3 = lang_results.get("val/Bleu_3", -1)
    b4 = lang_results.get("val/Bleu_4", -1)
    rouge = lang_results.get("val/ROUGE_L", -1.0)
    cider = lang_results.get("val/CIDEr", -1.0)

    latex_row = (
        f"{args.model_name} \\\\ Step {args.step} "
        f"str_acc: {output['string_accuracy'] * 100:.2f} & "
        f"{output['accuracy'] * 100:.2f} & {output['chatgpt']:.2f} & "
        f"{b1:.3f} & {b2:.3f} & {b3:.3f} & {b4:.3f} & "
        f"{rouge:.3f} & {cider:.4f} & {output['match']:.2f} & "
        f"{final_score * 100:.2f} \\\\"
    )
    with open(args.latex_out, "a") as f:
        f.write(latex_row + "\n")
    print(f"LaTeX row appended to: {args.latex_out}")
    print("Finished.")
