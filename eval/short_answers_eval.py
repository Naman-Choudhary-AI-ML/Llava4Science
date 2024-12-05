import os
import json
import argparse
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util

# Load a semantic similarity model
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight semantic model for sentence similarity

def normalize_answer(answer):
    """Normalize answers by handling case, True/False equivalents, and tokenization."""
    answer = answer.strip().lower()
    if answer in {"true", "yes"}:
        return "true"
    elif answer in {"false", "no"}:
        return "false"
    return answer

def tokens_match(pred_answer, gt_answer):
    """
    Check if the predicted answer matches the ground truth answer
    using token overlap or exact match.
    """
    pred_tokens = set(normalize_answer(pred_answer).split())
    gt_tokens = set(normalize_answer(gt_answer).split())
    return pred_tokens.issubset(gt_tokens) or gt_tokens.issubset(pred_tokens)

def semantic_match(pred_answer, gt_answer, threshold=0.85):
    """
    Check if the predicted answer semantically matches the ground truth.
    Uses a sentence transformer model to compute similarity.
    """
    pred_embedding = semantic_model.encode(pred_answer, convert_to_tensor=True)
    gt_embedding = semantic_model.encode(gt_answer, convert_to_tensor=True)
    similarity = util.cos_sim(pred_embedding, gt_embedding).item()
    return similarity >= threshold

def evaluate(annotation_file, result_file):
    """Evaluate answers in the result file against the ground truth annotations."""
    print(f"Evaluating: {os.path.basename(result_file)}")

    # Load annotation (ground truth) and result files
    with open(annotation_file, 'r') as f:
        annotations = [json.loads(line) for line in f]
    with open(result_file, 'r') as f:
        results = [json.loads(line) for line in f]

    # Ensure matching IDs
    if len(annotations) != len(results):
        raise ValueError("Mismatch in number of entries between annotation and result files.")

    total = len(annotations)
    correct = 0
    mismatched_ids = []

    for annotation, result in zip(annotations, results):
        if annotation['id'] != result['id']:
            mismatched_ids.append((annotation['id'], result['id']))
            continue

        # Extract answers
        question = annotation['conversations'][0]['value']  # From annotation
        gt_answer = annotation['conversations'][1]['value']  # Ground truth answer
        pred_answer = result['text']  # Model's predicted answer

        # Compare answers using both token-based and semantic matching
        if tokens_match(pred_answer, gt_answer) or semantic_match(pred_answer, gt_answer):
            correct += 1

    accuracy = 100.0 * correct / total
    print(f"Samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

    if mismatched_ids:
        print("\nWarning: The following IDs do not match between the files:")
        for gt_id, pred_id in mismatched_ids:
            print(f"Ground Truth ID: {gt_id}, Predicted ID: {pred_id}")

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate short answers in JSONL files.")
    parser.add_argument("--annotation-file", type=str, required=True, help="Path to ground truth JSONL file.")
    parser.add_argument("--result-file", type=str, required=True, help="Path to model-generated answers JSONL file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    evaluate(args.annotation_file, args.result_file)
