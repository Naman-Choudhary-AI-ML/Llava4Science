import argparse
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
from nltk.translate.meteor_score import meteor_score
import torch
from transformers import BertTokenizer, BertModel

# Load a SentenceTransformer model for semantic similarity
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize BERT components for BERTScore
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

def calculate_metrics(ground_truth, model_answer):
    """
    Calculate evaluation metrics: BLEU, ROUGE, METEOR, F1, Cosine Similarity, BERTScore.
    """
    metrics = {}

    # BLEU Score
    bleu_score = sentence_bleu([ground_truth.split()], model_answer.split())
    metrics['bleu'] = bleu_score

    # ROUGE Scores
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge.score(ground_truth, model_answer)
    metrics['rouge1'] = rouge_scores['rouge1'].fmeasure
    metrics['rouge2'] = rouge_scores['rouge2'].fmeasure
    metrics['rougeL'] = rouge_scores['rougeL'].fmeasure

    # METEOR Score
    meteor = meteor_score([ground_truth], model_answer)
    metrics['meteor'] = meteor

    # F1 Score (if ground truth and model answer are classes/labels, e.g., "A", "B", "C")
    if len(ground_truth.split()) == 1 and len(model_answer.split()) == 1:  # Classifications
        f1 = f1_score([ground_truth], [model_answer], average='micro')
        metrics['f1'] = f1
    else:
        metrics['f1'] = None  # Not applicable for long-text answers

    # Cosine Similarity
    gt_embedding = embedder.encode(ground_truth, convert_to_tensor=True)
    model_embedding = embedder.encode(model_answer, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(gt_embedding, model_embedding).item()
    metrics['cosine_similarity'] = cosine_sim

    # BERTScore
    def compute_bert_score(gt, pred):
        with torch.no_grad():
            gt_tokens = bert_tokenizer(gt, return_tensors='pt', truncation=True, max_length=512)
            pred_tokens = bert_tokenizer(pred, return_tensors='pt', truncation=True, max_length=512)

            gt_embedding = bert_model(**gt_tokens).pooler_output
            pred_embedding = bert_model(**pred_tokens).pooler_output

            similarity = torch.nn.functional.cosine_similarity(gt_embedding, pred_embedding)
            return similarity.item()

    bert_score = compute_bert_score(ground_truth, model_answer)
    metrics['bert_score'] = bert_score

    return metrics

def evaluate(samples):
    """
    Evaluate the dataset using the specified metrics.
    """
    results = []

    for sample in tqdm(samples):
        ground_truth = sample['ground_truth']
        model_answer = sample['model_answer']

        # Calculate metrics
        metrics = calculate_metrics(ground_truth, model_answer)
        sample['metrics'] = metrics

        results.append(sample)

    return results

def main(args):
    # Load data
    with open(args.ground_truth_file) as f:
        ground_truth_data = [json.loads(line) for line in f]

    with open(args.model_file) as f:
        model_data = [json.loads(line) for line in f]

    # Prepare samples
    samples = []
    for gt, model in zip(ground_truth_data, model_data):
        if gt['id'] != model['id']:
            raise ValueError(f"ID mismatch: ground truth ID {gt['id']} does not match model ID {model['id']}.")

        question = gt['conversations'][0]['value']
        ground_truth_answer = gt['conversations'][1]['value']
        model_answer = model['text']

        samples.append({
            "id": gt['id'],
            "question": question,
            "ground_truth": ground_truth_answer,
            "model_answer": model_answer
        })

    # Evaluate samples
    results = evaluate(samples)

    # Save results
    with open(args.output_file, 'w') as f:
        for row in results:
            f.write(json.dumps(row) + '\n')

    print(f"Evaluation completed. Results saved to {args.output_file}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate model answers against ground truth using various metrics.')
    parser.add_argument('--ground-truth-file', required=True, help='Path to ground truth JSONL file.')
    parser.add_argument('--model-file', required=True, help='Path to model-generated answers JSONL file.')
    parser.add_argument('--output-file', required=True, help='Path to save evaluation results.')
    args = parser.parse_args()

    main(args)
