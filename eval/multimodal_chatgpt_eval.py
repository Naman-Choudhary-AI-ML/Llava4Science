import argparse
import json
import time
from tqdm import tqdm

NUM_SECONDS_TO_SLEEP = 0.5



def get_eval(content: str, max_tokens: int):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model='gpt-4o',
                messages=[{
                    'role': 'system',
                    'content': 'You are a helpful and precise assistant for checking the quality of the answer.'
                }, {
                    'role': 'user',
                    'content': content,
                }],
                temperature=0.2,  # TODO: figure out which temperature is best for evaluation
                max_tokens=max_tokens,
            )
            break
        except openai.error.RateLimitError:
            pass
        except Exception as e:
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    return response['choices'][0]['message']['content']


def infer(samples, max_tokens=512):
    """
    Perform evaluation by comparing model and ground truth answers.
    """
    results = []

    print('Starting Evaluation...')
    for sample in tqdm(samples):
        # Generate input for GPT-4 evaluation
        content = (f"[Question]\n{sample['question']}\n\n"
                   f"[Ground Truth]\n{sample['ground_truth']}\n\n"
                   f"[Model Answer]\n{sample['model_answer']}\n\n"
                   f"[System]\n"
                   f"Please compare the provided answers based on their helpfulness, relevance, accuracy, and detail. "
                   f"Rate each response on a scale of 1 to 10, and explain your evaluation in detail.")

        # Query GPT-4 for evaluation
        evaluation_result = get_eval(content, max_tokens=max_tokens)

        # Add evaluation result to sample
        sample['evaluation'] = evaluation_result.strip()
        results.append(sample)

    return results


def main(args):
    # Load ground truth and model-generated answers
    ground_truth_data = [json.loads(line) for line in open(args.ground_truth_file)]
    model_data = [json.loads(line) for line in open(args.model_file)]

    # Prepare data for evaluation
    samples = []
    for gt, model in zip(ground_truth_data, model_data):
        if gt['id'] != model['id']:
            raise ValueError(f"ID mismatch: ground truth ID {gt['id']} does not match model ID {model['id']}.")

        # Extract fields for comparison
        question = gt['conversations'][0]['value']  # From ground truth
        ground_truth_answer = gt['conversations'][1]['value']  # Ground truth answer
        model_answer = model['text']  # Model's generated answer

        # Add to evaluation samples
        samples.append({
            "id": gt['id'],
            "question": question,
            "ground_truth": ground_truth_answer,
            "model_answer": model_answer
        })

    # Perform evaluation
    results = infer(samples)

    # Save results to output file
    with open(args.output_file, 'w') as f:
        for row in results:
            f.write(json.dumps(row) + '\n')

    print(f"Evaluation completed. Results saved to {args.output_file}.")


if __name__ == '__main__':
    import openai
    import os

    # Ensure OpenAI API key is set in the environment
    openai.api_key = os.getenv("OPENAI_API_KEY")

    parser = argparse.ArgumentParser(description='Evaluate model answers against ground truth using GPT-4.')
    parser.add_argument('--ground-truth-file', required=True, help='Path to ground truth JSONL file.')
    parser.add_argument('--model-file', required=True, help='Path to model-generated answers JSONL file.')
    parser.add_argument('--output-file', required=True, help='Path to save evaluation results.')
    args = parser.parse_args()

    main(args)
