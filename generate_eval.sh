#!/bin/bash

# convert json to jsonl
python /user_data/amulyam/Projects/LLAVA-FineTune/convert_to_jsonl.py \
    --input-file /user_data/amulyam/Projects/combined_dataset/SciCOT/CLEVR_Math_valid_50.json \
    --output-file /user_data/amulyam/Projects/combined_dataset/SciCOT/CLEVR_Math_valid_50.jsonl


# generate answers from fine-tuned model
CUDA_VISIBLE_DEVICES=8,9 python eval/model_vqa.py \
    --model-path /user_data/amulyam/Projects/LLAVA-FineTune/finetuned_models/science_data/scienceGraphtune10/llava-merged_model \
    --question-file /user_data/amulyam/Projects/combined_dataset/SciCOT/CLEVR_Math_valid_50.jsonl \
    --image-folder /user_data/amulyam/Projects/combined_dataset/combined_images \
    --answers-file finetuned_models/science_data/scienceCLEVRtune10/results/CLEVR_Math_valid_50_ans.jsonl \
    --conv-mode llava_v1

python -c 'print("Finished generating answers")'



# Evaluate answers using GPT-4, OPEN API KEY
export OPENAI_API_KEY=""

OPENAI_API_KEY="" python eval/multimodal_chatgpt_eval.py \
    --ground-truth-file /user_data/amulyam/Projects/combined_dataset/SciCOT/CLEVR_Math_valid_50.jsonl \
    --model-file finetuned_models/science_data/scienceCLEVRtune10/results/CLEVR_Math_valid_50_ans.jsonl \
    --output-file finetuned_models/science_data/scienceCLEVRtune10/results/CLEVR_Math_evaluation_results.jsonl

python -c 'print("Finished evaluating answers")'


# Evaluating answers using scores
python eval/eval_more_metrics.py \
    --ground-truth-file /user_data/amulyam/Projects/combined_dataset/SciCOT/CLEVR_Math_valid_50.jsonl \
    --model-file finetuned_models/science_data/scienceCLEVRtune10/results/CLEVR_Math_valid_50_ans.jsonl \
    --output-file finetuned_models/science_data/scienceCLEVRtune10/results/CLEVR_Math_evaluation_results2.jsonl


python -c 'print("Done with second evaluation")'



# python /user_data/amulyam/Projects/LLaVA/FineTune/scripts/short_answers_eval.py \
#     --annotation-file /user_data/amulyam/Projects/EHRXQA/ehrxqa/new_data/modified_valid_file.jsonl \
#     --result-file /user_data/amulyam/Projects/LLaVA/chesttune_llavamed13b/results/ans13bllavamed.jsonl


# python /user_data/amulyam/Projects/LLaVA/FineTune/llava/eval/summarize_gpt_review2.py \
#     --scores-file /user_data/amulyam/Projects/LLaVA/chesttune_10/results/evaluation_results.jsonl






