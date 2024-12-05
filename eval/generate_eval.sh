#!/bin/bash

# python /user_data/amulyam/Projects/LLaVA/FineTune/convert_to_jsonl.py \
#     --input-file /user_data/amulyam/Projects/SciGraphQA/sci_train_data_20.json \
#     --output-file /user_data/amulyam/Projects/SciGraphQA/sci_train_data_20.jsonl


python /user_data/amulyam/Projects/LLaVA/FineTune/llava/eval/model_vqa.py \
    --model-path /user_data/amulyam/Projects/LLaVA/scienceGeotune/llava-merged_model \
    --question-file /user_data/amulyam/Projects/SciCOT/GeoQA_valid.jsonl \
    --image-folder /user_data/amulyam/Projects/combined_dataset/combined_images \
    --answers-file /user_data/amulyam/Projects/LLaVA/scienceGraphtune/results/GeoQA_valid.jsonl_ans.jsonl \
    --conv-mode llava_v1

python -c 'print("Finished generating answers")'


# # Convert model answers to JSONL
# python /user_data/amulyam/Projects/LLaVA/FineTune/convert_to_jsonl.py \
#     --input-file /user_data/amulyam/Projects/LLaVA/chesttune_mistral/results/valid10_llava-mistral-7b.json \
#     --output-file /user_data/amulyam/Projects/LLaVA/chesttune_mistral/results/valid10_llava-mistral-7b.jsonl



export OPENAI_API_KEY=""

OPENAI_API_KEY="" python /user_data/amulyam/Projects/LLaVA/FineTune/scripts/multimodal_chatgpt_eval.py \
    --ground-truth-file /user_data/amulyam/Projects/SciCOT/GeoQA_valid.jsonl \
    --model-file  /user_data/amulyam/Projects/LLaVA/scienceGraphtune/results/GeoQA_valid.jsonl_ans.jsonl \
    --output-file /user_data/amulyam/Projects/LLaVA/scienceGraphtune/results/evaluation_results.jsonl


python -c 'print("Finished evaluating answers")'



# python /user_data/amulyam/Projects/LLaVA/FineTune/scripts/short_answers_eval.py \
#     --annotation-file /user_data/amulyam/Projects/EHRXQA/ehrxqa/new_data/modified_valid_file.jsonl \
#     --result-file /user_data/amulyam/Projects/LLaVA/chesttune_llavamed13b/results/ans13bllavamed.jsonl


# python /user_data/amulyam/Projects/LLaVA/FineTune/llava/eval/summarize_gpt_review2.py \
#     --scores-file /user_data/amulyam/Projects/LLaVA/chesttune_10/results/evaluation_results.jsonl






