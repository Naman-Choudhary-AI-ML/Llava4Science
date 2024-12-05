#!/bin/bash

deepspeed /user_data/amulyam/Projects/LLaVA/FineTune/train/train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /user_data/amulyam/Projects/LLaVA/FineTune/scripts/zero3_offload.json \
    --model_name_or_path microsoft/llava-med-v1.5-mistral-7b \
    --version v1 \
    --data_path /user_data/amulyam/Projects/EHRXQA/ehrxqa/new_data/modified_train_file.json \
    --image_folder /user_data/amulyam/Projects/EHRXQA/ehrxqa/study_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --fp16 True \
    --output_dir /user_data/amulyam/Projects/LLaVA/chesttune_mistral/llava-v1.5-7b \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 512 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb 

# /user_data/amulyam/Projects/train_copy_missing_bin_file__sr.sh