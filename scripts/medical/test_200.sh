#!/bin/bash

# Test training with 200 examples

# Set PYTHONPATH to use local llava module instead of site-packages
export PYTHONPATH=/gpfs/scratch/gs4342/LLaVA:$PYTHONPATH

python llava/train/train_mem.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --vision_tower medical_vit \
    --vit_checkpoint_path /gpfs/data/razavianlab/hh2740/FM_model_weights/FM_CT_pretrained_clear.pth \
    --use_all_vit_tokens True \
    --mm_projector_type mlp2x_gelu \
    --mm_hidden_size 768 \
    --tune_mm_mlp_adapter True \
    --freeze_vision_tower True \
    --freeze_backbone True \
    --data_path ./data/medical_stage1_200.json \
    --image_folder ./ \
    --fp16 True \
    --output_dir ./checkpoints/test-200 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_strategy epoch \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to none
