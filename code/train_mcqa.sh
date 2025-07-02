#!/bin/bash

pip install -r train_mcqa/mcqa_requirements.txt

python train_mcqa/train_mcqa.py \
  --hf_user "LinaSad" \
  --dataset_name "LinaSad/MNLP_M3_mcqa_dataset" \
  --seed 42 \
  --base_model "Qwen/Qwen3-0.6B-Base" \
  --output_dir "./qwen-general-lora" \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --bf16
