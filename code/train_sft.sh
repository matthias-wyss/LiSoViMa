#!/bin/bash

pip install -r train_sft/requirements.txt

python train_sft/sft_training.py \
  --hf_token "" \ # Set your Hugging Face token here
  --hf_user "NFX74" \
  --dataset_name "NFX74/SFT_STEM_100k" \
  --source_dataset "PrimeIntellect/SYNTHETIC-1-SFT-Data" \
  --num_rows 100000 \
  --seed 42 \
  --base_model "Qwen/Qwen3-0.6B-Base" \
  --output_dir "./train_sft/sft-output" \
  --train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --bf16

python train_sft/push_best_model.py \
  --hf_token "" \ # Set your Hugging Face token here
  --repo_name "NFX74/Qwen3-0.6B-Base-SFT-STEM" \
  --output_dir "./train_sft/sft-output"
