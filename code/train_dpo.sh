#!/bin/bash
python train_dpo/train_dpo.py \
  --hf_user "thdsofia" \
  --dataset_name "thdsofia/DPO_STEM_training"
  --source_dataset "argilla/ultrafeedback-binarized-preferences" \
  --seed 42 \
  --base_model "thdsofia/general_sft" \
  --output_dir "./dpo-output" \
  --batch_size 2 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --bf16 \