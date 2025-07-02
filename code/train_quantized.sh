#!/bin/bash
HF_TOKEN="" # to set for dataset pushing and model pushing
HF_USER="mkartofel" # to set for dataset pushing and model pushing
MISTRAL_API_KEY="" # to set for dataset building

pip install -r train_quantized/quantized_requirements.txt

python train_quantized/train_quantized.py \
  --hf_user "$HF_USER" \
  --hf_token "$HF_TOKEN" \
  --dataset_name "mkartofel/MNLP_M3_quantized_dataset" \
  --base_model "Qwen/Qwen3-0.6B-Base" \
  --output_dir "./Qwen3-0.6B-qlora-MCQA_lora_final_30k" \
  --sample_size 30000 \
  --mistral_api_key "$MISTRAL_API_KEY" \
