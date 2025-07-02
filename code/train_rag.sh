#!/bin/bash

# === Hugging Face Credentials ===
HF_TOKEN="" # to set for dataset pushing and model pushing
HF_USER="NFX74"

# === Mistral API Key ===
MISTRAL_API_KEY="" # to set for dataset building

# === Retrieval & Chunking Parameters ===
K=3
CHUNK_SIZE=300          # Number of tokens per chunk
CHUNK_OVERLAP=0         # Number of overlapping tokens between chunks
SIMILARITY_FN="cosine"  # Options: cosine, dot_product, max_inner_product, jaccard

# === Paths & Dataset Names ===
PDF_FOLDER="train_rag/STEM_books"
RAG_CORPUS_NAME="${HF_USER}/rag_corpus_stem_books_chunked_${CHUNK_SIZE}"
RAG_DATASET_NAME="NFX74/MNLP_M2_rag_dataset"

# === Models ===
BASE_MODEL_NAME="NFX74/Qwen3-0.6B-Base-SFT-STEM"
EMBEDDING_MODEL_NAME="thenlper/gte-small"
RAG_MODEL_NAME="${HF_USER}/Qwen3-0.6B-Base-SFT-STEM-LoRA-SciQ-RAG"

# === Evaluation Settings ===
EVAL_DATASETS=("zechen-nlp/MNLP_STEM_mcqa_demo" "LiSoViMa/MNLP_STEM_mcqa_evals" "LiSoViMa/mmlu" "LiSoViMa/SCiQ_formatted" "LiSoViMa/AquaRat" "LiSoViMa/ai2_arc" "LinaSad/firstm_test_set" "LinaSad/Synth_mistral")
FIRST_N=0              # Evaluate on first n examples (0 for all)




# === Install dependencies ===
pip install -r train_rag/requirements.txt

# === Build RAG Corpus ===
python train_rag/build_rag_corpus.py \
  --hf_token "$HF_TOKEN" \
  --rag_corpus_name "$RAG_CORPUS_NAME" \
  --pdf_folder "$PDF_FOLDER" \
  --mistral_api_key "$MISTRAL_API_KEY" \
  --base_model "$BASE_MODEL_NAME" \
  --chunk_size "$CHUNK_SIZE" \
  --chunk_overlap "$CHUNK_OVERLAP" \

# === Run RAG fine tuning ===
python -W ignore train_rag/train_rag_lora.py \
  --hf_token "$HF_TOKEN" \
  --rag_corpus_name "$RAG_CORPUS_NAME" \
  --rag_dataset_name "$RAG_DATASET_NAME" \
  --base_model_name "$BASE_MODEL_NAME" \
  --embedding_model_name "$EMBEDDING_MODEL_NAME" \
  --merged_hub_repo "$RAG_MODEL_NAME" \
  --k "$K" \
  --chunk_size "$CHUNK_SIZE" \
  --chunk_overlap "$CHUNK_OVERLAP" \
  --similarity_fn "$SIMILARITY_FN" \
  --train_only  # If set, only trains and saves the merged model locally without pushing to Hugging Face Hub

# === Run RAG Evaluation ===
python train_rag/evaluate_rag.py \
  --base_model_name "$RAG_MODEL_NAME" \
  --embedding_model_name "$EMBEDDING_MODEL_NAME" \
  --rag_corpus_name "$RAG_CORPUS_NAME" \
  --k "$K" \
  --chunk_size "$CHUNK_SIZE" \
  --chunk_overlap "$CHUNK_OVERLAP" \
  --similarity_fn "$SIMILARITY_FN" \
  --eval_dataset_names "${EVAL_DATASETS[@]}" \
  --first_n "$FIRST_N"