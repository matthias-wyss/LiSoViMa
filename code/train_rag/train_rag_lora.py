import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from langchain_huggingface import HuggingFaceEmbeddings
from helper import (
    prepare_model_for_lora,
    cast_lm_head_to_float,
    build_or_load_faiss_db,
    format_rag_mcqa,
    push_lora_model,
    get_lora_folder_name,
)

def main(args):

    # LoRA configuration (could also be parametrized if needed)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "gate_proj", "down_proj"
        ],
    )

    folder_name = get_lora_folder_name(
        base_model_name=args.base_model_name,
        rag_corpus_name=args.rag_corpus_name,
        embedding_model_name=args.embedding_model_name,
        k=args.k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        similarity_fn=args.similarity_fn
    )

    # Training configuration
    training_args = SFTConfig(
        output_dir=folder_name,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        save_strategy="steps",
        logging_dir="./LoRA/logs",
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        eval_steps=100,
        save_safetensors=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        completion_only_loss=True,
    )

    # Load dataset
    raw_dataset = load_dataset(args.rag_dataset_name)
    train_dataset = raw_dataset["train"]
    val_dataset = raw_dataset["validation"]

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model_name, trust_remote_code=True)

    # Prepare model for LoRA
    prepare_model_for_lora(model)
    model.lm_head = cast_lm_head_to_float(model.lm_head)
    model = get_peft_model(model, lora_config)

    # Build FAISS retriever
    retrieval_tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, trust_remote_code=True)
    embedding_model = HuggingFaceEmbeddings(model_name=args.embedding_model_name)

    db = build_or_load_faiss_db(
        rag_corpus_name=args.rag_corpus_name,
        tokenizer=retrieval_tokenizer,
        embedding_model=embedding_model,
        base_model_name=args.base_model_name,
        embedding_model_name=args.embedding_model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        similarity_fn=args.similarity_fn,
    )

    # Format datasets for RAG
    train_dataset = train_dataset.map(lambda x: format_rag_mcqa(x, db, args.k))
    val_dataset = val_dataset.map(lambda x: format_rag_mcqa(x, db, args.k))

    # Train model
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )
    trainer.train()

    # Save LoRA adapters
    model.save_pretrained(folder_name)

    # Push LoRA model to hub
    push_lora_model(
        base_model=args.base_model_name,
        lora_model_dir=folder_name,
        merged_repo=args.merged_hub_repo,
        hf_token=args.hf_token,
        train_only=args.train_only,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAG model with LoRA adapters")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--rag_corpus_name", type=str, required=True, help="Name of the RAG corpus dataset")
    parser.add_argument("--rag_dataset_name", type=str, required=True, help="Name of the RAG training dataset")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base LM model name")
    parser.add_argument("--embedding_model_name", type=str, required=True, help="Embedding model name")
    parser.add_argument("--merged_hub_repo", type=str, required=True, help="Hugging Face repo for merged model")
    parser.add_argument("--train_only", action="store_true", help="Only train and save merged model locally, do not push to hub")
    parser.add_argument("--k", type=int, default=5, help="Number of chunks for retrieval")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=0, help="Chunk overlap for text splitting")
    parser.add_argument("--similarity_fn", type=str, default="cosine", help="Similarity function for retrieval")
    
    args = parser.parse_args()
    main(args)
