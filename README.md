# MNLP M3 — Final Project Submission

This repository contains the complete pipeline, codebase, and configuration files to reproduce and evaluate four fine-tuned language models developed during the Modern Natural Language Processing course project (M3):  
- **MCQA model**
- **Quantized model**
- **Retrieval-Augmented Generation (RAG) model**
- **Direct Preference Optimization (DPO) model**

We also include a pipeline for Supervised Fine-Tuning (SFT) on STEM data, which serves as a base for some of the above models.

---

## Repository Structure

```
.
├── code/
│   ├── train_mcqa/               # Code to fine-tune the MCQA model
│   ├── train_quantized/          # Code for quantization-based fine-tuning
│   ├── train_rag/                # Code for training the RAG model
│   ├── train_dpo/                # Code to train the DPO model
│   ├── train_sft/                # Extra: pipeline for supervised fine-tuning on STEM
│   ├── train_mcqa.sh             # Training script for MCQA model
│   ├── train_quantized.sh        # Training script for Quantized model
│   ├── train_rag.sh              # Training script for RAG model
│   ├── train_dpo.sh              # Training script for DPO model
│   └── train_sft.sh           # Optional script to run the SFT pipeline
│
├── model_configs/
│   ├── mcqa_model.yaml
│   ├── quantized_model.yaml
│   ├── rag_model.yaml
│   └── dpo_model.yaml
│
├── data/
│   └── data_repo.json            # Pointers to training datasets on Hugging Face Hub
│
├── pdf/
│   └── LiSoViMa.pdf          # Final project report
│
├── _templates/
│   └── mnlp_report_template.tex  # Report template used for the final project report
│
└── _test/
    └── run_tests.py              # Scripts to validate submission format
```

---

## Training Scripts

Each model has its own training script that reproduces the fine-tuning process:

```bash
bash train_mcqa.sh
bash train_quantized.sh
bash train_rag.sh
bash train_dpo.sh
bash train_sft.sh
```

These scripts will:
- Construct the appropriate datasets
- Train the model using your specified configuration
- Save the final and best checkpoints locally

You should be able to reproduce the submission by running each script independently.

### `train_sft.sh`

This optional script trains a base language model on STEM-related multiple-choice QA data using a Supervised Fine-Tuning (SFT) approach. It is useful for pretraining a foundational model before applying more advanced techniques like MCQA, RAG or quantization.

The script performs the following:
 
- Select a subset of num_rows rows from source_dataset
- Pushes the subset dataset on Hugging Face Hub
- Fine-tunes base_model using the indicated arguments on the subset dataset
- Pushes the final checkpoint to the Hugging Face Hub

Usage: 
```bash
bash train_sft.sh
```

### `train_mcqa.sh`
This script fine-tunes the MCQA model on STEM-related multiple-choice QA data using the Low-Rank Adaptation technique. It leverages a pre-trained base model and adapts it for multiple-choice question answering tasks by incorporating LoRA to improve performance with fewer parameters.

The script performs the following steps:

- Installs the necessary dependencies from train_mcqa/requirements.txt.
- Prepares and preprocesses the MCQA dataset, or create and preprocess it if non-existant.
- Loads the base model (Qwen/Qwen3-0.6B-Base or given one) and tokenizer.
- Fine-tunes the base model using LoRA adapters, with specific hyperparameters (learning rate, batch size, gradient accumulation steps).
- Saves the final trained model to the specified output directory.

Usage: 
```bash
bash train_mcqa.sh
```

### `train_quantized.sh`
This script trains a quantized version of the model using 4-bit QLoRA fine-tuning on STEM-related multiple-choice QA data. It builds the dataset automatically if not found, and applies 4-bit quantization with LoRA adapters to optimize for memory efficiency without sacrificing accuracy.

The script performs the following steps:
-	Installs dependencies from train_quantized/quantized_requirements.txt
-	Prepares and preprocesses the MCQA dataset, or create and preprocess it if non-existant
-	Loads the Qwen3-0.6B-Base model in 4-bit (nf4, double quant, bfloat16)
-	Injects LoRA adapters into all linear layers of the quantized model
-	Fine-tunes the model on the MCQA dataset
-	Merges the LoRA adapters back into the base model
-	Saves the final trained model to the specified output directory

Usage: 
```bash
bash train_quantized.sh
```

### `train_rag.sh`

This script fine-tunes a RAG-ready language model on STEM-related multiple-choice QA data using LoRA and a retrieval-augmented generation (RAG) setup. It assumes a prior base model fine-tuned via SFT, and adds retrieval capabilities by aligning the model with an external corpus built from PDF textbooks.

The script performs the following steps:

- Installs dependencies from `train_rag/requirements.txt`
- Applies OCR (via Mistral API) to extract markdown from PDFs
- Splits the markdown content into tokenized chunks using a Hugging Face tokenizer
- Uploads the resulting RAG corpus to the Hugging Face Hub
- Builds a FAISS index on the RAG corpus using the specified embedding model and stores it locally in a uniquely named folder (based on key RAG parameters)
- Loads the base SFT model and applies a LoRA adapter using RAG-formatted MCQA data
- Fine-tunes the model and stores it locally in a uniquely named folder (based on key RAG parameters)
- Merging LoRA + base
- Stores the merged model 
- Stores the FAISS database locally in the same folder, enabling efficient evaluation
- Evaluates the merged model on several MCQA benchmarks

The LoRA model, merged LoRA + base model and the retriever FAISS index are saved in a uniquely named output folder (under `./LoRA/`, `./LoRA/merged/` and `./FAISS/`) based on:

- base model name
- RAG corpus name
- embedding model
- chunking parameters
- similarity function used

This setup ensures full reproducibility and allows evaluating or reusing the model without recomputation.

Usage:  
```bash
bash train_rag.sh
```

### `train_dpo.sh`
This script trains a causal language model using the Direct Preference Optimization (dpo) method. It performs the following steps: 

- Load `/source_dataset/` from Hugging Face Hub.
- Select only `\["prompt", "chosen", "rejected"]`\ columns.
- Split the data into 90% train and 10% validation using seed.
- Load a pretrained base model and tokenizer from Hugging Face.
- Configure DPO training using DPOConfig.
- Initialize the DPO trainer.
- Check if there is any checkpoint saved in the output directory and starts from the latest one. Otherwise, trains from scratch.
- Save final model and tokenizer to `\output_dir`\ named `\dpo_output`\.

Usage: 
```bash
bash train_dpo.sh
```

## Notes on Config Files

The base model used in `model_configs/rag_model.yaml` is the same as the one in `model_configs/mcqa_model.yaml`, as it yielded the best results.  
The embedding model specified in `model_configs/rag_model.yaml` is `thenlper/gte-small`.
