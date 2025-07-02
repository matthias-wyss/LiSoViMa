from typing import List, Tuple
import os
from datasets import load_dataset
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from lighteval.tasks.default_prompts import LETTER_INDICES
import itertools
import json
from peft import PeftModel
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


def tokenize_and_chunk_dataset(
    rag_corpus_name: str,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    chunk_overlap: int,
    split: str = "train"
) -> List[Document]:
    """
    Load a HuggingFace dataset and split each example into token-based chunks.

    Args:
        rag_corpus_name: Name of the HuggingFace dataset.
        tokenizer: HuggingFace tokenizer.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        split: Dataset split to load (default is "train").

    Returns:
        A list of LangChain Document objects, each representing a text chunk with metadata.
    """
    ds = load_dataset(rag_corpus_name, split=split)

    def chunk_text(text: str) -> List[str]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        step = chunk_size - chunk_overlap
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)
        return chunks

    documents = []
    for example in tqdm(ds, desc=f"Chunking dataset '{rag_corpus_name}'"):
        chunks = chunk_text(example["text"])
        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"source": example.get("source", ""), "chunk_id": idx}
            ))

    print(f"[✓] {len(ds)} documents -> {len(documents)} token chunks generated.")
    return documents


def load_model_and_tokenizer(
    base_model_name: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a causal language model and its tokenizer from HuggingFace.

    Args:
        base_model_name: Model identifier or local path.

    Returns:
        A tuple of (model, tokenizer) ready for inference.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode (disable dropout, gradients)
    return model, tokenizer


def compute_log_likelihood_per_choice(
    prompt: str,
    choices: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer
) -> List[float]:
    """
    Compute log-likelihood scores for each answer choice given a prompt.

    Args:
        prompt: The prompt containing question and context.
        choices: List of answer choice strings (e.g. [" A", " B", " C", " D"]).
        model: Causal language model.
        tokenizer: Corresponding tokenizer.

    Returns:
        A list of log-likelihood scores, one per choice.
    """
    scores = []
    for choice in choices:
        full_input = prompt + choice
        input_ids = tokenizer(full_input, return_tensors="pt").input_ids.to(model.device)
        target_len = tokenizer(choice, return_tensors="pt").input_ids.shape[1]

        # Mask all tokens except the choice tokens to compute conditional likelihood of the choice given prompt
        labels = input_ids.clone()
        labels[:, :-target_len] = -100  # Ignore loss for prompt tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            # Negative loss multiplied by number of tokens gives total log-likelihood
            log_likelihood = -outputs.loss.item() * target_len
            scores.append(log_likelihood)
    return scores


def evaluate_example_rag(
    example: dict,
    db: FAISS,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    k: int = 5
) -> bool:
    """
    Evaluate one multiple-choice QA example using Retrieval-Augmented Generation (RAG) with log-likelihood scoring.

    Args:
        example: Dictionary with keys "question", "choices" (list), and "answer" (correct letter).
        db: FAISS vector store for retrieval.
        model: Causal language model.
        tokenizer: Corresponding tokenizer.
        k: Number of top retrieved documents to use.

    Returns:
        True if predicted answer matches the gold label, else False.
    """
    topic = "knowledge and skills in advanced master-level STEM courses"

    # Construct retrieval query with question and multiple choices
    query = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    query += example["question"] + "\n"
    query += "".join([f"{letter}. {choice}\n" for letter, choice in zip(LETTER_INDICES, example["choices"])])
    query += "Answer:"

    # Retrieve top-k relevant documents
    retrieved_docs = db.similarity_search(query=query, k=k)
    context = "\nRelevant Documents:\n" + "\n\n".join([
        f"Document {i}:::\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
    ])

    # Full prompt = retrieved context + question + choices
    prompt = context + "\n\n" + query

    # Score each choice by computing its conditional log-likelihood
    choices = [" A", " B", " C", " D"]
    log_likelihoods = compute_log_likelihood_per_choice(prompt, choices, model, tokenizer)

    # Pick the answer with highest log-likelihood
    predicted_idx = torch.tensor(log_likelihoods).argmax().item()
    predicted_letter = LETTER_INDICES[predicted_idx]

    return predicted_letter == example["answer"]


def get_faiss_folder_name(
    rag_corpus_name: str,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    similarity_fn: str = "cosine"
) -> str:
    """
    Construct the folder name for the FAISS index based on model and dataset parameters.
    """
    emb = embedding_model_name.split("/")[-1].replace(".", "")
    corpus = rag_corpus_name.split("/")[-1].replace(".", "")
    folder_name = f"train_rag/FAISS/corpus_{corpus}_emb_{emb}_chunks{chunk_size}_overlap{chunk_overlap}_{similarity_fn}"
    return folder_name


def get_lora_folder_name(
    base_model_name: str,
    rag_corpus_name: str,
    embedding_model_name: str,
    k: int,
    chunk_size: int,
    chunk_overlap: int,
    similarity_fn: str = "cosine"
) -> str:
    """
    Construct the folder name for LoRA based on model and dataset parameters.
    """
    base = base_model_name.split("/")[-1].replace(".", "")
    emb = embedding_model_name.split("/")[-1].replace(".", "")
    corpus = rag_corpus_name.split("/")[-1].replace(".", "")
    folder_name = f"train_rag/LoRA/model_{base}_corpus_{corpus}_emb_{emb}_k{k}_chunks{chunk_size}_overlap{chunk_overlap}_{similarity_fn}"
    return folder_name


def get_eval_folder_name(
    base_model_name: str,
    rag_corpus_name: str,
    embedding_model_name: str,
    k: int,
    chunk_size: int,
    chunk_overlap: int,
    similarity_fn: str = "cosine"
) -> str:
    """
    Construct the folder name for the results based on model and dataset parameters.
    """
    base = base_model_name.split("/")[-1].replace(".", "")
    emb = embedding_model_name.split("/")[-1].replace(".", "")
    corpus = rag_corpus_name.split("/")[-1].replace(".", "")
    folder_name = f"train_rag/results/model_{base}_corpus_{corpus}_emb_{emb}_k{k}_chunks{chunk_size}_overlap{chunk_overlap}_{similarity_fn}"
    return folder_name


def build_or_load_faiss_db(
    rag_corpus_name: str,
    tokenizer: AutoTokenizer,
    embedding_model,
    base_model_name: str,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    similarity_fn: str = "cosine"
) -> FAISS:
    """
    Build a FAISS index from a dataset or load it if already saved locally.

    Args:
        rag_corpus_name: HuggingFace dataset identifier.
        tokenizer: Tokenizer used for chunking.
        embedding_model: LangChain-compatible embedding model.
        base_model_name: Language model name (for naming convention).
        embedding_model_name: Embedding model name (for naming convention).
        chunk_size: Number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between chunks.
        similarity_fn: Similarity function to use, options are:
                       "cosine", "dot_product", "max_inner_product", "jaccard".

    Returns:
        Loaded or newly created FAISS vector store.
    """

    # Map string to DistanceStrategy enum
    similarity_fn_map = {
        "cosine": DistanceStrategy.COSINE,
        "dot_product": DistanceStrategy.DOT_PRODUCT,
        "max_inner_product": DistanceStrategy.MAX_INNER_PRODUCT,
        "jaccard": DistanceStrategy.JACCARD,
    }

    if similarity_fn not in similarity_fn_map:
        raise ValueError(f"Invalid similarity function '{similarity_fn}'. Must be one of {list(similarity_fn_map.keys())}")

    folder_name = get_faiss_folder_name(
        rag_corpus_name,
        embedding_model_name,
        chunk_size,
        chunk_overlap,
        similarity_fn,
    )

    distance_strategy = similarity_fn_map[similarity_fn]

    if os.path.exists(folder_name):
        print(f"✅ FAISS DB already exists at '{folder_name}/'. Loading...")
        db = FAISS.load_local(
            folder_name,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
            distance_strategy=distance_strategy,
            normalize_L2=True if similarity_fn == "cosine" else False,
        )
    else:
        print(f"[•] FAISS DB not found at '{folder_name}/'. Building new index...")
        docs = tokenize_and_chunk_dataset(
            rag_corpus_name=rag_corpus_name,
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        db = FAISS.from_documents(docs, embedding_model)
        db.save_local(folder_name)
        print(f"✅ FAISS DB saved to '{folder_name}/'")

    return db


def evaluate_and_save_all_datasets(
    eval_dataset_names,
    db,
    model,
    tokenizer,
    k,
    folder_name,
    first_n=0  # If 0, evaluate all examples; else only first_n examples per dataset
):
    os.makedirs(folder_name, exist_ok=True)
    base_filename = f"k_{k}" + ("" if first_n == 0 else f"_first_{first_n}")
    txt_save_path = os.path.join(folder_name, f"{base_filename}_evaluations.txt")
    json_save_path = os.path.join(folder_name, f"{base_filename}_evaluations_summary.json")


    accuracy_dict = {}

    with open(txt_save_path, "w", encoding="utf-8") as f:
        for eval_dataset_name in eval_dataset_names:
            eval_dataset = load_dataset(eval_dataset_name)["test"]
            correct, total = 0, 0
            f.write(f"Evaluation on dataset: {eval_dataset_name}\n")
            
            # Limit dataset size if first_n > 0
            dataset_iterator = (
                eval_dataset if first_n == 0 else itertools.islice(eval_dataset, first_n)
            )
            
            for example in tqdm(dataset_iterator, desc=f"Evaluating '{eval_dataset_name}'"):
                is_correct = evaluate_example_rag(example, db, model, tokenizer, k=k)
                if is_correct:
                    correct += 1
                total += 1
                f.write(f"Example {total}: {'Correct' if is_correct else 'Incorrect'}\n")

            accuracy = correct / total if total > 0 else 0.0
            accuracy_dict[eval_dataset_name] = accuracy
            f.write(f"Accuracy: {accuracy:.3%} ({correct}/{total})\n\n")
            print(f"\n✅ Accuracy on '{eval_dataset_name}': {accuracy:.3%} ({correct}/{total})")

    # Save accuracy summary as JSON
    with open(json_save_path, "w", encoding="utf-8") as jf:
        json.dump(accuracy_dict, jf, indent=4)
    
    print(f"\nAll evaluation results saved to: {txt_save_path}")
    print(f"Summary JSON saved to: {json_save_path}")






def prepare_model_for_lora(model):
    """Freeze all model weights and enable gradient checkpointing."""
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()


def cast_lm_head_to_float(lm_head):
    """Ensure output head is in float32 for stability."""
    class CastOutputToFloat(torch.nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)
    return CastOutputToFloat(lm_head)


def format_rag_mcqa(example, db, k=5):
    """Construct a RAG-formatted prompt with retrieved documents for MCQA."""
    topic = "knowledge and skills in advanced master-level STEM courses"
    query = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    query += example["question"] + "\n"
    for key, choice in zip(['A', 'B', 'C', 'D'], example["choices"]):
        query += f"{key}. {choice}\n"
    query += "Answer:"

    retrieved_docs = db.similarity_search(query=query, k=k)
    context = "\nRelevant Documents:\n" + "\n\n".join(
        [f"Document {i}:::\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)]
    )

    prompt = context + "\n\n" + query
    response = " " + example["answer"] + ". " + example.get("support", "")
    return {"prompt": prompt, "completion": response}



def push_lora_model(base_model, lora_model_dir, merged_repo, hf_token, train_only=False, save_path=None):
    """
    Merge a base model with its LoRA adapter, save it locally, and optionally push it to the Hugging Face Hub.

    Args:
        base_model (str): Hugging Face name or path of the base model.
        lora_model_dir (str): Path to the directory containing LoRA adapter weights.
        merged_repo (str): Hugging Face repo name for the merged model.
        hf_token (str): Hugging Face API token.
        train_only (bool): If True, only save the merged model locally (do not push).
        save_path (str): Path to save the merged model locally. Defaults to lora_model_dir/merged.
    """
    login(token=hf_token)

    # Load and merge base model
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load and merge LoRA
    model = PeftModel.from_pretrained(base, lora_model_dir)
    model = model.merge_and_unload()

    # Saving merged model locally
    save_dir = save_path or os.path.join(lora_model_dir, "merged")
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Merged model saved locally at {save_dir}")

    # Pushing merged model to Hugging Face
    if not train_only:
        model.push_to_hub(merged_repo, safe_serialization=True)
        tokenizer.push_to_hub(merged_repo)
        print(f"✅ Merged model pushed to {merged_repo}")
