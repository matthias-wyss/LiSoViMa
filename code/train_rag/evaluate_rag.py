import warnings
warnings.filterwarnings("ignore")

import argparse
from datasets import load_dataset
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

from helper import *

def main(args):
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.base_model_name)
    embedding_model = HuggingFaceEmbeddings(model_name=args.embedding_model_name)

    # Build or load FAISS DB
    db = build_or_load_faiss_db(
        rag_corpus_name=args.rag_corpus_name,
        tokenizer=tokenizer,
        embedding_model=embedding_model,
        base_model_name=args.base_model_name,
        embedding_model_name=args.embedding_model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        similarity_fn=args.similarity_fn
    )

    # Prepare evaluation folder name
    folder_name = get_eval_folder_name(
        base_model_name=args.base_model_name,
        rag_corpus_name=args.rag_corpus_name,
        embedding_model_name=args.embedding_model_name,
        k=args.k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        similarity_fn=args.similarity_fn
    )

    # Evaluate
    evaluate_and_save_all_datasets(
        eval_dataset_names=args.eval_dataset_names,
        db=db,
        model=model,
        tokenizer=tokenizer,
        k=args.k,
        folder_name=folder_name,
        first_n=args.first_n
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG model on MCQA datasets")
    parser.add_argument("--base_model_name", type=str, required=True, help="Base LM model name")
    parser.add_argument("--embedding_model_name", type=str, required=True, help="Embedding model name")
    parser.add_argument("--rag_corpus_name", type=str, required=True, help="RAG corpus dataset name")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=0, help="Chunk overlap for text splitting")
    parser.add_argument("--similarity_fn", type=str, default="cosine", choices=["cosine", "dot_product", "max_inner_product", "jaccard"], help="Similarity function for retrieval")
    parser.add_argument("--k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--first_n", type=int, default=0, help="Evaluate on first n examples, 0 for all")
    parser.add_argument("--eval_dataset_names", nargs="+", default=[
        "zechen-nlp/MNLP_STEM_mcqa_demo"
    ], help="List of evaluation dataset names")

    args = parser.parse_args()
    main(args)
