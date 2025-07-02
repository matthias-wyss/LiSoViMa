import os
import sys
import base64
import time
import fitz  # PyMuPDF
import re
import argparse
from mistralai import Mistral
from tqdm import tqdm
from datasets import Dataset
from huggingface_hub import login, hf_api
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoTokenizer

# --- OCR Configuration ---
chunk_size = 10
ocr_model = "mistral-ocr-latest"

def encode_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            return base64.b64encode(pdf_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding {pdf_path}: {e}")
        return None

def save_chunk_pdf(input_path, output_path, start_page, end_page):
    doc = fitz.open(input_path)
    new_doc = fitz.open()

    total_pages = len(doc)

    for i in range(start_page, min(end_page, total_pages)):
        try:
            new_doc.insert_pdf(doc, from_page=i, to_page=i)
        except Exception:
            continue

    if len(new_doc) > 0:
        new_doc.save(output_path)

    new_doc.close()
    doc.close()




def extract_text_from_pdf(pdf_path, client, retries=3, delay=10):
    base64_pdf = encode_pdf(pdf_path)
    if base64_pdf is None:
        return None

    for attempt in range(retries):
        try:
            ocr_response = client.ocr.process(
                model=ocr_model,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{base64_pdf}"
                },
                include_image_base64=False
            )
            return "\n".join([page.markdown for page in ocr_response.pages])
        except Exception as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"Rate limit hit for {pdf_path}, retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"OCR failed for {pdf_path}: {e}")
                return None
    print(f"OCR failed for {pdf_path} after {retries} retries.")
    return None

def save_markdown(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def process_pdfs(pdfs_path, api_key):
    client = Mistral(api_key=api_key)
    md_folder = os.path.join(pdfs_path, "md_outputs")
    os.makedirs(md_folder, exist_ok=True)

    pdf_files = [f for f in os.listdir(pdfs_path) if f.endswith(".pdf")]

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        full_path = os.path.join(pdfs_path, pdf_file)
        doc = fitz.open(full_path)
        total_pages = len(doc)
        doc.close()
        
        base_name = os.path.splitext(pdf_file)[0]
        subfolder = os.path.join(pdfs_path, base_name)
        os.makedirs(subfolder, exist_ok=True)

        for i, start in enumerate(range(0, total_pages, chunk_size)):
            end = start + chunk_size
            chunk_name = f"{start}_{min(end-1, total_pages-1)}.pdf"
            chunk_path = os.path.join(subfolder, chunk_name)
            print(f"⏳ Chunking {pdf_file} pages {start} to {end-1}")
            save_chunk_pdf(full_path, chunk_path, start, end)

        full_markdown = ""
        chunk_files = sorted(f for f in os.listdir(subfolder) if f.endswith(".pdf"))
        for chunk_file in chunk_files:
            chunk_path = os.path.join(subfolder, chunk_file)
            text = extract_text_from_pdf(chunk_path, client)
            time.sleep(2)
            if text:
                #full_markdown += f"\n\n<!-- {chunk_file} -->\n\n"
                clean_text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
                full_markdown += clean_text

        output_md_path = os.path.join(md_folder, f"{base_name}.md")
        save_markdown(full_markdown, output_md_path)
    print(f"✅ OCR processing completed. Markdown files saved in '{md_folder}'.")

    return md_folder


def chunk_text(text, tokenizer, chunk_size, chunk_overlap):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []

    i = 0
    while i < len(tokens):
        token_chunk = tokens[i:i + chunk_size]
        text_chunk = tokenizer.decode(token_chunk, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        chunks.append(text_chunk.strip())
        i += chunk_size - chunk_overlap

    return chunks



def upload_to_huggingface(md_folder, hf_token, rag_corpus_name):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    login(token=hf_token)
    api = hf_api.HfApi()
    repo_id = rag_corpus_name

    try:
        api.dataset_info(repo_id)
        print(f"✅ Dataset '{repo_id}' already exists on HF Hub. Skipping upload.")
        return
    except RepositoryNotFoundError:
        print(f"Uploading new dataset '{repo_id}'...")

    md_files = [f for f in os.listdir(md_folder) if f.endswith(".md")]
    data = []
    for md_file in tqdm(md_files, desc="Loading markdowns"):
        path = os.path.join(md_folder, md_file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        book_title = os.path.splitext(md_file)[0].replace("_", " ")
        chunks = chunk_text(text, tokenizer, args.chunk_size, args.chunk_overlap)
        for i, chunk in enumerate(chunks):
            data.append({
                "text": chunk,
                "source": f"{book_title} (converted to markdown via Mistral OCR) - chunk {i+1}"
            })

    dataset = Dataset.from_list(data)
    dataset.push_to_hub(repo_id, token=hf_token)
    print(f"✅ RAG corpus uploaded: https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and upload RAG dataset from PDFs using OCR.")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--rag_corpus_name", type=str, required=True, help="Name of the dataset on Hugging Face Hub")
    parser.add_argument("--pdf_folder", type=str, required=True, help="Folder containing PDF files")
    parser.add_argument("--mistral_api_key", type=str, required=True, help="API key for Mistral OCR")
    parser.add_argument("--base_model", type=str, required=True, help="Base model used to tokenize and chunk (e.g. Qwen/Qwen3-0.6B-Base)")
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=0, help="Chunk overlap for text splitting")


    args = parser.parse_args()

    api = hf_api.HfApi()
    md_folder = os.path.join(args.pdf_folder, "md_outputs")

    # 1. Check if dataset exists on HF Hub
    try:
        api.dataset_info(args.rag_corpus_name)
        print(f"✅ Dataset '{args.rag_corpus_name}' already exists on HF Hub. Skipping all processing and upload.")
        sys.exit(0)
    except RepositoryNotFoundError:
        print(f"Dataset '{args.rag_corpus_name}' not found on HF Hub. Checking for existing markdown folder...")

    # 2. Check if markdown folder already exists (means OCR was already done)
    if os.path.exists(md_folder) and len([f for f in os.listdir(md_folder) if f.endswith(".md")]) > 0:
        print(f"✅ Markdown folder '{md_folder}' already exists with .md files. Skipping OCR processing.")
    else:
        print("Markdown folder does not exist or is empty. Starting OCR processing...")
        # 3. Run OCR processing
        md_folder = process_pdfs(args.pdf_folder, args.mistral_api_key)
    
    # 4. Upload (force upload to HF even if markdown existed before)
    upload_to_huggingface(md_folder, args.hf_token, args.rag_corpus_name)


