import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import login
import hashlib
import re
import pandas as pd
from mistralai import Mistral
from tqdm import tqdm
import time
import random
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training


def preprocess_mcqa(example):
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    prompt += example["question"] + "\n"
    for key, choice in zip(['A', 'B', 'C', 'D'], example["choices"]):
        prompt += f"{key}. {choice}\n"
    prompt += "Answer: "
    response = " " + example["answer"]
    return {"prompt": prompt, "completion": response}

def push_dataset(repo_name, formatted_ds, hf_token):
    login(token=hf_token)
    formatted_ds.push_to_hub(
        repo_name,
        private=False,
        token=hf_token,
        commit_message="Upload formatted dataset"
    )

def preprocess_sciq(hf_user, hf_token):
    def format_example(example):
        LETTER_INDICES = ["A", "B", "C", "D"]
        answers = [
            example["correct_answer"],
            example["distractor1"],
            example["distractor2"],
            example["distractor3"]
        ]
        seed = int(hashlib.md5(example["question"].encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)
        random.shuffle(answers)
        
        correct_index = answers.index(example["correct_answer"])
        correct_letter = LETTER_INDICES[correct_index]
        
        choices = answers
        
        return {
            "question": example["question"],
            "choices": choices,
            "answer": correct_letter,
            "support": example.get("support", None),
            "source": "SCiQ"
        }

    scq = load_dataset("allenai/sciq")
    formatted_ds = DatasetDict()
    for split in scq.keys():
        formatted_ds[split] = scq[split].map(format_example, remove_columns=scq[split].column_names)
    push_dataset(f"{hf_user}/mnlp_sciq", formatted_ds, hf_token)
    

def preprocess_aquarat(hf_user, hf_token):
    LETTER_INDICES_4 = ["A", "B", "C", "D"]
    LETTER_INDICES_5 = ["A", "B", "C", "D", "E"]

    def clean_choice_text(text):
        return re.sub(r"^[A-E][\)\.]?\s*", "", text).strip()

    def format_aquarat_example(example):
        question = example["question"]
        options = example["options"]
        correct_letter = example["correct"]
        support = example.get("rationale", None) 
        
        cleaned_options = [clean_choice_text(opt) for opt in options]

        seed = int(hashlib.md5(question.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)

        correct_index = LETTER_INDICES_5.index(correct_letter)
        correct_choice = cleaned_options[correct_index]

        wrong_options = [(i, opt) for i, opt in enumerate(cleaned_options) if i != correct_index]

        removed_index, _ = random.choice(wrong_options)

        filtered_options = [opt for i, opt in enumerate(cleaned_options) if i != removed_index]

        paired_options = list(enumerate(filtered_options))
        random.shuffle(paired_options)

        shuffled_options = [opt for i, opt in paired_options]

        for new_idx, (orig_idx, opt) in enumerate(paired_options):
            if opt == correct_choice:
                new_correct_index = new_idx
                break

        new_correct_letter = LETTER_INDICES_4[new_correct_index]

        return {
            "question": question,
            "choices": shuffled_options,
            "answer": new_correct_letter,
            "support": support,
            "source": "AquaRat"
        }

    splits = ["train", "validation", "test"]
    formatted_ds = DatasetDict()

    for split in splits:
        ds_split = load_dataset("deepmind/aqua_rat", split=split)
        formatted_ds[split] = ds_split.map(format_aquarat_example, remove_columns=ds_split.column_names)

    push_dataset(f"{hf_user}/mnlp_aquarat", formatted_ds, hf_token)


def preprocess_mmlu(hf_user, hf_token, mistral_api_key):
    subjects = [
        'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge',
        'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine',
        'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering',
        'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry',
        'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics',
        'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology',
        'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law',
        'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes',
        'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine',
        'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy',
        'virology', 'world_religions'
    ]   
    stem_subjects = [
        'abstract_algebra', 'astronomy', 'college_biology', 'college_chemistry', 'college_computer_science', 
        'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'econometrics', 
        'electrical_engineering', 'high_school_biology', 'high_school_chemistry', 
        'high_school_computer_science', 'high_school_mathematics', 'high_school_physics', 'machine_learning', 
        'medical_genetics', 'professional_medicine', 'virology', 'anatomy', 'clinical_knowledge', 'conceptual_physics', 'high_school_statistics'
    ]   
    dataset = load_dataset("cais/mmlu", "auxiliary_train")
    df = dataset["train"].to_pandas()
    df = pd.json_normalize(df["train"])
    df = df.rename(columns={"question": "question_text", "choices": "options", "answer": "answer"})

    client = Mistral(api_key=mistral_api_key)
    model = "mistral-large-latest"

    start_idx = df[df["subject"].isna()].index.min()

    if pd.isna(start_idx):
        start_idx = 0

    for idx in tqdm(range(start_idx, len(df))):
        row = df.iloc[idx]
        
        question = row["question_text"]
        options = row["options"]
        


        prompt = f"""
            Given the multiple choice question below, classify it into its most relevant academic subject from the list provided.

            Subjects:
            {', '.join(subjects)}

            Question:
            {question}

            Options:
            {options}

            Respond with only one subject from the list above.
        """

        try:
            chat_response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt.strip()}]
            )

            subject = chat_response.choices[0].message.content.strip()
            df.iloc[idx, df.columns.get_loc("subject")] = subject
            time.sleep(3)

        except Exception as e:
            df.iloc[idx, df.columns.get_loc("subject")] = "error"
            time.sleep(3)

    df = df[df['subject'].isin(stem_subjects)]

    def clean_choices(option_str):
        if not isinstance(option_str, str):
            return []
        # Match both '...' and "..." quoted options
        matches = re.findall(r"""(['"])(.*?)\1""", option_str, flags=re.DOTALL)
        return [m[1].strip() for m in matches]


    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    
    formatted_df = pd.DataFrame({
        "question": df["question_text"],
        "support": "",  # Empty support
        "choices": df["options"].apply(clean_choices),
        "answer": df["answer"].map(answer_map),
        "source": "MMLU_train"
    })

    formatted_df = formatted_df[formatted_df["choices"].apply(lambda x: len(x) == 4)].reset_index(drop=True)

    push_dataset(f"{hf_user}/mnlp_mmlu", formatted_df, hf_token)


def merge_datasets(hf_user, hf_token):
    def standardize_split(split_dataset, source_name):
        if 'support' not in split_dataset.column_names:
            split_dataset = split_dataset.add_column('support', [''] * len(split_dataset))

        keep_cols = ['question', 'support', 'choices', 'answer']
        remove_cols = [col for col in split_dataset.column_names if col not in keep_cols]
        split_dataset = split_dataset.remove_columns(remove_cols)

        split_dataset = split_dataset.add_column('source', [source_name] * len(split_dataset))
        return split_dataset

    # Group splits
    splits = {
        "train": [],
        "validation": [],
        "test": []
    }

    datasets = [
        (f"{hf_user}/mnlp_aquarat", "AquaRat"),
        (f"{hf_user}/mnlp_sciq", "SCiQ"),
        (f"{hf_user}/mnlp_mmlu", "MMLU")
    ]

    for dataset_name, source_tag in datasets:
        ds_dict = load_dataset(dataset_name)
        for split_name, split_data in ds_dict.items():
            if split_name not in splits:
                splits[split_name] = []  # this is to handle unexpected split names gracefully
            cleaned = standardize_split(split_data, f"{source_tag}_{split_name}")

            if split_name == "train":
                cleaned = cleaned.shuffle(seed=42).select(range(10000))

            splits[split_name].append(cleaned)

    final_splits = {}
    for split_name, datasets in splits.items():
        if datasets: 
            merged_split = concatenate_datasets(datasets).shuffle(seed=42)
            final_splits[split_name] = merged_split

    final_dataset = DatasetDict(final_splits)
    push_dataset(f"{hf_user}/MNLP_M3_mcqa_dataset", final_dataset, hf_token)


def preprocess_dataset(hf_user, hf_token, mistral_api_key):
    preprocess_sciq(hf_user, hf_token)
    preprocess_aquarat(hf_user, hf_token)
    preprocess_mmlu(hf_user, hf_token, mistral_api_key)
    merge_datasets(hf_user, hf_token)


def main():
    parser = argparse.ArgumentParser(description="Train LoRA Quantized model script")

    # Dataset preparation args
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face username or org")
    parser.add_argument("--dataset_name", type=str, default="mkartofel/MNLP_M3_quantized_dataset", help="Dataset name for training")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face token for authentication")
    parser.add_argument("--mistral_api_key", type=str, required=True, help="API key for Mistral OCR")

    # Training args
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base", help="Base model identifier")
    parser.add_argument("--output_dir", type=str, default="./Qwen3-0.6B-qlora-MCQA_lora_final_30k", help="Output directory for checkpoints and model")
    parser.add_argument("--sample_size", type=int, default=30000, help="Number of samples to use for training")

    args = parser.parse_args()

    print("Loading dataset...")
    try:
        train_dataset = load_dataset(args.dataset_name)['train']
        val_dataset = load_dataset(args.dataset_name)['validation']
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Dataset not found or error loading: {e}")
        print("Preprocessing dataset...")
        preprocess_dataset(args.hf_user, args.hf_token, args.mistral_api_key)
        train_dataset = load_dataset(args.dataset_name)['train']
        val_dataset = load_dataset(args.dataset_name)['validation']

    print("Preprocessing dataset...")
    train_dataset = train_dataset.map(preprocess_mcqa)
    val_dataset = val_dataset.map(preprocess_mcqa)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # nf4 introduced in Qlora paper
        bnb_4bit_use_double_quant=True, # Qlora paper: increase size reduction for no accuracy loss
        bnb_4bit_compute_dtype=torch.bfloat16, # recommended in Qlora paper
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, # Qlora paper: unrelated to final performance if LoRA is used on all layers as can be seen
        lora_alpha=16,
        lora_dropout=0.05, # Qlora paper: 0.05 for smaller models
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # all Qwen3 linear layers
    )
    model = get_peft_model(model, lora_config)

    max_seq_len = 2048

    LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    def tokenize_mcqa(sample):
        topic = "knowledge and skills in advanced master-level STEM courses"
        prompt = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
        prompt += sample["question"] + "\n"
        prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, sample["choices"])])
        prompt += "Answer:"
        answer = " " + sample["answer"].strip()
        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=max_seq_len, padding=False, add_special_tokens=False
        )["input_ids"]
        answer_ids = tokenizer(
            answer, truncation=True, max_length=8, padding=False, add_special_tokens=False
        )["input_ids"]
        input_ids = prompt_ids + answer_ids
        # Labels: -100 everywhere except answer
        labels = [-100] * len(prompt_ids) + answer_ids
        pad_len = max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids += [tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
        else:
            input_ids = input_ids[:max_seq_len]
            labels = labels[:max_seq_len]
        attn_mask = [1] * len(prompt_ids + answer_ids) + [0] * pad_len
        attn_mask = attn_mask[:max_seq_len]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn_mask}

    train_dataset = train_dataset.shuffle(seed=42).select(range(args.sample_size))
    train_dataset = train_dataset.map(tokenize_mcqa, batched=False)
    train_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=1,
        learning_rate=2e-4, # Qlora paper: 2e-4 for smaller models
        fp16=False,
        bf16=True, # recommended if gpu supports it
        save_steps=500,
        output_dir=args.output_dir,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
        },
    )

    trainer.train()

    model_merged = model.merge_and_unload() # merge LoRA weights into base model

    model_merged.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
