import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import login
import hashlib
import re
import pandas as pd
from mistralai import Mistral
from tqdm import tqdm
import time
import random


def preprocess_mcqa(example):
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    prompt += example["question"] + "\n"
    for key, choice in zip(['A', 'B', 'C', 'D'], example["choices"]):
        prompt += f"{key}. {choice}\n"
    prompt += "Answer: "
    response = " " + example["answer"]
    return {"prompt": prompt, "completion": response}

def push_dataset(repo_name, formatted_ds):
    login(token="")
    formatted_ds.push_to_hub(
    repo_name,
    private=False,
    token="",
    commit_message="Upload formatted dataset"
)



def preprocess_dataset(): 
    preprocess_sciq()
    preprocess_aquarat()
    preprocess_mmlu()
    merge_datasets()


def preprocess_sciq():
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
    push_dataset("LinaSad/mnlp_sciq", formatted_ds)
    

def preprocess_aquarat():
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

    push_dataset("LinaSad/mnlp_aquarat", formatted_ds)


def preprocess_mmlu():
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

    client = Mistral(api_key="")
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

    push_dataset("LinaSad/mnlp_mmlu", formatted_df)


def merge_datasets():
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
        ("LinaSad/mnlp_aquarat", "AquaRat"),
        ("LinaSad/mnlp_sciq", "SCiQ"),
        ("LinaSad/mnlp_mmlu", "MMLU")
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
    push_dataset("LinaSad/MNLP_M3_mcqa_dataset", final_dataset)


def main():
    parser = argparse.ArgumentParser(description="Train LoRA MCQA model script")

    # Dataset preparation args
    parser.add_argument("--hf_user", type=str, required=True, help="Hugging Face username or org")
    parser.add_argument("--dataset_name", type=str, default="LinaSad/MNLP_M3_mcqa_dataset", help="Dataset name for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Training args
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B-Base", help="Base model identifier")
    parser.add_argument("--output_dir", type=str, default="./qwen-general-lora", help="Output directory for checkpoints and model")
    parser.add_argument("--batch_size", type=int, default=4, help="Train batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--fp16", action='store_true', help="Use fp16 mixed precision training")
    parser.add_argument("--bf16", action='store_true', help="Use bf16 mixed precision training")

    args = parser.parse_args()

    print("Loading dataset...")
    try:
        train_dataset = load_dataset(args.dataset_name)['train']
        val_dataset = load_dataset(args.dataset_name)['validation']
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Dataset not found or error loading: {e}")
        print("Preprocessing dataset...")
        preprocess_dataset()
        train_dataset = load_dataset(args.dataset_name)['train']
        val_dataset = load_dataset(args.dataset_name)['validation']

    print("Preprocessing dataset...")
    train_dataset = train_dataset.map(preprocess_mcqa)
    val_dataset = val_dataset.map(preprocess_mcqa)

    print(f"Loading base model {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)

    # Freezing base model weights for LoRA
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Adding LoRA adapters
    peft_config = LoraConfig(
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
    model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy="steps",
        logging_dir="./logs",
        logging_steps=100,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        gradient_checkpointing=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",      
        eval_steps=100,
        save_safetensors=False,
        load_best_model_at_end=True,     
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=True,
    )

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete. Model saved to", args.output_dir)

if __name__ == "__main__":
    main()
