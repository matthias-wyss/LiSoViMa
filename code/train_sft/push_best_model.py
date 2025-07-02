import argparse
import os
import json
from huggingface_hub import HfApi, login
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_best_model(output_dir):
    best_loss = float("inf")
    best_model_path = None

    # Search in checkpoints
    for subdir in os.listdir(output_dir):
        if subdir.startswith("checkpoint-"):
            state_path = os.path.join(output_dir, subdir, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path, "r") as f:
                    state = json.load(f)
                eval_loss = state.get("best_metric")
                if eval_loss is not None and eval_loss < best_loss:
                    best_loss = eval_loss
                    best_model_path = os.path.join(output_dir, subdir)

    # Search in final model directory
    final_state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(final_state_path):
        with open(final_state_path, "r") as f:
            final_state = json.load(f)
        final_eval_loss = final_state.get("eval_loss") or final_state.get("best_metric")
        if final_eval_loss is not None and final_eval_loss < best_loss:
            best_loss = final_eval_loss
            best_model_path = output_dir

    if best_model_path is None:
        print("No checkpoint with trainer_state.json found. Using final model.")
        return output_dir
    else:
        print(f"Best checkpoint: {best_model_path} with eval_loss = {best_loss}")
        return best_model_path


def push_model_to_hub(hf_token, repo_name, output_dir):
    login(token=hf_token)
    api = HfApi(token=hf_token)

    try:
        api.model_info(repo_name)
        print(f"✅ Repo already exists on Hugging Face Hub: '{repo_name}'. Aborting push.")
        return
    except RepositoryNotFoundError:
        print("Repo not found. Proceeding with push...")

    print("Preparing best model...")
    best_model_path = get_best_model(output_dir)

    print(f"📤 Pushing model and tokenizer from {best_model_path} to {repo_name}...")
    tokenizer = AutoTokenizer.from_pretrained(best_model_path)
    model = AutoModelForCausalLM.from_pretrained(best_model_path)

    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"✅ Model successfully pushed to: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push best checkpoint to Hugging Face Hub")

    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face API token")
    parser.add_argument("--repo_name", type=str, required=True, help="Repo name to push to (e.g. user/model-name)")
    parser.add_argument("--output_dir", type=str, default="./sft-output", help="Directory containing training outputs")

    args = parser.parse_args()

    push_model_to_hub(args.hf_token, args.repo_name, args.output_dir)
