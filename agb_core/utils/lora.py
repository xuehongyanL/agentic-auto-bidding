"""
LoRA utility: merge adapter weights into base model and save as a standalone HF directory.
"""

import argparse
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(adapter_path: str, base_path: str, output_path: str) -> None:
    """
    Load base model, apply LoRA adapter, merge weights, and save to output_path.

    Args:
        adapter_path: Directory containing LoRA adapter (adapter_config.json, adapter_model.safetensors).
        base_path: Directory containing the base model.
        output_path: Directory to save the merged model.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path,
        device_map='cpu',
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge LoRA adapter into base model')
    parser.add_argument('--adapter', type=str, required=True)
    parser.add_argument('--base', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    merge_lora(args.adapter, args.base, args.output)
