"""
Think Model SFT 训练脚本

从 evaluate() 输出的轨迹 pkl 中读取每步的最佳 thought，以 prompt-answer 对的形式
对 Think LLM 进行 SFT。
使用 TRL 的 SFTTrainer，尽可能复用默认实践。
"""

import datetime
import os
import pickle
import sys

import torch
import transformers
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)
from trl.trainer.sft_trainer import SFTTrainer

from agb_core.utils.argparse import ArgumentParser
from agb_core.utils.path import glob_data_paths


def extract_samples(trajectories):
    """
    从 trajectories 中提取所有 prompt-answer 样本对。

    evaluate pkl 中每个 step 已有 'prompt' (list[dict]) 和 'thoughts' (list[str])。
    取 score 最高的 sample 作为最佳样本，拼成 chat 格式的 text。

    Args:
        trajectories: evaluate pkl 加载出的 episode 列表

    Returns:
        list of dicts: [{'messages': list[dict]}, ...]
    """
    samples = []
    for ep in trajectories:
        for step in ep['steps']:
            scores = step['scores']
            if not scores:
                continue
            # 取 score 最高的 sample
            best_idx = scores.index(max(scores))
            best_thought = step['thoughts'][best_idx]
            if not best_thought.strip():
                continue

            # step['prompt'] 是 list[dict] (system + user messages)
            # 直接在其后追加 assistant response
            messages = list(step['prompt'])  # copy
            messages.append({'role': 'assistant', 'content': best_thought})
            samples.append({'messages': messages})

    return samples


def load_trajectories(data_path: str):
    """加载轨迹 pkl，支持 glob 模式。"""
    trajectories = []
    for p in glob_data_paths(data_path):
        with open(p, 'rb') as f:
            data = pickle.load(f)
        trajectories.extend(data)
    return trajectories


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/agent_auctionnet.yaml',
                        help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = parser.apply_overrides(config)

    think_cfg = config['train']['think']
    model_path = config['model']['think']['llm_backend']['model_path']

    # 保存目录
    output_dir = think_cfg.get('output_dir')
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        output_dir = f'./saved_model/think_{timestamp}'
        os.makedirs(output_dir, exist_ok=True)
    print(f'[Config] output_dir: {output_dir}')
    print(f'[Config] model_path: {model_path}')

    # ---------- 1. 加载数据 ----------
    data_path = think_cfg['data_path']
    print(f'Loading trajectories from: {data_path}')
    trajectories = load_trajectories(data_path)
    print(f'Loaded {len(trajectories)} episodes')

    samples = extract_samples(trajectories)
    print(f'Extracted {len(samples)} training samples')

    if len(samples) == 0:
        print('ERROR: no valid samples found. Exiting.')
        sys.exit(1)

    dataset = Dataset.from_list(samples)
    print(f'Dataset: {len(dataset)} samples')

    # ---------- 2. 初始化 tokenizer ----------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 用 apply_chat_template 将 messages 列表转为字符串
    def format_fn(example):
        text = tokenizer.apply_chat_template(
            example['messages'], tokenize=False, add_generation_prompt=False
        )
        return {'text': text}

    dataset = dataset.map(format_fn, remove_columns=['messages'])

    # ---------- 3. 初始化模型 ----------
    finetune_mode = think_cfg.get('finetune_mode', 'qlora')

    if finetune_mode == 'full':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            trust_remote_code=True,
        )
        peft_config = None

    elif finetune_mode == 'qlora':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=think_cfg.get('lora_r', 16),
            lora_alpha=think_cfg.get('lora_alpha', 32),
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=think_cfg.get('lora_dropout', 0.05),
            bias='none',
            task_type='CAUSAL_LM',
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        raise ValueError(f"finetune_mode must be 'qlora' or 'full', got '{finetune_mode}'")

    # ---------- 4. SFTTrainer ----------
    trainer = SFTTrainer(
        model=model,  # type: ignore[arg-type]  # PeftModel / PreTrainedModel 均被接受，stub 签名不完整
        args=transformers.TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=think_cfg.get('num_epochs', 3),
            per_device_train_batch_size=think_cfg.get('per_device_train_batch_size', 4),
            gradient_accumulation_steps=think_cfg.get('gradient_accumulation_steps', 4),
            learning_rate=think_cfg.get('learning_rate', 1e-4),
            lr_scheduler_type='cosine',
            warmup_ratio=think_cfg.get('warmup_ratio', 0.1),
            bf16=True,
            logging_steps=think_cfg.get('logging_steps', 10),
            save_steps=think_cfg.get('save_steps', 1000),
            save_total_limit=think_cfg.get('save_total_limit', None),
            remove_unused_columns=False,
            optim='adamw_torch',
        ),
        train_dataset=dataset,
        peft_config=peft_config,
    )

    # 保存 config 副本
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(output_dir)
    print(f'Training done. Model saved to {output_dir}')


if __name__ == '__main__':
    main()
