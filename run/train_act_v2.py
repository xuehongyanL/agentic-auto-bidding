"""
Agent 模式训练脚本 v2

基于 agent_auctionnet.yaml 配置，训练 ActModelV1/V2/DT（全参数训练）。

act_model_v2 新特性：
1. ActEmbeddingLayer 封装 LLM forward，对外只暴露 last_hidden_state
2. 各子类（V1/V2/DT）实现独立的 _forward_batch 和 _get_action
3. get_loss(traj, thoughts) 内部完成 target 构造，无需外部传入
   - V1: action loss
   - V2: action + rtg loss
   - DT: state + action + rtg loss（全序列）
4. 推理用 predict_batch，通过 _get_action 统一返回 [B, A]
5. 训练时 a_{W-1} 使用真实值（causal attention 保证无信息泄露）

训练流程：
1. 加载 YAML 配置
2. 根据 model.act_v2.type 构建 ActModelV1/V2/DT
3. 从 dataset 获取归一化统计量并注入模型
4. 训练循环：get_loss -> sum losses -> backward -> step
5. 定期保存 checkpoint
"""

import datetime
import os
import pickle
import time

import torch
import yaml
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from agb_auctionnet.data.dataset import AuctionNetDataset
from agb_auctionnet.infer.evaluate import evaluate
from agb_auctionnet.model.think_model import AuctionNetThinkModel
from agb_core.model.act_model_v2 import ActModelDT, ActModelV1, ActModelV2
from agb_core.model.agent_model import AgentModel
from agb_core.utils.argparse import ArgumentParser
from agb_core.utils.llm_backend import build_llm_backend
from agb_core.utils.path import glob_data_paths

MODEL_CLASSES = {
    'ActModelV1': ActModelV1,
    'ActModelV2': ActModelV2,
    'ActModelDT': ActModelDT,
}

TORCH_DTYPES = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}


def build_model(config: dict, resume: str | None, dataloader: DataLoader, torch_dtype: torch.dtype | None = None):
    """
    根据配置构建 ActModelV1/V2/DT，支持从 resume checkpoint 恢复。

    Args:
        config: 配置 dict
        resume: checkpoint 路径，None 表示 from scratch
        dataloader: 用于 from scratch 时获取归一化统计量
        torch_dtype: LLM 加载精度，默认 None (FP32)

    Returns:
        (model, optimizer, resume_step)
    """
    device = config['device']
    act_cfg = config['model']['act']
    train_cfg = config['train']['act']

    model_name = act_cfg.get('name', 'ActModelV2')
    if model_name not in MODEL_CLASSES:
        raise ValueError(f'train_act_v2.py 不支持 {model_name}，可选: {list(MODEL_CLASSES.keys())}')

    model_cls = MODEL_CLASSES[model_name]
    print(f'[Model] class={model_name}, torch_dtype={torch_dtype}')
    model = model_cls(
        base_model_path=act_cfg['path'],
        model_type=act_cfg.get('backend', 'transformers'),
        state_dim=config['task']['state_dim'],
        action_dim=config['task']['action_dim'],
        device=device,
        torch_dtype=torch_dtype,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 0.0),
    )

    if resume:
        print(f'[Setup] resume from {resume}')
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_step = checkpoint['step']
    else:
        print(f'[Setup] from scratch')
        assert isinstance(dataloader.dataset, AuctionNetDataset)
        model.set_normalize(dataloader.dataset.state_mean, dataloader.dataset.state_std)
        resume_step = 0

    model.to(device)
    return model, optimizer, resume_step


def build_dataloader(config: dict) -> DataLoader:
    """从环境 data_path 构建训练 DataLoader。"""
    task_cfg = config['task']
    train_cfg = config['train']['act']

    pattern_paths = glob_data_paths(train_cfg['data_path'])

    data_dicts = []
    for pp in pattern_paths:
        with open(pp, 'rb') as f:
            data_dict = pickle.load(f)
        if isinstance(data_dict, list):
            data_dicts.extend(data_dict)
        else:
            data_dicts.append(data_dict)

    print(f'[DataLoader] loaded {len(pattern_paths)} file(s) -> {len(data_dicts)} part(s)')

    dataset = AuctionNetDataset(
        state_dim=task_cfg['state_dim'],
        action_dim=task_cfg['action_dim'],
        window_size=task_cfg['window_size'],
    )
    dataset.load(data_dicts)

    ft_cfg = train_cfg.get('filter_thoughts')
    if ft_cfg:
        filter_mode = ft_cfg.get('mode', 'any')
        default_action = ft_cfg.get('default_action', 1.0)
        print(f'[DataLoader] filtering thoughts: mode={filter_mode}, default_action={default_action}')
        n_keep, n_filtered = dataset.filter_thoughts(mode=filter_mode, default_action=default_action)
        print(f'[DataLoader] filtered thoughts: {n_filtered} removed, {n_keep} kept (total {n_keep + n_filtered})')

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    return dataloader


def train(
    config: dict,
    save_dir: str,
    dataloader: DataLoader,
    model,
    optimizer,
    resume_step: int = 0,
    torch_dtype: torch.dtype | None = None,
    use_amp: bool = False,
):
    """
    训练循环。

    核心变化（相比 v1）：
    - 直接调用 model.get_loss(traj, thoughts)，无需手动构造 placeholder / target / normalize
    - get_loss 返回 dict[str, Tensor]，可按需加权求和
    - 支持 AMP 混合精度训练（use_amp=True 时启用 GradScaler）
    """
    device = config['device']
    train_cfg = config['train']['act']
    n_step = train_cfg['n_step']
    grad_accum_steps = train_cfg.get('grad_accum_steps', 1)
    eval_interval = train_cfg.get('eval_interval', None)
    save_interval = train_cfg.get('save_interval', None)
    log_interval = train_cfg.get('log_interval', None)

    # 必须显式配置，缺少则报错
    loss_weights = train_cfg['loss_weights']

    print(f'[Train] Device: {device}')
    print(f'[Train] Total steps: {n_step}, batch_size: {train_cfg["batch_size"]}, lr: {train_cfg["learning_rate"]}')
    print(f'[Train] Loss weights: {loss_weights if loss_weights else "all=1.0"}')
    use_grad_scaler = use_amp and (torch_dtype == torch.float16)
    print(f'[Train] AMP: autocast={use_amp}({torch_dtype}), grad_scaler={use_grad_scaler}')
    scaler = GradScaler('cuda', enabled=use_grad_scaler)
    assert isinstance(dataloader.dataset, AuctionNetDataset)
    print(f'[Train] Dataset size: {len(dataloader.dataset)}, batches: {len(dataloader)}')

    use_grad_scaler = use_amp and (torch_dtype == torch.float16)
    scaler = GradScaler('cuda', enabled=use_grad_scaler)
    model.train()

    eval_think_model = None
    if eval_interval:
        bcfg = config['model']['think']['llm_backend']
        llm_backend = build_llm_backend(bcfg)
        eval_think_model = AuctionNetThinkModel(llm_backend=llm_backend, verbose=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f'[Train] Trainable params: {sum(p.numel() for p in trainable_params):,}')

    scheduler_cfg = train_cfg.get('scheduler')
    if scheduler_cfg:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=train_cfg['learning_rate'],
            total_steps=n_step,
            pct_start=0.1,
        )
    else:
        scheduler = None

    step = resume_step
    accum_step = step % grad_accum_steps
    total_loss = 0.0
    epoch = 0

    print('[Train] Starting training...')
    start_time = time.time()

    while step < n_step:
        epoch += 1
        for batch in dataloader:
            traj, thoughts, step_info, traj_info = batch

            traj = traj._replace(
                states=traj.states.to(device),
                actions=traj.actions.to(device),
                rtgs=traj.rtgs.to(device),
                timesteps=traj.timesteps.to(device),
                attention_mask=traj.attention_mask.to(device),
            )
            with autocast('cuda', enabled=use_amp, dtype=torch_dtype or torch.bfloat16):
                losses = model.get_loss(traj, thoughts)
            total_batch_loss: torch.Tensor = torch.stack(
                [losses[k] * loss_weights[k] for k in losses]
            ).sum()

            # 梯度累积
            accum_step += 1
            if accum_step == grad_accum_steps:
                if use_grad_scaler:
                    scaler.scale(total_batch_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                accum_step = 0
                step += 1

                total_loss += total_batch_loss.item()

                if log_interval and step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    elapsed_time = time.time() - start_time
                    elapsed_steps = step - resume_step
                    lr = optimizer.param_groups[0]['lr']

                    loss_str = ' | '.join(f'{k}={losses[k].item():.4f}' for k in losses)
                    print(
                        f'[Step {step}/{n_step}] total={avg_loss:.6f} | {loss_str} | '
                        f'lr={lr:.2e} | elapsed={elapsed_time:.0f}s | {elapsed_steps/elapsed_time:.1f}it/s'
                    )
                    total_loss = 0.0

                if eval_think_model is not None and step % eval_interval == 0:
                    model.eval()
                    agent_model = AgentModel(eval_think_model, model)
                    metrics, _ = evaluate(agent_model, config, split='valid')
                    print(
                        f'[Eval @ Step {step}] '
                        f'gmv={metrics["avg_gmv"]:.2f} | '
                        f'cost={metrics["avg_cost"]:.2f} | '
                        f'cpa_ratio={metrics["avg_cpa_ratio"]:.2f} | '
                        f'score={metrics["avg_score"]:.2f}'
                    )
                    model.train()

                if save_interval and step % save_interval == 0:
                    save_path = os.path.join(save_dir, f'{step}.pt')
                    save_checkpoint(model, optimizer, step, save_path)
                    print(f'[Save] checkpoint saved to {save_path}')

                if step >= n_step:
                    break
                del losses, total_batch_loss
            else:
                del total_batch_loss

        if step < n_step:
            print(f'[Epoch {epoch}] dataset exhausted, reshuffling...')

    final_path = os.path.join(save_dir, f'{step}.pt')
    save_checkpoint(model, optimizer, step, final_path)
    print(f'[Train] Done! Final checkpoint: {final_path}')


def save_checkpoint(model, optimizer, step, path: str):
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config/agent_auctionnet.yaml',
                        help='Path to config YAML')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config = parser.apply_overrides(config)

    amp_dtype = config['train']['act'].get('amp')
    torch_dtype = TORCH_DTYPES.get(amp_dtype)
    use_amp = amp_dtype is not None

    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        save_dir = f'./saved_model/agent_train_v2_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print(f'[Config] save_dir: {save_dir}')

    dataloader = build_dataloader(config)
    model, optimizer, resume_step = build_model(config, args.resume, dataloader, torch_dtype)

    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    train(config, save_dir, dataloader, model, optimizer, resume_step,
          torch_dtype=torch_dtype, use_amp=use_amp)


if __name__ == '__main__':
    main()
