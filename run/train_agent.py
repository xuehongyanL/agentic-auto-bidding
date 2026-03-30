"""
Agent 模式训练脚本

基于 agent_auctionnet.yaml 配置，训练 ActModel（全参数训练）。

训练流程：
1. 加载 YAML 配置和 normalize_dict
2. 构建 AuctionNetDataset（从 env.data_path 的 CSV）
3. 初始化 ActModel，加载 LLM backbone（全参数可训练）
4. 设置优化器（AdamW，优化所有参数）
5. 训练循环：DataLoader -> 前向传播 -> MSE Loss -> 反向传播 -> 梯度更新
6. 定期保存 checkpoint
"""

import argparse
import datetime
import os
import pickle
import time

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from agb_auctionnet.data.dataset import AuctionNetDataset
from agb_auctionnet.infer.evaluate import evaluate
from agb_auctionnet.model.think_model import AuctionNetThinkModel
from agb_core.model.act_model import ActModel
from agb_core.model.agent_model import AgentModel
from agb_core.utils.llm_backend import build_llm_backend
from agb_core.utils.path import glob_data_paths


def setup_model(config: dict, resume: str | None, dataloader: DataLoader):
    """
    构建 model 和 optimizer，支持从 resume checkpoint 恢复。

    Args:
        config: 配置 dict
        resume: checkpoint 路径，None 表示 from scratch
        dataloader: 用于 from scratch 时获取归一化统计量

    Returns:
        (model, optimizer, resume_step)
    """
    device = config['device']
    train_cfg = config['train']['act']

    model = ActModel(
        model_path=config['model']['act']['path'],
        model_type=config['model']['act']['backend'],
        state_dim=config['task']['state_dim'],
        action_dim=config['task']['action_dim'],
        device=device,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg.get('weight_decay', 0.0),
    )

    if resume:
        print(f'[Setup] resume from {resume}')
        checkpoint = torch.load(resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resume_step = checkpoint['step']
    else:
        print('[Setup] from scratch')
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

    # 过滤 thoughts
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


def train(config: dict, save_dir: str, dataloader: DataLoader, model, optimizer, resume_step: int = 0):
    device = config['device']
    train_cfg = config['train']['act']
    n_step = train_cfg['n_step']
    grad_accum_steps = train_cfg.get('grad_accum_steps', 1)
    eval_interval = train_cfg.get('eval_interval', None)
    save_interval = train_cfg.get('save_interval', None)
    log_interval = train_cfg.get('log_interval', None)

    print(f'[Train] Device: {device}')
    print(f'[Train] Total steps: {n_step}, batch_size: {train_cfg['batch_size']}, lr: {train_cfg['learning_rate']}')
    assert isinstance(dataloader.dataset, AuctionNetDataset)
    print(f'[Train] Dataset size: {len(dataloader.dataset)}, batches: {len(dataloader)}')

    model.train()

    # 预构建 eval 专用的 Think 模型（复用，训练过程中不更新）
    bcfg = config['model']['think']['llm_backend']
    llm_backend = build_llm_backend(bcfg)
    eval_think_model = AuctionNetThinkModel(llm_backend=llm_backend, verbose=0)

    # 可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f'[Train] Trainable params: {sum(p.numel() for p in trainable_params):,}')

    # 学习率调度器（可选）
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

    # 损失函数
    criterion = nn.MSELoss()

    # 训练状态
    step = resume_step
    accum_step = step % grad_accum_steps  # 独立的累积计数器
    total_loss = 0.0
    epoch = 0

    print(f'[Train] Starting training...')
    start_time = time.time()

    while step < n_step:
        epoch += 1
        for batch in dataloader:
            traj, thoughts, step_info, traj_info = batch

            # 目标动作：取每个 trajectory 最后一步的真实动作
            # attention_mask: [B, W, A]，max over A 维得到每步是否有效
            # argmax 得到最后有效位置（valid 全是1，padding 全是0）
            valid_mask = traj.attention_mask.max(dim=2)[0]  # [B, W]
            B = traj.states.shape[0]
            target_tensor = traj.actions[:, -1].to(device)  # [B, action_dim]

            # 构造模型输入：与推理阶段完全对齐，无未来信息泄漏
            # dataset 原始: states[W+1], actions[W], rtgs[W+1]
            # 模型期望: states[W], actions[W]（末尾 placeholder）, rtgs[W]
            #   states: 去掉最后一个 next_state
            #   actions: 历史动作 + placeholder（末尾替换 target）
            #   rtgs: 去掉最后一个 rtg
            W = valid_mask.shape[1]
            A = model._action_dim
            placeholder = torch.zeros(B, 1, A, device=traj.actions.device, dtype=traj.actions.dtype)
            traj_for_model = traj._replace(
                states=traj.states[:, :-1],                       # [B, W+1, S] -> [B, W, S]
                actions=torch.cat([traj.actions[:, :-1], placeholder], dim=1),  # [B, W, A]
                rtgs=traj.rtgs[:, :-1],                          # [B, W+1, 1] -> [B, W, 1]
                timesteps=traj.timesteps,                         # [B, W]
            )
            # 直接调 forward 保留梯度（不用 predict_batch，因为它会 detach 转 numpy）
            state_mean = model._state_mean.to(traj_for_model.states.device, non_blocking=True)
            state_std = model._state_std.to(traj_for_model.states.device, non_blocking=True)
            states_norm = (traj_for_model.states - state_mean) / (state_std + 1e-9)
            traj_for_model = traj_for_model._replace(states=states_norm)
            actions_pred_tensor = model._forward_batch(thoughts, traj_for_model)

            # 计算损失
            loss = criterion(actions_pred_tensor, target_tensor)

            # 梯度累积：只有累积到最后一步时才 backward，其余步骤跳过
            accum_step += 1
            if accum_step == grad_accum_steps:
                # 累积完成，执行 backward 和 optimizer 更新
                loss.backward()

                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()

                accum_step = 0
                step += 1

                total_loss += loss.item()

                # 日志
                if log_interval and step % log_interval == 0:
                    avg_loss = total_loss / log_interval
                    elapsed = time.time() - start_time
                    lr = optimizer.param_groups[0]['lr']
                    print(
                        f'[Step {step}/{n_step}] loss={avg_loss:.6f} | lr={lr:.2e} | '
                        f'elapsed={elapsed:.0f}s | steps/s={step/elapsed:.1f}'
                    )
                    total_loss = 0.0

                # 验证
                if eval_interval and step % eval_interval == 0:
                    model.eval()
                    agent_model = AgentModel(eval_think_model, model)
                    metrics = evaluate(agent_model, config, split='valid')
                    print(
                        f'[Eval @ Step {step}] '
                        f'gmv={metrics["avg_gmv"]:.2f} | '
                        f'cost={metrics["avg_cost"]:.2f} | '
                        f'cpa={metrics["avg_cpa"]:.2f} | '
                        f'score={metrics["avg_score"]:.2f}'
                    )
                    model.train()

                # 保存 checkpoint
                if save_interval and step % save_interval == 0:
                    save_path = os.path.join(save_dir, f'{step}.pt')
                    save_checkpoint(model, optimizer, step, save_path)
                    print(f'[Save] checkpoint saved to {save_path}')

                if step >= n_step:
                    break
            else:
                # 非累积步骤：纯 forward，不 backward，节省显存
                # detach 防止构建无用的计算图
                del actions_pred_tensor, loss

        # epoch 结束，重新打乱
        if step < n_step:
            print(f'[Epoch {epoch}] dataset exhausted, reshuffling...')

    # 最后保存
    final_path = os.path.join(save_dir, f'{step}.pt')
    save_checkpoint(model, optimizer, step, final_path)
    print(f'[Train] Done! Final checkpoint: {final_path}')


def save_checkpoint(model, optimizer, step, path: str):
    """保存模型权重和优化器状态。"""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/agent_auctionnet.yaml',
                        help='Path to config YAML')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 保存目录
    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        save_dir = f'./saved_model/agent_train_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    print(f'[Config] save_dir: {save_dir}')

    # 构建 dataloader（用于归一化统计量）
    dataloader = build_dataloader(config)

    # 构建 model 和 optimizer
    model, optimizer, resume_step = setup_model(config, args.resume, dataloader)

    # 保存 config 副本
    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)

    train(config, save_dir, dataloader, model, optimizer, resume_step)


if __name__ == '__main__':
    main()
