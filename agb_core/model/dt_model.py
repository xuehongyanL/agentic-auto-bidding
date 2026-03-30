"""
DT Model 实现

基于 GAVE/Decision Transformer 的出价模型。
"""

import math
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agb_core.data.trajectory import Trajectory
from agb_core.model.base_model import BaseModel, DecisionModel


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config['n_embd'] % config['n_head'] == 0
        self.key = nn.Linear(config['n_embd'], config['n_embd'])
        self.query = nn.Linear(config['n_embd'], config['n_embd'])
        self.value = nn.Linear(config['n_embd'], config['n_embd'])
        self.attn_drop = nn.Dropout(config['attn_pdrop'])
        self.resid_drop = nn.Dropout(config['resid_pdrop'])
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config['n_ctx'], config['n_ctx'])).view(
                1, 1, config['n_ctx'], config['n_ctx']
            )
        )
        self.register_buffer('masked_bias', torch.tensor(-1e4))
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.n_head = config['n_head']

    def forward(self, x, mask):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        mask = mask.view(B, -1)
        mask = mask[:, None, None, :]
        mask = (1.0 - mask) * -10000.0
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.where(self.bias[:, :, :T, :T].bool(), att, self.masked_bias.to(att.dtype)) # type: ignore
        att = att + mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config['n_embd'], config['n_inner']),
            nn.GELU(),
            nn.Dropout(config['resid_pdrop']),
            nn.Linear(config['n_inner'], config['n_embd']),
        )

    def forward(self, inputsembeds, attention_mask):
        x = inputsembeds + self.attn(self.ln1(inputsembeds), attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DTModel(DecisionModel, nn.Module):
    """
    Decision Transformer 模型

    基于 GAVE 架构的出价模型，根据状态序列预测 pacer。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        target_rtg: float,
        hidden_size: int,
        n_layer: int,
        n_head: int,
        n_inner: int,
        scale: float,
        block_config: dict,
        output_mode: str,
        max_timestep_len: int,
    ):
        """
        初始化 DT Model

        Args:
            model_path: 预训练模型路径
            state_dim: 状态维度
            action_dim: 动作维度
            device: 设备
            target_rtg: 目标返回值
            hidden_size: 隐藏层维度
            n_layer: Transformer层数
            n_head: 注意力头数
            n_inner: FFN中间层维度
            scale: rtg缩放因子，用于与GAVE对齐
        """
        self._scale = scale
        self._device = device
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._output_mode = output_mode
        self._target_rtg = target_rtg
        self._hidden_size = hidden_size
        self._n_layer = n_layer
        self._n_head = n_head
        self._n_inner = n_inner
        self._block_config = block_config
        self._max_timestep_len = max_timestep_len

        super().__init__()

        # 注册为 buffer（无梯度，随 model.state_dict() 自动导出/导入）
        self.register_buffer('_state_mean', torch.zeros(state_dim, dtype=torch.float32))
        self.register_buffer('_state_std', torch.ones(state_dim, dtype=torch.float32))

        self._build_model()

        self.to(device)

    def _build_model(self):
        """构建模型结构"""
        self._time_dim = 8
        self.transformer = nn.ModuleList([Block(self._block_config) for _ in range(self._block_config['n_layer'])])
        self.embed_timestep = nn.Embedding(self._max_timestep_len, self._time_dim)
        self.embed_rtg = nn.Linear(1, self._hidden_size)
        self.embed_state = nn.Linear(self._state_dim, self._hidden_size)
        self.embed_action = nn.Linear(self._action_dim, self._hidden_size)
        self.trans_rtg = nn.Linear(self._time_dim + self._hidden_size, self._hidden_size)
        self.trans_state = nn.Linear(self._time_dim + self._hidden_size, self._hidden_size)
        self.trans_action = nn.Linear(self._time_dim + self._hidden_size, self._hidden_size)
        self.embed_ln = nn.LayerNorm(self._hidden_size)
        self.predict_action = nn.Sequential(
            nn.Linear(self._hidden_size, self._action_dim),
        )
        self.predict_state = nn.Linear(self._hidden_size, self._state_dim)
        self.predict_rtg = nn.Sequential(
            nn.Linear(self._hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.predict_value = nn.Sequential(
            nn.Linear(self._hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.predict_beta = nn.Sequential(
            nn.Linear(self._hidden_size, 16),
            nn.GELU(),
            nn.Linear(16, 8),
            nn.GELU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def load_model(self, model_path: str) -> 'DTModel':
        """加载预训练模型"""
        state_dict_raw = torch.load(model_path, map_location=self._device, weights_only=False)

        # 向后兼容：旧模型的 key 映射到新模型
        key_to_map = {
            'embed_return.weight': 'embed_rtg.weight',
            'embed_return.bias': 'embed_rtg.bias',
            'trans_return.weight': 'trans_rtg.weight',
            'trans_return.bias': 'trans_rtg.bias',
            'predict_return.0.weight': 'predict_rtg.0.weight',
            'predict_return.0.bias': 'predict_rtg.0.bias',
            'predict_return.2.weight': 'predict_rtg.2.weight',
            'predict_return.2.bias': 'predict_rtg.2.bias',
            'predict_return.4.weight': 'predict_rtg.4.weight',
            'predict_return.4.bias': 'predict_rtg.4.bias',
        }
        key_to_delete = {
            'embed_reward.weight', 'embed_reward.bias',
            'trans_reward.weight', 'trans_reward.bias',
        }
        state_dict: dict[str, Any] = {}
        for k, v in state_dict_raw.items():
            if k in key_to_map:
                state_dict[key_to_map[k]] = v
            elif k not in key_to_delete:
                state_dict[k] = v

        # 归一化参数（向后兼容：旧 checkpoint 没有则保留 __init__ 的默认值）
        if '_state_mean' not in state_dict:
            state_dict['_state_mean'] = self._state_mean
        if '_state_std' not in state_dict:
            state_dict['_state_std'] = self._state_std

        self.load_state_dict(state_dict)
        self.eval()
        return self

    def set_normalize(self, state_mean: np.ndarray, state_std: np.ndarray) -> 'DTModel':
        """
        设置归一化参数（运行时更新或向后兼容旧 checkpoint）。

        Args:
            state_mean: 状态均值向量
            state_std: 状态标准差向量
        """
        self._state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self._device)
        self._state_std = torch.tensor(state_std, dtype=torch.float32, device=self._device)
        return self

    def predict(
        self,
        traj: Trajectory,
        prompt = None,
        context = None,
    ) -> tuple[None, np.ndarray]:
        """
        根据历史序列预测动作

        Args:
            prompt: 忽略此参数（保留接口兼容性）
            context: 忽略此参数（保留接口兼容性）
            traj: Trajectory

        Returns:
            (None, action): response 为 None，action 是出价系数
        """
        traj = traj._replace(
            states=np.expand_dims(traj.states, axis=0),
            actions=np.expand_dims(traj.actions, axis=0),
            rtgs=np.expand_dims(traj.rtgs, axis=0),
            timesteps=np.expand_dims(traj.timesteps, axis=0),
            attention_mask=np.expand_dims(traj.attention_mask, axis=0),
        )
        action = self.predict_batch(traj)[1][0]
        return None, action

    def predict_batch(
        self,
        traj: Trajectory,
        prompts = None,
        contexts = None,
    ) -> tuple[None, np.ndarray]:
        action = self._get_action(traj)
        action = action.detach().cpu().numpy()
        return None, action

    def _get_action(self, trajectory: Trajectory):
        # 归一化 states（z-score）；buffer 已在模型设备上，无需额外移动
        raw_states = torch.from_numpy(trajectory.states).to(self._device)
        states = (raw_states - self._state_mean) / (self._state_std + 1e-9)
        actions = torch.from_numpy(trajectory.actions).to(self._device)
        rtgs = torch.from_numpy(trajectory.rtgs).to(self._device)
        timesteps = torch.from_numpy(trajectory.timesteps).to(self._device)
        attention_mask = torch.from_numpy(trajectory.attention_mask).to(self._device)

        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)
        rtgs = rtgs.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        batch_size = 1
        seq_length = states.shape[1]

        # print(states, actions, rtgs, timesteps, attention_mask)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        rtg_embeddings = self.embed_rtg(rtgs)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = torch.cat((state_embeddings, time_embeddings), dim=-1)
        action_embeddings = torch.cat((action_embeddings, time_embeddings), dim=-1)
        rtg_embeddings = torch.cat((rtg_embeddings, time_embeddings), dim=-1)

        state_embeddings = self.trans_state(state_embeddings)
        action_embeddings = self.trans_action(action_embeddings)
        rtg_embeddings = self.trans_rtg(rtg_embeddings)

        stacked_inputs = torch.stack(
            (rtg_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length, self._hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # 通过 torch.max 现场计算 timestep_mask (batch, seq_len)
        timestep_mask = torch.max(attention_mask, dim=-1)[0]
        # 扩展到 3 份 (rtgs, states, actions)
        stacked_attention_mask = torch.stack(
            ([timestep_mask for _ in range(3)]), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length).to(stacked_inputs.dtype)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.reshape(-1, seq_length, 3, self._hidden_size).permute(0, 2, 1, 3)
        action_preds = self.predict_action(x[:, 1])
        return action_preds[0, -1].detach()
