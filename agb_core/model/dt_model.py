"""
DT Model 实现

基于 GAVE/Decision Transformer 的出价模型。
"""

import math
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agb_core.model.base_model import BaseModel


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


class DTModel(BaseModel, nn.Module):
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
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
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
            state_mean: 状态均值，用于归一化
            state_std: 状态标准差，用于归一化
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

        if state_mean is None:
            state_mean = np.zeros(state_dim, dtype=np.float32)
        if state_std is None:
            state_std = np.ones(state_dim, dtype=np.float32)
        self._state_mean = torch.from_numpy(state_mean.astype(np.float32)).to(device)
        self._state_std = torch.from_numpy(state_std.astype(np.float32)).to(device)

        super().__init__()

        self._build_model()

        self.to(device)

    def _build_model(self):
        """构建模型结构"""
        self._time_dim = 8
        self.transformer = nn.ModuleList([Block(self._block_config) for _ in range(self._block_config['n_layer'])])
        self.embed_timestep = nn.Embedding(96, self._time_dim)
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

        self.load_state_dict(state_dict)
        self.eval()
        return self

    def predict(
        self,
        prompt: Optional[str],
        numeral: Optional[Any] = None
    ) -> Tuple[Optional[str], Optional[Any]]:
        """
        根据历史序列预测动作

        Args:
            prompt: 忽略此参数（保留接口兼容性）
            numeral: 二元组 (context_dict, dt_tuple)

        Returns:
            (None, action): response 为 None，action 是出价系数
        """
        if numeral is None:
            raise ValueError("DTModel requires numeral input")

        _, (states, actions, rtgs, timesteps, attention_mask) = numeral
        action = self._get_action(states, actions, rtgs, timesteps, attention_mask)
        # 确保返回 numpy array
        action = action.detach().cpu().numpy()
        return None, action

    def _get_action(self, states, actions, rtgs, timesteps, attention_mask):
        states = torch.from_numpy(states).to(self._device)
        actions = torch.from_numpy(actions).to(self._device)
        rtgs = torch.from_numpy(rtgs).to(self._device)
        timesteps = torch.from_numpy(timesteps).to(self._device)
        attention_mask = torch.from_numpy(attention_mask).to(self._device)

        states = states.unsqueeze(0)
        actions = actions.unsqueeze(0)
        rtgs = rtgs.unsqueeze(0)
        timesteps = timesteps.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)

        batch_size = 1
        seq_length = states.shape[1]


        states = torch.where(attention_mask.view(-1, 1) == 1,
                             (states - self._state_mean) / (self._state_std + 1e-9),
                             states)

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

        if attention_mask is not None:
            stacked_attention_mask = torch.stack(
                ([attention_mask for _ in range(3)]), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length).to(stacked_inputs.dtype)
        else:
            stacked_attention_mask = torch.ones(batch_size, 3 * seq_length, dtype=torch.long, device=stacked_inputs.device)

        x = stacked_inputs
        for block in self.transformer:
            x = block(x, stacked_attention_mask)

        x = x.reshape(-1, seq_length, 3, self._hidden_size).permute(0, 2, 1, 3)
        action_preds = self.predict_action(x[:, 1])
        return action_preds[0, -1].detach()
