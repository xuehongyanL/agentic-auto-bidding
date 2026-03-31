import contextlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from agb_core.data.trajectory import Trajectory
from agb_core.model.base_model import BaseModel


class DecisionEmbeddingLayer(nn.Module):
    """
    决策嵌入层：将 DT 数值数据投影到 LLM embedding 维度

    论文设定：
    - 每个数值元素（RTG/状态/动作）整体投影为一个 embed_dim 维向量
    - 序列结构：{R_{t-L}, s_{t-L}, a_{t-L}, ..., R_t, s_t}
    """

    def __init__(self, llm_embedding_dim: int = 896, state_dim: int = 16, action_dim: int = 1, device: str = 'cuda'):
        super().__init__()
        self._embed_dim = llm_embedding_dim
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self.to(device)

        self.state_mlp = nn.Sequential(
            nn.Linear(self._state_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(self._action_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
        )

        self.rtg_mlp = nn.Sequential(
            nn.Linear(1, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
        )

    def forward(self, trajectory: Trajectory) -> torch.Tensor:
        """
        按照论文结构: {R_{t-L}, s_{t-L}, a_{t-L}, ..., R_t, s_t}
        每个元素（RTG/状态/动作）整体投影为一个 token

        Args:
            trajectory: Trajectory namedtuple，各字段 shape [B, T, dim]

        Returns:
            dt_embeddings: [B, seq_len, embed_dim]
        """
        states = trajectory.states
        actions = trajectory.actions
        rtgs = trajectory.rtgs

        embeddings_list = []

        if rtgs.shape[1] > 0:
            rtg_embedded = self.rtg_mlp(rtgs)
            rtg_embedded = rtg_embedded.unsqueeze(2)
            embeddings_list.append(rtg_embedded)

        if states.numel() > 0:
            states_embedded = self.state_mlp(states)
            states_embedded = states_embedded.unsqueeze(2)
            embeddings_list.append(states_embedded)

        if actions.numel() > 0:
            actions_embedded = self.action_mlp(actions)
            actions_embedded = actions_embedded.unsqueeze(2)
            embeddings_list.append(actions_embedded)

        if embeddings_list:
            dt_embeddings = torch.cat(embeddings_list, dim=2)
            B, T_p1, _, E = dt_embeddings.shape
            dt_embeddings = dt_embeddings.view(B, T_p1 * 3, E)
        else:
            B = states.shape[0]
            dt_embeddings = torch.zeros(B, 0, self._embed_dim, device=self._device)

        return dt_embeddings


class ActionHead(nn.Module):
    """从 LLM 隐藏状态预测动作"""

    def __init__(self, hidden_size: int, action_dim: int = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

    def forward(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            last_hidden_state: [batch, seq_len, hidden_size]

        Returns:
            action: [batch, action_dim]
        """
        return self.fc(last_hidden_state[:, -1, :])


class ActModel(BaseModel, nn.Module):
    """
    Act Model - 动作模型

    输入 prompt 和 traj，输出仅有 action（response 为 None）。

    双输入：
    - prompt: str, 构建好的文本 prompt
    - traj: Trajectory

    双输出：
    - response: None
    - action: 预测的 pacer 值
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = 'transformers',
        state_dim: int = 16,
        action_dim: int = 1,
        device: str = 'cuda',
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__()
        self._model_path = model_path
        self._model_type = model_type
        self._device = device
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._output_mode = 'pacer'

        self.register_buffer('_state_mean', torch.zeros(state_dim, dtype=torch.float32))
        self.register_buffer('_state_std', torch.ones(state_dim, dtype=torch.float32))

        self._target_rtg = 0.0
        self._scale = 1.0

        if model_type != 'transformers':
            raise ValueError(f'不支持的模型类型: {model_type}')
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map=device)
            self._llm.gradient_checkpointing_enable()
            self._llm.get_input_embeddings().requires_grad_(False)

        self._state_mean = self._state_mean.to(device)
        self._state_std = self._state_std.to(device)

        llm_hidden_size = self._llm.config.hidden_size

        self._decision_embedding = DecisionEmbeddingLayer(
            llm_embedding_dim=llm_hidden_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        self._action_head = ActionHead(
            hidden_size=llm_hidden_size,
            action_dim=action_dim,
        )

        self._decision_embedding.to(device)
        self._action_head.to(device)

    def set_normalize(self, state_mean: np.ndarray, state_std: np.ndarray) -> 'ActModel':
        self._state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self._device)
        self._state_std = torch.tensor(state_std, dtype=torch.float32, device=self._device)
        return self

    def load_model(self, model_path: str) -> 'ActModel':
        ckpt = torch.load(model_path, map_location=self._device, weights_only=False)
        state_dict = ckpt['model_state_dict']

        if '_state_mean' not in state_dict:
            state_dict['_state_mean'] = self._state_mean.to(self._device)
        if '_state_std' not in state_dict:
            state_dict['_state_std'] = self._state_std.to(self._device)

        self.load_state_dict(state_dict)
        return self

    def predict(
        self,
        prompt: str,
        traj: Trajectory,
        context = None,
    ) -> tuple[None, np.ndarray]:
        """
        单样本预测。

        Args:
            prompt: 文本 prompt
            context: 忽略此参数（保留接口兼容性）
            traj: Trajectory
        """
        traj = traj._replace(
            states=np.expand_dims(traj.states, axis=0),
            actions=np.expand_dims(traj.actions, axis=0),
            rtgs=np.expand_dims(traj.rtgs, axis=0),
            timesteps=np.expand_dims(traj.timesteps, axis=0),
            attention_mask=np.expand_dims(traj.attention_mask, axis=0),
        )
        _, action = self.predict_batch(prompts=[prompt], contexts=None, traj=traj)
        return None, action[0]

    def predict_batch(
        self,
        prompts: list[str],
        traj: Trajectory,
        contexts = None,
    ) -> tuple[None, np.ndarray]:
        """
        批量预测。

        Args:
            prompts: list of text prompts
            contexts: 忽略此参数（保留接口兼容性）
            traj: batched Trajectory
        """
        traj = traj._replace(
            states=torch.from_numpy(traj.states).to(self._device),
            actions=torch.from_numpy(traj.actions).to(self._device),
            rtgs=torch.from_numpy(traj.rtgs).to(self._device),
            timesteps=torch.from_numpy(traj.timesteps).to(self._device),
            attention_mask=torch.from_numpy(traj.attention_mask).to(self._device),
        )

        state_mean = self._state_mean
        state_std = self._state_std
        states = (traj.states - state_mean) / (state_std + 1e-9)
        traj = traj._replace(states=states)

        action = self._forward_batch(prompts, traj)
        return None, action.detach().cpu().numpy()

    def _forward_batch(self, text_prompts: list[str], trajectory: Trajectory) -> torch.Tensor:
        """
        批量前向传播

        Args:
            text_prompts: list[str]，每个样本一个 prompt
            trajectory: Trajectory namedtuple，各字段 shape [B, T, dim]

        Returns:
            action: [B]
        """
        context = torch.no_grad() if not self.training else contextlib.nullcontext()
        with context:
            text_embeds = self._tokenize_batch(text_prompts)
            dt_embeds = self._decision_embedding(trajectory)
            combined_embeds = torch.cat([text_embeds, dt_embeds], dim=1)
            outputs = self._llm(
                inputs_embeds=combined_embeds,
                return_dict=True,
                output_hidden_states=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
            action = self._action_head(last_hidden_state)
            return action

    def _tokenize_batch(self, texts: list[str]) -> torch.Tensor:
        """
        将文本 prompt 转换为 token embeddings

        Args:
            texts: list[str]

        Returns:
            embeddings: [B, seq_len, embed_dim]
        """
        input_ids = self._tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True,
        ).input_ids.to(self._device)
        token_embeddings = self._llm.get_input_embeddings()(input_ids)
        del input_ids
        return token_embeddings
