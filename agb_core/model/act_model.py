import contextlib

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

        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_embedding_dim),
        )

        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_embedding_dim),
        )

        self.rtg_mlp = nn.Sequential(
            nn.Linear(1, llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_embedding_dim),
            nn.GELU(),
            nn.Linear(llm_embedding_dim, llm_embedding_dim),
        )

        self.to(device)

    def forward(self, trajectory: Trajectory) -> torch.Tensor:
        """
        按 DT 交错结构组织序列: [R_0, s_0, a_0, R_1, s_1, a_1, ..., R_{W-1}, s_{W-1}]

        输入形状（train_agent.py 构造后）:
        - states:  [B, W, S]   — 去掉了 next_state，末尾 s_{W-1} 是最后决策步状态
        - actions: [B, W, A]   — 末尾 a_{W-1} 是 placeholder（待预测位）
        - rtgs:    [B, W, 1]   — 去掉了末尾 rtg，R_{W-1} 是最后有效累积回报

        去掉 placeholder action 后，末尾为 [R_{W-1}, s_{W-1}]，共 3W-1 个 token。

        标准 causal mask 自动保证:
        - s_t (pos 3t+1) 无法 attend 到 a_t (pos 3t+2)，因为 3t+1 < 3t+2
        - a_t (pos 3t+2) 无法 attend 到 s_{t+1} (pos 3t+4)，因为 3t+2 < 3t+4

        Args:
            trajectory: Trajectory namedtuple

        Returns:
            dt_embeddings: [B, 3*W - 1, embed_dim]
        """
        states = trajectory.states
        actions = trajectory.actions
        rtgs = trajectory.rtgs

        if not states.numel() > 0:
            B = states.shape[0]
            return torch.zeros(B, 0, self._embed_dim, device=self._device)

        W = actions.shape[1]  # = states.shape[1] = rtgs.shape[1]

        # [B, W, embed_dim]
        rtg_embedded = self.rtg_mlp(rtgs)           # R_0 .. R_{W-1}
        state_embedded = self.state_mlp(states)     # s_0 .. s_{W-1}
        action_embedded = self.action_mlp(actions[:, :-1])  # a_0 .. a_{W-2}（去掉 placeholder）

        # [B, W-1, 3, embed_dim] -> [B, 3*(W-1), embed_dim]
        interleaved = torch.stack(
            [rtg_embedded[:, :-1], state_embedded[:, :-1], action_embedded], dim=2
        ).view(states.shape[0], 3 * (W - 1), self._embed_dim)

        # 追加末尾的 [R_{W-1}, s_{W-1}]（无对应 action）
        tail = torch.cat(
            [rtg_embedded[:, -1:], state_embedded[:, -1:]], dim=1
        )  # [B, 2, embed_dim]

        return torch.cat([interleaved, tail], dim=1)  # [B, 3W - 1, embed_dim]


class ActionHead(nn.Module):
    """
    从 LLM 隐藏状态预测动作。

    交错序列结构: [R_0, s_0, a_0, R_1, s_1, a_1, ..., R_{W-1}, s_{W-1}] (共 3W-1)
    - 无 action token，末尾 token 为 s_{W-1}（最后决策步状态）
    - s_{W-1} 隐含了预测 a_{W-1} 所需的完整上下文
    """

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
        base_model_path: str,
        model_type: str = 'transformers',
        state_dim: int = 16,
        action_dim: int = 1,
        device: str = 'cuda',
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        super().__init__()
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
            self._tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
            self._llm = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, device_map=device)
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

        因果性保证（仅需拼接顺序，无需额外 position_ids）：
        - torch.cat([dt_embeds, text_embeds]) 后：
        -   dt 位置 [0, 1, ..., 3W-2]  <  text 位置 [3W-1, ...]
        -   Causal mask: 位置 i 只能 attend 到 j < i
        -   dt token (pos 3W-2): j < 3W-2 → 不包含任何 text 位置 → dt 无法 attend 到 text
        -   text token (pos 3W-1): j < 3W-1 → 包含所有 dt 位置 → text 可以 attend 到 dt
        - 结果: 单向 cross-attention: dt → text，text 无法回看 dt

        Args:
            text_prompts: list[str]，每个样本一个 prompt
            trajectory: Trajectory namedtuple，各字段 shape [B, T, dim]

        Returns:
            action: [B]
        """
        context = torch.no_grad() if not self.training else contextlib.nullcontext()
        with context:
            text_embeds = self._tokenize_batch(text_prompts)  # [B, L, E]
            dt_embeds = self._decision_embedding(trajectory)    # [B, 3W-1, E]
            # dt 在前 text 在后 → 因果 mask 自动保证单向 cross-attention
            combined_embeds = torch.cat([dt_embeds, text_embeds], dim=1)
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
