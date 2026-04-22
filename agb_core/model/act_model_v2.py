import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agb_core.data.trajectory import Trajectory
from agb_core.model.base_model import BaseModel


class DecisionEmbeddingLayer(nn.Module):
    """
    决策嵌入层：将 DT 数值数据投影到 LLM embedding 维度

    论文设定：
    - 每个数值元素（RTG/状态/动作）整体投影为一个 embed_dim 维向量
    - 序列结构：{R_0, s_0, a_0, R_1, s_1, a_1, ..., R_{W-1}, s_{W-1}, a_{W-1}}
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
        按 DT 交错结构组织序列: [R_0, s_0, a_0, R_1, s_1, a_1, ..., R_{W-1}, s_{W-1}, a_{W-1}]

        输入形状（train_agent.py 构造后）:
        - states:  [B, W, S]   — 去掉了 next_state，末尾 s_{W-1} 是最后决策步状态
        - actions: [B, W, A]   — 训练时 a_{W-1}=真实值，推理时 a_{W-1}=zero placeholder
        - rtgs:    [B, W, 1]   — 去掉了末尾 rtg，R_{W-1} 是最后有效累积回报

        输出序列共 3W 个 token，末尾为 [R_{W-1}, s_{W-1}, a_{W-1}]。
        拼接顺序为 text-first，整体序列: [text, R_0, s_0, a_0, ..., R_{W-1}, s_{W-1}, a_{W-1}]

        因果性保证（标准 causal mask）：
        - s_{W-1} (pos -2) 无法 attend 到 a_{W-1} (pos -1)，因为 -2 < -1

        Args:
            trajectory: Trajectory namedtuple

        Returns:
            dt_embeddings: [B, 3*W, embed_dim]
        """
        states = trajectory.states
        actions = trajectory.actions
        rtgs = trajectory.rtgs

        if not states.numel() > 0:
            B = states.shape[0]
            return torch.zeros(B, 0, self._embed_dim, device=self._device)

        W = actions.shape[1]  # = states.shape[1] = rtgs.shape[1]

        # [B, W, embed_dim]
        rtg_embedded = self.rtg_mlp(rtgs)       # R_0 .. R_{W-1}
        state_embedded = self.state_mlp(states)  # s_0 .. s_{W-1}
        action_embedded = self.action_mlp(actions)  # a_0 .. a_{W-1}（包含末尾 action）

        # [B, W, 3, embed_dim] -> [B, 3W, embed_dim]
        return torch.stack(
            [rtg_embedded, state_embedded, action_embedded], dim=2
        ).view(states.shape[0], 3 * W, self._embed_dim)


class ActEmbeddingLayer(nn.Module):
    """
    输入 embedding + LLM forward 层。

    接收 text_prompts 和 trajectory，归一化状态，拼接后送入 LLM，
    返回最后一层 hidden state: [B, T+3W, H]。

    拼接顺序（text-first）: [text, R_0, s_0, a_0, ..., R_{W-1}, s_{W-1}, a_{W-1}]
    """

    def __init__(
        self,
        model_path: str,
        state_dim: int,
        action_dim: int,
        device: str = 'cuda',
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._device = device
        self._torch_dtype = torch_dtype

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self._llm = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, device_map=device,
            torch_dtype=torch_dtype,
        )
        self._llm.gradient_checkpointing_enable()
        self._llm.get_input_embeddings().requires_grad_(False)

        self._embedding = DecisionEmbeddingLayer(
            llm_embedding_dim=self._llm.config.hidden_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        self.hidden_size = self._llm.config.hidden_size

        self.register_buffer('_state_mean', torch.zeros(state_dim, dtype=torch.float32))
        self.register_buffer('_state_std', torch.ones(state_dim, dtype=torch.float32))
        self._state_mean = self._state_mean.to(device)
        self._state_std = self._state_std.to(device)

    def set_normalize(self, state_mean: np.ndarray, state_std: np.ndarray) -> 'ActEmbeddingLayer':
        self._state_mean = torch.tensor(state_mean, dtype=torch.float32, device=self._device)
        self._state_std = torch.tensor(state_std, dtype=torch.float32, device=self._device)
        return self

    def forward(self, text_prompts: list[str], trajectory: Trajectory) -> torch.Tensor:
        """
        Args:
            text_prompts: list[str]，每个样本一个 prompt
            trajectory: Trajectory，各字段 shape [B, W, dim]，states 未归一化

        Returns:
            last_hidden_state: [B, T+3W, H]
        """
        # 归一化 states
        states = (trajectory.states - self._state_mean) / (self._state_std + 1e-9)
        trajectory = trajectory._replace(states=states)

        # text embeddings: [B, T, H]
        text_embeds = self._tokenize(text_prompts)
        # decision embeddings: [B, 3W, H]
        dt_embeds = self._embedding(trajectory)
        # text-first 拼接: [B, T+3W, H]
        combined_embeds = torch.cat([text_embeds, dt_embeds], dim=1)

        # LLM forward
        outputs = self._llm(
            inputs_embeds=combined_embeds,
            return_dict=True,
            output_hidden_states=True,
        )
        return outputs.hidden_states[-1]

    def _tokenize(self, texts: list[str]) -> torch.Tensor:
        input_ids = self._tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True,
        ).input_ids.to(self._device)
        token_embeddings = self._llm.get_input_embeddings()(input_ids)
        del input_ids
        return token_embeddings


class ActOutputHead(nn.Module):
    def forward(self, hidden_state: torch.Tensor, W: int):
        raise NotImplementedError


class ActionHead(ActOutputHead):
    """
    从 LLM 隐藏状态预测动作。

    序列结构: [text, R_0, s_0, a_0, ..., R_{W-1}, s_{W-1}, a_{W-1}]
    来自 s_{W-1} 的 hidden state，符合马尔可夫决策（动作由当前状态决定）。
    """

    def __init__(self, hidden_size: int, action_dim: int = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, action_dim),
        )

    def forward(self, hidden_state: torch.Tensor, W: int) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, T+3W, H]，text-first 后的完整 hidden state
            W: window size（此头中未使用，保留接口一致性）

        Returns:
            action: [B, action_dim]
        """
        return self.fc(hidden_state[:, -2, :])


class RTGHead(ActOutputHead):
    """
    从 LLM 隐藏状态预测 RTG。

    序列结构: [text, R_0, s_0, a_0, ..., R_{W-1}, s_{W-1}, a_{W-1}]
    来自 a_{W-1} 的 hidden state，逻辑上像 Reward Model。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_state: torch.Tensor, W: int) -> torch.Tensor:
        """
        Args:
            hidden_state: [B, T+3W, H]，text-first 后的完整 hidden state
            W: window size（此头中未使用，保留接口一致性）

        Returns:
            rtg_pred: [B]
        """
        return self.fc(hidden_state[:, -1, :]).squeeze(-1)


class DTOutputHead(ActOutputHead):
    """
    传统 DT 输出头：对每个 timestep 输出 (state_pred, action_pred, rtg_pred)。

    序列结构: [text, R_0, s_0, a_0, R_1, s_1, a_1, ..., R_{W-1}, s_{W-1}, a_{W-1}]
    dt 段位置: [T, T+3W)，即 dt_offset = T
      - R_t → T + 3t
      - s_t → T + 3t + 1
      - a_t → T + 3t + 2
    """

    def __init__(self, hidden_size: int, state_dim: int, action_dim: int):
        super().__init__()
        self._state_dim = state_dim
        self._action_dim = action_dim

        self.state_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, state_dim),
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, action_dim),
        )
        self.rtg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, hidden_state: torch.Tensor, W: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: [B, T+3W, H]，text-first 后的完整 hidden state
            W: window size

        Returns:
            state_preds:  [B, W, state_dim]
            action_preds: [B, W, action_dim]
            rtg_preds:    [B, W]
        """
        total_len = hidden_state.shape[1]
        dt_offset = total_len - 3 * W  # text 长度 T

        device = hidden_state.device
        rtg_idx = dt_offset + torch.arange(0, 3 * W, 3, device=device)           # [T+0, T+3, T+6, ...]
        state_idx = dt_offset + torch.arange(1, 3 * W, 3, device=device)        # [T+1, T+4, T+7, ...]
        action_idx = dt_offset + torch.arange(2, 3 * W, 3, device=device)       # [T+2, T+5, T+8, ...]

        rtg_hidden = hidden_state[:, rtg_idx, :]         # [B, W, H]
        state_hidden = hidden_state[:, state_idx, :]     # [B, W, H]
        action_hidden = hidden_state[:, action_idx, :]   # [B, W, H]

        state_preds = self.state_head(state_hidden)       # [B, W, S]
        action_preds = self.action_head(action_hidden)    # [B, W, A]
        rtg_preds = self.rtg_head(rtg_hidden).squeeze(-1)  # [B, W]

        return state_preds, action_preds, rtg_preds


class ActModelBase(BaseModel, nn.Module):
    """
    ACT 模型基类：统一 embedding 层和预测接口，各子类通过不同输出头实现差异化行为。

    共享组件：
    - _embedding_layer: ActEmbeddingLayer — embedding + LLM forward，返回 [B, T+3W, H]
    - set_normalize / load_model / predict / predict_batch — 通用逻辑

    子类职责：
    - __init__: 创建各自的输出头
    - _forward_batch: 调用 embedding_layer，应用输出头，返回 tuple
    - _get_action: 从 _forward_batch 结果中提取 action: [B, A]

    序列结构（text-first）: [text, R_0, s_0, a_0, ..., R_{W-1}, s_{W-1}, a_{W-1}]
    """

    def __init__(
        self,
        base_model_path: str,
        model_type: str = 'transformers',
        state_dim: int = 16,
        action_dim: int = 1,
        device: str = 'cuda',
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self._device = device
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._target_rtg = 0.0
        self._scale = 1.0
        self._output_mode = 'pacer'

        if model_type != 'transformers':
            raise ValueError(f'不支持的模型类型: {model_type}')

        self._embedding_layer = ActEmbeddingLayer(
            model_path=base_model_path,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            torch_dtype=torch_dtype,
        )

        self.to(device)

    def set_normalize(self, state_mean: np.ndarray, state_std: np.ndarray) -> 'ActModelBase':
        self._embedding_layer.set_normalize(state_mean, state_std)
        return self

    def load_model(self, model_path: str) -> 'ActModelBase':
        ckpt = torch.load(model_path, map_location=self._device, weights_only=False)
        state_dict = ckpt['model_state_dict']

        # v2 embedding layer 中 _state_mean/_state_std 已是 registered buffer，
        # 从 checkpoint 中移除这两个 key，避免与 buffer 冲突
        state_dict.pop('_state_mean', None)
        state_dict.pop('_state_std', None)

        self.load_state_dict(state_dict)
        return self

    def predict(
        self,
        prompt: str,
        traj: Trajectory,
        context=None,
    ) -> tuple[None, np.ndarray]:
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
        contexts=None,
    ) -> tuple[None, np.ndarray]:
        traj = traj._replace(
            states=torch.from_numpy(traj.states).to(self._device),
            actions=torch.from_numpy(traj.actions).to(self._device),
            rtgs=torch.from_numpy(traj.rtgs).to(self._device),
            timesteps=torch.from_numpy(traj.timesteps).to(self._device),
            attention_mask=torch.from_numpy(traj.attention_mask).to(self._device),
        )

        outputs = self._forward_batch(prompts, traj)
        action = self._get_action(*outputs)
        return None, action.detach().cpu().numpy()

    def _get_action(self, *outputs) -> torch.Tensor:
        """
        从 _forward_batch 结果中提取 action 预测。

        推理时由 predict_batch 调用，输出 [B, A]，作为 agent 的最终决策。
        """
        raise NotImplementedError

    def _forward_batch(self, text_prompts: list[str], trajectory: Trajectory):
        raise NotImplementedError

    def get_loss(self, traj: Trajectory, thoughts: list[str]) -> dict[str, torch.Tensor]:
        """
        训练接口：给定完整轨迹和 thoughts，计算各输出头的 loss。

        训练 vs 推理的核心区别：
        - 推理时（predict_batch）：a_{W-1} 为零 placeholder，LLM 基于 s_{W-1} 做单步决策，
          无需知道 a_{W-1} 的真实值（因为 a_{W-1} 本身就是要预测的）。
        - 训练时（get_loss）：a_{W-1} 使用真实值。在 causal attention 下，a_{W-1} 只能 attend
          到 [text, R_0..a_{W-2}]，不会 attend 到自己，因此不存在信息泄露。
          真正要预测的是 R_{W-1}，而 R_{W-1} 的 label 来自 traj.rtgs[:, -1, 0]。

        子类须在内部完成 target 的构造，不依赖外部传入。
        """
        raise NotImplementedError


class ActModelV1(ActModelBase):
    """
    ACT V1：仅动作输出头，推理时输出 action。

    序列结构（text-first）: [text, R_0, s_0, a_0, ..., R_{W-1}, s_{W-1}, a_{W-1}]
    Action 来自 s_{W-1} (pos -2)，符合马尔可夫决策。
    """

    def __init__(
        self,
        base_model_path: str,
        model_type: str = 'transformers',
        state_dim: int = 16,
        action_dim: int = 1,
        device: str = 'cuda',
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__(base_model_path, model_type, state_dim, action_dim, device, torch_dtype)

        self._action_head = ActionHead(
            hidden_size=self._embedding_layer.hidden_size,
            action_dim=action_dim,
        )
        self.to(device)

    def _forward_batch(self, text_prompts: list[str], trajectory: Trajectory) -> tuple[torch.Tensor]:
        context = torch.no_grad() if not self.training else contextlib.nullcontext()
        with context:
            last_hidden_state = self._embedding_layer(text_prompts, trajectory)
            W = trajectory.actions.shape[1]
            action = self._action_head(last_hidden_state, W)
            return (action,)

    def _get_action(self, action: torch.Tensor) -> torch.Tensor:
        return action

    def get_loss(
        self,
        traj: Trajectory,
        thoughts: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        训练：输入长度为 W 的完整轨迹，shift 后取前 W 步推理，target 为 a_{W-1}。

        traj.actions[:, -1]      — label: [B, A]
        traj_slice.actions[:, :] — 输入序列用真实 a_{W-1}，非 placeholder
        """
        W = traj.actions.shape[1]
        target_actions = traj.actions[:, -1]
        traj_slice = traj._replace(
            states=traj.states[:, :W],
            actions=traj.actions[:, :W],
            rtgs=traj.rtgs[:, :W],
        )
        (action,) = self._forward_batch(thoughts, traj_slice)
        return {'action': F.mse_loss(action, target_actions)}


class ActModelV2(ActModelBase):
    """
    ACT V2：动作输出头 + RTG 辅助输出头。

    Action 来自 s_{W-1} (pos -2)，RTG 来自 a_{W-1} (pos -1)。
    RTG 预测仅用于训练阶段的辅助 loss，推理时 action 输出与 V1 一致。
    """

    def __init__(
        self,
        base_model_path: str,
        model_type: str = 'transformers',
        state_dim: int = 16,
        action_dim: int = 1,
        device: str = 'cuda',
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__(base_model_path, model_type, state_dim, action_dim, device, torch_dtype)

        self._action_head = ActionHead(
            hidden_size=self._embedding_layer.hidden_size,
            action_dim=action_dim,
        )
        self._rtg_head = RTGHead(hidden_size=self._embedding_layer.hidden_size)
        self.to(device)

    def _forward_batch(self, text_prompts: list[str], trajectory: Trajectory) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.no_grad() if not self.training else contextlib.nullcontext()
        with context:
            last_hidden_state = self._embedding_layer(text_prompts, trajectory)
            W = trajectory.actions.shape[1]
            action = self._action_head(last_hidden_state, W)
            rtg_pred = self._rtg_head(last_hidden_state, W)
            return action, rtg_pred

    def _get_action(self, action: torch.Tensor, rtg_pred: torch.Tensor) -> torch.Tensor:
        return action

    def get_loss(
        self,
        traj: Trajectory,
        thoughts: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        训练：输入长度为 W 的完整轨迹，shift 后取前 W 步推理，target 为 a_{W-1} 和 R_{W-1}。

        target_actions = traj.actions[:, -1]  — label: [B, A]
        target_rtgs    = traj.rtgs[:, -1, 0]   — label: [B]，累积回报标签
        traj_slice 中的 a_{W-1} 为真实值，非 placeholder
        """
        W = traj.actions.shape[1]
        target_actions = traj.actions[:, -1]
        target_rtgs = traj.rtgs[:, -1, 0]
        traj_slice = traj._replace(
            states=traj.states[:, :W],
            actions=traj.actions[:, :W],
            rtgs=traj.rtgs[:, :W],
        )
        action, rtg_pred = self._forward_batch(thoughts, traj_slice)
        return {
            'action': F.mse_loss(action, target_actions),
            'rtg': F.mse_loss(rtg_pred, target_rtgs),
        }


class ActModelDT(ActModelBase):
    """
    Decision Transformer 输出头：对每个 timestep 输出 (state_pred, action_pred, rtg_pred)。

    _forward_batch 返回 (state_preds, action_preds, rtg_preds)，shape 均为 [B, W, dim]。
    _get_action 取最后一步 action_preds[:, -1, :]，返回 [B, A]，对应推理时的单步决策。

    序列结构（text-first）: [text, R_0, s_0, a_0, ..., R_{W-1}, s_{W-1}, a_{W-1}]
    """

    def __init__(
        self,
        base_model_path: str,
        model_type: str = 'transformers',
        state_dim: int = 16,
        action_dim: int = 1,
        device: str = 'cuda',
        torch_dtype: torch.dtype | None = None,
    ):
        super().__init__(base_model_path, model_type, state_dim, action_dim, device, torch_dtype)

        self._dt_head = DTOutputHead(
            hidden_size=self._embedding_layer.hidden_size,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        self.to(device)

    def _forward_batch(
        self, text_prompts: list[str], trajectory: Trajectory
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = torch.no_grad() if not self.training else contextlib.nullcontext()
        with context:
            last_hidden_state = self._embedding_layer(text_prompts, trajectory)
            W = trajectory.actions.shape[1]
            state_preds, action_preds, rtg_preds = self._dt_head(last_hidden_state, W)
            return state_preds, action_preds, rtg_preds

    def _get_action(self, state_preds: torch.Tensor, action_preds: torch.Tensor, rtg_preds: torch.Tensor) -> torch.Tensor:
        return action_preds[:, -1, :]

    def get_loss(
        self,
        traj: Trajectory,
        thoughts: list[str],
    ) -> dict[str, torch.Tensor]:
        """
        训练：每个 timestep 同时预测 state/action/rtg，DT 一次性输出全序列。

        输入 shift：
          - traj.states 去掉最后一步 s_{W-1} 作为输入（对应 t=0..W-2）
          - traj.actions 使用完整 W 步作为 target（包含 a_{W-1}）
        无需 placeholder：a_{W-1} 以真实值作为输入，但 R_{W-1} 的 label 来自 traj.rtgs[:, -1, 0]，
        由于 causal attention，a_{W-1} 无法 attend 到 R_{W-1}，不存在信息泄露。

        target_actions = traj.actions[:, :]       — [B, W, A]，全序列 label
        target_states  = traj.states[:, :-1]       — [B, W-1, S]
        target_rtgs    = traj.rtgs[:, :-1, 0]      — [B, W-1]
        """
        target_actions = traj.actions
        target_states = traj.states[:, :-1]
        target_rtgs = traj.rtgs[:, :-1, 0]
        traj_slice = traj._replace(
            states=traj.states[:, :-1],
            actions=traj.actions,
            rtgs=traj.rtgs[:, :-1],
        )
        state_preds, action_preds, rtg_preds = self._forward_batch(thoughts, traj_slice)
        return {
            'action': F.mse_loss(action_preds, target_actions),
            'state': F.mse_loss(state_preds, target_states),
            'rtg': F.mse_loss(rtg_preds, target_rtgs),
        }
