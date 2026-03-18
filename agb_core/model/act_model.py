"""
ActModel 实现

对应原来的 LBMModel，输入 prompt 和 numeral，输出仅有 action。
采用论文方法：使用 MLP 将数值数据投影到 LLM embedding 维度，然后拼接送入 LLM。
"""

from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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
        # 将所有参数移动到指定设备
        self.to(device)

        # 状态 MLP: [state_dim] -> [embed_dim]
        self.state_mlp = nn.Sequential(
            nn.Linear(self._state_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
        )

        # 动作 MLP: [action_dim] -> [embed_dim]
        self.action_mlp = nn.Sequential(
            nn.Linear(self._action_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
        )

        # RTG MLP: [1] -> [embed_dim]
        self.rtg_mlp = nn.Sequential(
            nn.Linear(1, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
            nn.GELU(),
            nn.Linear(self._embed_dim, self._embed_dim),
        )

    def forward(self, dt_tuple: Tuple) -> torch.Tensor:
        """
        按照论文结构: {R_{t-L}, s_{t-L}, a_{t-L}, ..., R_t, s_t}
        每个元素（RTG/状态/动作）整体投影为一个 token

        Args:
            dt_tuple: (states, actions, rtgs, timesteps, attention_mask)
                states: [T, state_dim]
                actions: [T, 1]
                rtgs: [T+1, 1]
                timesteps: [T]
                attention_mask: [T]

        Returns:
            dt_embeddings: [1, seq_len, embed_dim]
        """
        states, actions, rtgs, timesteps, attention_mask = dt_tuple

        # 转换为 tensor 并移动到设备上
        states = torch.from_numpy(states).float().to(self._device)
        actions = torch.from_numpy(actions).float().to(self._device)
        rtgs = torch.from_numpy(rtgs).float().to(self._device)

        embeddings_list = []

        # 1. 投影 rtgs (return-to-go): [T+1, 1] -> [T+1, 1, embed_dim]
        # 保持 T+1 个 RTG
        if rtgs.shape[0] > 0:
            rtg_embedded = self.rtg_mlp(rtgs)  # [T+1, 1] -> [T+1, embed_dim]
            rtg_embedded = rtg_embedded.unsqueeze(1)  # [T+1, 1, embed_dim]
            embeddings_list.append(rtg_embedded)

        # 2. 投影 states: [T, state_dim] -> [T, 1, embed_dim]
        # 每个时间步的状态作为一个 token（整体投影，不是打散成 state_dim 个）
        if states.numel() > 0:
            states_embedded = self.state_mlp(states)  # [T, state_dim] -> [T, embed_dim]
            states_embedded = states_embedded.unsqueeze(1)  # [T, 1, embed_dim]
            embeddings_list.append(states_embedded)

        # 3. 投影 actions: [T, 1] -> [T, 1, embed_dim]
        if actions.numel() > 0:
            actions_embedded = self.action_mlp(actions)  # [T, 1] -> [T, embed_dim]
            actions_embedded = actions_embedded.unsqueeze(1)  # [T, 1, embed_dim]
            embeddings_list.append(actions_embedded)

        # 在序列维度拼接: [T+1, 3, embed_dim]（RTG有T+1个，state和action各T个）
        if embeddings_list:
            dt_embeddings = torch.cat(embeddings_list, dim=1)
            # 压缩序列维度: [T+1, 3, embed_dim] -> [1, (T+1)*3, embed_dim]
            T_plus_1 = dt_embeddings.shape[0]
            dt_embeddings = dt_embeddings.view(1, T_plus_1 * 3, self._embed_dim)
        else:
            dt_embeddings = torch.zeros(1, 0, self._embed_dim, device=self._device)

        return dt_embeddings  # [1, seq_len, embed_dim]


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
        # 取最后一个有效位置的隐藏状态
        return self.fc(last_hidden_state[:, -1, :])


class ActModel(BaseModel, nn.Module):
    """
    Act Model - 动作模型

    对应原来的 LBMModel。
    输入 prompt 和 numeral，输出仅有 action（response 为 None）。

    双输入：
    - prompt: str, 构建好的文本 prompt
    - numeral: 二元组 (context_dict, dt_tuple)

    双输出：
    - response: None
    - action: 预测的 pacer 值

    融合方式（参考论文）：
    1. 文本 prompt -> Tokenizer -> Token Embeddings
    2. DT tuple -> Decision Embedding Layer -> DT Embeddings
    3. 序列维度拼接后送入 LLM
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
        # DT 归一化参数
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
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

        # 初始化归一化参数（保持在 CPU 上用 numpy 操作，避免设备转换开销）
        if state_mean is None:
            state_mean = np.zeros(state_dim, dtype=np.float32)
        if state_std is None:
            state_std = np.ones(state_dim, dtype=np.float32)
        self._state_mean = torch.from_numpy(state_mean.astype(np.float32)).to(device)
        self._state_std = torch.from_numpy(state_std.astype(np.float32)).to(device)

        # 占位符属性，与策略兼容
        self._target_rtg = 0.0
        self._scale = 1.0

        # 加载 tokenizer
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # 获取 LLM backbone
        from transformers import AutoModelForCausalLM
        self._llm = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map=device,
        )
        self._llm.eval()

        # 从 LLM 配置中获取 embedding 维度
        llm_hidden_size = self._llm.config.hidden_size

        # 禁用 LLM 的梯度计算
        for param in self._llm.parameters():
            param.requires_grad = False

        # 构建 Decision Embedding Layer
        self._decision_embedding = DecisionEmbeddingLayer(
            llm_embedding_dim=llm_hidden_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        # 构建 Action Head
        self._action_head = ActionHead(
            hidden_size=llm_hidden_size,
            action_dim=action_dim,
        )

        # 将所有模块移动到指定设备
        self._decision_embedding.to(device)
        self._action_head.to(device)

    def predict(
        self,
        prompt: Optional[str],
        numeral: Optional[Any] = None
    ) -> tuple[Optional[str], Optional[Any]]:
        """
        根据文本 prompt 和 numeral 预测动作

        Args:
            prompt: 文本 prompt（忽略）
            numeral: 二元组 (context_dict, dt_tuple)
                - context_dict: 原始 dict（用于兼容接口，ActModel 不使用）
                - dt_tuple: (states, actions, rtgs, timesteps, attention_mask)

        Returns:
            (None, action): response 为 None，action 是预测的 pacer 值
        """
        if numeral is None:
            raise ValueError("ActModel requires numeral input")

        text_prompt = prompt if prompt is not None else ""

        # 解包二元组 (context_dict, dt_input)
        context_dict, dt_tuple = numeral

        # 归一化 states
        states = dt_tuple[0]
        states = (states - self._state_mean.cpu().numpy()) / (self._state_std.cpu().numpy() + 1e-9)
        dt_tuple = (states, dt_tuple[1], dt_tuple[2], dt_tuple[3], dt_tuple[4])

        # 前向传播
        action = self._forward(text_prompt, dt_tuple)
        # 确保返回 numpy array
        action = action.detach().cpu().numpy()
        return None, action

    def _forward(self, text_prompt: str, dt_tuple: Tuple) -> torch.Tensor:
        """
        前向传播

        Args:
            text_prompt: str
            dt_tuple: (states, actions, rtgs, timesteps, attention_mask)

        Returns:
            action: tensor
        """
        # 1. 文本 -> Token Embeddings
        text_embeds = self._tokenize(text_prompt)

        # 2. DT tuple -> Decision Embeddings
        dt_embeds = self._decision_embedding(dt_tuple)

        # 3. 序列维度拼接
        combined_embeds = torch.cat([text_embeds, dt_embeds], dim=1)

        # 4. 通过 LLM 获取 hidden states
        with torch.no_grad():
            outputs = self._llm(
                inputs_embeds=combined_embeds,
                return_dict=True,
                output_hidden_states=True,
            )

        # 5. 从最后一层 hidden states 解码动作
        last_hidden_state = outputs.hidden_states[-1]
        action = self._action_head(last_hidden_state)

        return action[0, 0]  # 返回标量

    def _tokenize(self, text: str) -> torch.Tensor:
        """
        将文本 prompt 转换为 token embeddings

        Args:
            text: str

        Returns:
            embeddings: [1, seq_len, embed_dim]
        """
        # 使用 tokenizer 获取 input_ids
        inputs = self._tokenizer(text, return_tensors='pt', padding=True)
        input_ids = inputs.input_ids.to(self._device)

        # 获取 token embeddings
        token_embeddings = self._llm.get_input_embeddings()(input_ids)

        return token_embeddings

    def get_text_response(self, text_prompt: str, dt_tuple: Tuple) -> str:
        """
        获取 LLM 的文本响应（用于推理过程）

        Args:
            text_prompt: str
            numeral: 二元组 (context_dict, dt_input)

        Returns:
            text_response: str
        """
        # 简化的文本生成（不实际调用，用于 debug）
        return ""
