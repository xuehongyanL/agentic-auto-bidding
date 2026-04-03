"""
AgentModel 实现

组合 Think 和 Act 两个子模型：
1. Think 模型：输入 context 和 traj，输出 response
2. Act 模型：输入 prompt 和 traj，输出 action
"""

from typing import Any

import numpy as np

from agb_core.data.trajectory import Trajectory
from agb_core.model.act_model import ActModel
from agb_core.model.base_model import DecisionModel
from agb_core.model.think_model import ThinkModel


class AgentModel(DecisionModel):
    """
    Agent Model - Agent 模型

    组合 Think 和 Act 两个子模型：
    1. Think 模型：输入 context 和 traj，输出 response
    2. Act 模型：输入 prompt 和 traj，输出 action

    双输入（来自 numeral 拆分）：
    - context: context_dict
    - traj: Trajectory
    """

    def __init__(
        self,
        think_model: ThinkModel,
        act_model: ActModel,
        think_batch_size: int = 1,
        act_batch_size: int = 1,
    ):
        """
        初始化 Agent Model

        Args:
            think_model: Think 子模型，负责生成文本推理
            act_model: Act 子模型，负责输出动作
            think_batch_size: Think 阶段的批大小，控制 LLM 并发显存峰值
            act_batch_size: Act 阶段的批大小，控制 LLM 并发显存峰值
        """
        self._think_model = think_model
        self._act_model = act_model
        self._think_batch_size = think_batch_size
        self._act_batch_size = act_batch_size

        self._target_rtg = getattr(think_model, '_target_rtg', 0.0)
        self._scale = getattr(think_model, '_scale', 1.0)
        self._state_dim = act_model._state_dim
        self._action_dim = act_model._action_dim
        self._output_mode = act_model._output_mode

    def predict(
        self,
        context: dict,
        traj: Trajectory,
        prompt = None,
        skip_think: bool = False,
    ) -> tuple[str, np.ndarray]:
        """
        两阶段预测：
        1. 调用 Think 模型获取 response
        2. 将 response 作为 prompt，与 traj 一起传给 Act 模型获取 action

        Args:
            prompt: 忽略此参数（保留接口兼容性）
            context: context_dict，供 Think 模型使用
            traj: Trajectory，供 Think 和 Act 模型使用
            skip_think: 若为 True，跳过 Think 模型推理，response 用空字符串代替

        Returns:
            (response, action): response 是 Think 模型的文本响应，action 是 Act 模型预测的动作
        """
        # 第一步：调用 Think 模型获取 response
        if skip_think:
            think_response = ''
        else:
            think_response, _ = self._think_model.predict(context=context, traj=traj)

        # 第二步：将 response 作为 prompt，与 traj 一起传给 Act 模型
        _, action = self._act_model.predict(prompt=think_response, traj=traj)

        return think_response, action

    def predict_batch(
        self,
        contexts: list[dict],
        traj: Trajectory,
        prompts = None,
        skip_think: bool = False,
    ) -> tuple[list[str], list[Any]]:
        """
        批量预测：内部分块执行以控制 LLM 并发显存峰值。

        Args:
            prompts: 忽略此参数（保留接口兼容性）
            contexts: list of context_dicts，供 Think 模型使用
            traj: batched Trajectory，供 Think 和 Act 模型使用
            skip_think: 若为 True，跳过 Think 模型推理，responses 用空字符串代替

        Returns:
            (responses, actions):
                responses: list of Think 模型的文本响应
                actions: list of numpy arrays
        """
        return self.predict_batch_chunked(
            contexts=contexts,
            traj=traj,
            think_batch_size=self._think_batch_size,
            act_batch_size=self._act_batch_size,
            skip_think=skip_think,
        )

    def predict_batch_chunked(
        self,
        contexts: list[dict],
        traj: Trajectory,
        prompts = None,
        think_batch_size: int = 1,
        act_batch_size: int = 1,
        skip_think: bool = False,
    ) -> tuple[list[str], list[Any]]:
        """
        分块批量预测：think 和 act 分别按各自的批大小分块执行。

        Args:
            prompts: 忽略此参数（保留接口兼容性）
            contexts: list of context_dicts
            traj: batched Trajectory
            think_batch_size: think 阶段的批大小
            act_batch_size: act 阶段的批大小
            skip_think: 若为 True，跳过 Think 模型推理，responses 用空字符串代替

        Returns:
            (responses, actions): 同 predict_batch
        """
        n = len(contexts)
        all_responses = [''] * n
        all_actions = [None] * n

        if skip_think:
            # 跳过 Think，用空字符串作为 prompts
            pass
        else:
            # 分块 Think
            for start in range(0, n, think_batch_size):
                end = min(start + think_batch_size, n)
                chunk_contexts = contexts[start:end]
                chunk_traj = self._slice_trajectory(traj, start, end)
                chunk_responses, _ = self._think_model.predict_batch(contexts=chunk_contexts, traj=chunk_traj)
                for i, resp in enumerate(chunk_responses):
                    all_responses[start + i] = resp

        # 分块 Act
        for start in range(0, n, act_batch_size):
            end = min(start + act_batch_size, n)
            chunk_prompts = all_responses[start:end]
            chunk_traj = self._slice_trajectory(traj, start, end)
            _, chunk_actions = self._act_model.predict_batch(
                prompts=chunk_prompts, contexts=None, traj=chunk_traj
            )
            for i in range(end - start):
                all_actions[start + i] = chunk_actions[i]

        return all_responses, all_actions

    def get_prompt_messages(self, context: dict) -> list[dict]:
        """返回 system + user 的 prompt messages，透传给 Think 模型。"""
        return self._think_model.get_prompt_messages(context)

    @staticmethod
    def _slice_trajectory(traj: Trajectory, start: int, end: int) -> Trajectory:
        """从 batched Trajectory 中切片 [start:end]"""
        return Trajectory(
            states=traj.states[start:end],
            actions=traj.actions[start:end],
            rtgs=traj.rtgs[start:end],
            timesteps=traj.timesteps[start:end],
            attention_mask=traj.attention_mask[start:end],
        )
