"""
AuctionNetAgentMultiStrategy 实现

专门面向 AgentModel 的多环境并行策略，继承 AuctionNetMultiStrategy
的所有批量操作，并额外提供 bidding_chunked 两阶段分块执行能力。
"""

from typing import Any

import numpy as np

from agb_core.data.trajectory import Trajectory
from agb_core.model.agent_model import AgentModel
from agb_auctionnet.strategy.base_strategy import AuctionNetMultiStrategy


class AuctionNetAgentMultiStrategy(AuctionNetMultiStrategy):
    """
    面向 AgentModel 的多环境并行策略封装。

    继承 AuctionNetMultiStrategy 的所有批量操作（reset, bidding, update_batch 等），
    额外提供 bidding_chunked 方法，对 Think 和 Act 两个阶段分别按指定的批大小分块执行，
    以控制 LLM 并发调用的显存峰值。
    """

    def __init__(
        self,
        model: AgentModel,
        n_strategies: int,
        window_size: int = 20,
    ):
        """
        Args:
            model: AgentModel 实例（所有子策略共享同一模型）
            n_strategies: 并行策略数量
            window_size: 历史窗口大小
        """
        super().__init__(model, n_strategies, window_size)

    def bidding_chunked(self, think_batch_size: int, act_batch_size: int) -> list[tuple]:
        """
        分块批量 bidding：think 和 act 分别按各自的批大小分块执行。

        Args:
            think_batch_size: think 阶段的批大小
            act_batch_size: act 阶段的批大小

        Returns:
            list of (response, pacer) tuples
        """
        # 预处理：收集所有 context 和 trajectory
        context_dicts = []
        trajectories = []
        for s in self._strategies:
            context_dict, dt_input = s.pre_bidding()
            context_dicts.append(context_dict)
            trajectories.append(dt_input)

        # 合并为 batched Trajectory
        merged_traj = Trajectory(
            states=np.stack([t.states for t in trajectories]),
            actions=np.stack([t.actions for t in trajectories]),
            rtgs=np.stack([t.rtgs for t in trajectories]),
            timesteps=np.stack([t.timesteps for t in trajectories]),
            attention_mask=np.stack([t.attention_mask for t in trajectories]),
        )

        # 分块调用模型
        _, actions_list = self._model.predict_batch_chunked(
            prompts=None,
            contexts=context_dicts,
            traj=merged_traj,
            think_batch_size=think_batch_size,
            act_batch_size=act_batch_size,
        )

        # 后处理：转换并返回
        results = []
        for i, s in enumerate(self._strategies):
            results.append(s.post_bidding(None, actions_list[i]))
        return results
