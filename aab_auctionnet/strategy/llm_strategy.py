"""
AuctionNet LLM 策略实现

继承 DTStrategy，context 为二元组：(原始 dict, DT 多元组)。
"""

from typing import Any, Dict, Tuple

import numpy as np

from aab_auctionnet.strategy.dt_strategy import AuctionNetDTStrategy


class AuctionNetLLMStrategy(AuctionNetDTStrategy):
    """
    AuctionNet 数据集的 LLM 策略实现

    继承 DTStrategy，context 为二元组：
    - 第一个元素：原始 dict（来自 base_strategy 的 _build_context）
    - 第二个元素：DT 多元组 (states, actions, rewards, curr_score, timesteps, attention_mask)
    """

    def __init__(self, model, window_size: int = 20):
        """
        初始化 LLM 策略

        Args:
            model: LLM 模型实例
            window_size: 历史窗口大小（用于构建 DT 多元组）
        """
        super().__init__(model, window_size)
        self._last_pacer: float = 1.0
        # 独立的 pacer 历史记录
        self._history_pacers: list = []

    def reset(self) -> None:
        super().reset()
        self._last_pacer = 1.0
        self._history_pacers = []

    def bidding(self) -> float:
        """构建二元组 context 并调用 LLM 模型获取 pacer"""
        # 与 DTStrategy 相同的处理逻辑
        self._history_actions.append(0.)
        self._history_rewards.append(0.)

        context_dict = self._build_context()
        # 添加 budget 和 cpa_constraint 到 context_dict
        context_dict['budget'] = self._budget
        context_dict['cpa_constraint'] = self._cpa_constraint
        context_dict['num_timesteps'] = self._num_timesteps
        # 添加完整历史列表（用于 LLMModel 构建 prompt 时切片）
        context_dict['history_pacer'] = self._history_pacers
        context_dict['history_pv_num'] = self._history_pv_num
        context_dict['history_conversion'] = self._history_conversion
        context_dict['history_total_cost'] = self._history_total_cost
        context_dict['total_conversions'] = sum(self._history_conversion) if self._history_conversion else 0

        state = self._context_to_state(context_dict)
        self._history_states.append(state)

        # 构建 DT 多元组
        dt_input = self._build_model_input()

        # 二元组：(原始 dict, DT 多元组)
        context = (context_dict, dt_input)

        pacer = self._model.predict(context)
        self._history_pacers.append(pacer)
        self._last_pacer = pacer
        return pacer

    def update(self, env_step_result: Dict[str, Any]) -> None:
        # 调用父类的 update 处理历史记录更新
        # 需要手动处理 because DTStrategy.update 会调用 bidding 相关的逻辑
        pv_num = env_step_result.get('pv_num', 1)
        conversion_sum = env_step_result.get('conversion', 0.0)
        conversion_mean = conversion_sum / pv_num if pv_num > 0 else 0.0

        self._cum_reward += conversion_sum
        self._history_rewards[-1] = conversion_mean

        curr_score = self._calc_curr_score()
        self._history_scores.append(curr_score)

        # 更新 base_strategy 的历史记录
        from aab_auctionnet.strategy.base_strategy import AuctionNetBaseStrategy
        AuctionNetBaseStrategy.update(self, env_step_result)
