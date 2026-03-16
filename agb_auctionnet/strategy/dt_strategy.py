"""
AuctionNet DT 策略实现

继承基础策略，将上下文转换为 DT 模型所需格式。
"""

from typing import Any, Dict, Tuple

import numpy as np

from agb_auctionnet.strategy.base_strategy import AuctionNetBaseStrategy


class AuctionNetDTStrategy(AuctionNetBaseStrategy):
    """
    AuctionNet 数据集的 DT 策略实现

    继承基础策略，将上下文字典转换为模型所需的 numpy 数组格式。
    """

    def __init__(self, model, window_size=20):
        super().__init__(model)
        self._window_size = window_size
        self._history_states: list = []
        self._history_actions: list = []
        self._history_rewards: list = []
        self._history_scores: list = []
        self._last_pacer: float = 1.0
        self._cum_reward: float = 0.0

    def reset(self) -> None:
        super().reset()
        self._history_states = []
        self._history_actions = []
        self._history_rewards = []
        self._history_scores = [self._model._target_return]
        self._last_pacer = 1.0
        self._cum_reward = 0.0

    def bidding(self) -> float:
        """将历史序列转换为模型输入后调用模型"""
        # 为下一步 append 0（与 GAVE 对齐，get_action 前会 append 0）
        self._history_actions.append(0.)
        self._history_rewards.append(0.)

        state = self._context_to_state(self._build_context())
        self._history_states.append(state)

        states, actions, rewards, curr_score, timesteps, attention_mask = self._build_model_input()

        # print(self._history_states[-1])

        action = self._model.predict((states, actions, rewards, curr_score, timesteps, attention_mask))
        # 更新最后一个位置的 action（与 GAVE 对齐）
        self._history_actions[-1] = action
        pacer = action / self._cpa_constraint
        self._last_pacer = pacer
        return pacer

    def update(self, env_step_result: Dict[str, Any]) -> None:
        pv_num = env_step_result.get('pv_num', 1)
        conversion_sum = env_step_result.get('conversion', 0.0)
        conversion_mean = conversion_sum / pv_num if pv_num > 0 else 0.0

        # _cum_reward 累加总和，用于 curr_score 计算
        self._cum_reward += conversion_sum

        # 注意：_history_actions 和 _history_rewards 的 append 已在 bidding() 中处理
        # 这里只需要更新 rewards 的最后一个位置（因为 bidding() 中 append 了 0）
        self._history_rewards[-1] = conversion_mean

        curr_score = self._calc_curr_score()
        self._history_scores.append(curr_score)

        super().update(env_step_result)

    def _calc_curr_score(self) -> float:
        """计算当前 score，用于 curr_score = target_return - current_score，与GAVE对齐除以scale"""
        if not self._history_states or self._cum_reward <= 0:
            return self._model._target_return

        state = self._history_states[-1]
        budget_left = state[1]
        curr_cost = self._budget * (1 - budget_left)
        curr_cpa = curr_cost / (self._cum_reward + 1e-10)
        curr_coef = self._cpa_constraint / (curr_cpa + 1e-10)
        # GAVE 的逻辑: if coef^2 > 1: penalty = 1.0 else: penalty = coef
        curr_penalty_squared = curr_coef ** 2
        curr_penalty = 1.0 if curr_penalty_squared > 1.0 else curr_coef
        current_score = curr_penalty * self._cum_reward
        return float(self._model._target_return - current_score / self._model._scale)

    def _build_model_input(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """构建模型输入，padding 到 window_size"""
        T = len(self._history_states)
        state_dim = 16

        valid_len = min(T, self._window_size)
        pad_size = self._window_size - valid_len

        if pad_size > 0:
            states = np.array(self._history_states, dtype=np.float32)
            actions = np.array(self._history_actions, dtype=np.float32).reshape(-1, 1)
            rewards = np.array(self._history_rewards, dtype=np.float32).reshape(-1, 1)
            scores = np.array(self._history_scores, dtype=np.float32).reshape(-1, 1)
            timesteps = np.arange(T, dtype=np.int64)

            pad_state = np.zeros((pad_size, state_dim), dtype=np.float32)
            pad_action = np.zeros((pad_size, 1), dtype=np.float32)
            pad_reward = np.zeros((pad_size, 1), dtype=np.float32)
            pad_score = np.zeros((pad_size, 1), dtype=np.float32)
            pad_time = np.zeros(pad_size, dtype=np.int64)

            states = np.concatenate([pad_state, states], axis=0)
            actions = np.concatenate([pad_action, actions], axis=0)
            rewards = np.concatenate([pad_reward, rewards], axis=0)
            scores = np.concatenate([pad_score, scores], axis=0)
            timesteps = np.concatenate([pad_time, timesteps], axis=0)
            attention_mask = np.concatenate([np.zeros(pad_size, dtype=np.int64), np.ones(valid_len, dtype=np.int64)], axis=0)
            # print(states.shape, actions.shape, rewards.shape)
        else:
            states = np.array(self._history_states[-self._window_size:], dtype=np.float32)
            actions = np.array(self._history_actions[-self._window_size:], dtype=np.float32).reshape(-1, 1)
            rewards = np.array(self._history_rewards[-self._window_size:], dtype=np.float32).reshape(-1, 1)
            scores = np.array(self._history_scores[-self._window_size:], dtype=np.float32).reshape(-1, 1)
            # timesteps 应该对应实际的历史索引
            timesteps = np.arange(T - self._window_size, T, dtype=np.int64)
            attention_mask = np.ones(self._window_size, dtype=np.int64)
        return states, actions, rewards, scores, timesteps, attention_mask

    def _context_to_state(self, context: Dict[str, Any]) -> np.ndarray:
        """将上下文字典转换为模型输入状态向量"""
        state = np.array([
            context['time_left'],
            context['budget_left'],
            context['historical_bid_mean'],
            context['last_three_bid_mean'],
            context['historical_LeastWinningCost_mean'],
            context['historical_pValues_mean'],
            context['historical_conversion_mean'],
            context['historical_xi_mean'],
            context['last_three_LeastWinningCost_mean'],
            context['last_three_pValues_mean'],
            context['last_three_conversion_mean'],
            context['last_three_xi_mean'],
            context['current_pValues_mean'],
            context['current_pv_num'],
            context['last_three_pv_num_total'],
            context['historical_pv_num_total'],
        ], dtype=np.float32)

        return state
