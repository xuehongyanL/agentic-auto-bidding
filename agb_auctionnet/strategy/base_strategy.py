"""
AuctionNet 基础策略实现

维护历史上下文并调用 model 输出 pacer。
所有情况下的 numeral 都是 (原始 dict, Trajectory)。
"""

from typing import Any

import numpy as np

from agb_core.data.trajectory import Trajectory
from agb_core.model.base_model import BaseModel
from agb_core.strategy.base_strategy import BaseStrategy


class AuctionNetBaseStrategy(BaseStrategy):
    """
    AuctionNet 数据集的基础策略实现

    维护历史统计信息作为上下文，调用 model 预测 pacer。
    所有情况下的 numeral 都是 (原始 dict, Trajectory)。
    """

    def __init__(
        self,
        model: BaseModel,
        window_size: int = 20,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
    ):
        super().__init__(model)

        self._window_size = window_size

        # 从 model 获取 state_dim 和 action_dim
        self._state_dim = model._state_dim
        self._action_dim = model._action_dim

        # 状态归一化参数（AuctionNet 专属）
        if state_mean is None:
            state_mean = np.zeros(self._state_dim, dtype=np.float32)
        if state_std is None:
            state_std = np.ones(self._state_dim, dtype=np.float32)
        self._state_mean = state_mean.astype(np.float32)
        self._state_std = state_std.astype(np.float32)

        # 历史统计信息
        self._history_bid_mean: list[float] = []
        self._history_pvalue_mean: list[float] = []
        self._history_pv_num: list[int] = []
        self._history_conversion: list[float] = []
        self._history_xi: list[float] = []
        self._history_value_mean: list[float] = []
        self._history_least_winning_cost_mean: list[float] = []
        self._history_total_cost: list[float] = []

        # DT 相关历史记录
        self._history_states: list = []
        self._history_actions: list = []
        self._history_rtgs: list = []

        # LLM 策略需要的 pacer 历史
        self._history_pacers: list = []

        self._budget: float = 0.0
        self._cpa_constraint: float = 0.0
        self._num_timesteps: int = 0

        self._last_pacer: np.ndarray = np.array([1.0])
        self._cum_reward: float = 0.0

        # 当前时间步的流量信息（由 update() 自动从 env.step() 结果中注入，供下次 bidding() 使用）
        self.cpm: float = 0.0
        self.cpn: int = 0

    def reset(self) -> None:
        self._history_bid_mean = []
        self._history_pvalue_mean = []
        self._history_pv_num = []
        self._history_conversion = []
        self._history_xi = []
        self._history_value_mean = []
        self._history_least_winning_cost_mean = []
        self._history_total_cost = []

        self._history_states = []
        self._history_actions = []
        self._history_rtgs = [self._model._target_rtg]
        self._history_pacers = []
        self._last_pacer = np.array([1.0])
        self._cum_reward = 0.0

    def update(self, env_step_result: dict[str, Any]) -> None:
        pv_num = env_step_result.get('pv_num', 1)
        conversion_sum = env_step_result.get('conversion', 0.0)
        conversion_mean = conversion_sum / pv_num if pv_num > 0 else 0.0

        # 为下一次 bidding() 注入下一步的流量信息
        self.cpm = env_step_result.get('next_pvalue_mean', 0.0)
        self.cpn = env_step_result.get('next_pv_num', 0)

        # _cum_reward 累加总和，用于 rtg 计算
        self._cum_reward += conversion_sum

        rtg = self._calc_rtg()
        self._history_rtgs.append(rtg)

        # 更新历史统计信息
        self._history_bid_mean.append(env_step_result.get('bid_mean', 0))
        self._history_pvalue_mean.append(env_step_result.get('pvalue_mean', 0))
        self._history_pv_num.append(pv_num)
        self._history_conversion.append(conversion_mean)
        self._history_xi.append(env_step_result.get('win_rate', 0))
        self._history_value_mean.append(env_step_result.get('value_mean', 0))
        self._history_least_winning_cost_mean.append(env_step_result.get('least_winning_cost_mean', 0))
        self._history_total_cost.append(env_step_result.get('total_cost', 0))

    def bidding(self) -> tuple:
        """构建二元组 context 并调用模型"""
        # 为下一步 append 0 (统一为 array 模式)
        self._history_actions.append([0.] * self._action_dim)

        context_dict = self._build_context()

        # 添加 LLM 策略需要的额外字段
        context_dict['budget'] = self._budget
        context_dict['cpa_constraint'] = self._cpa_constraint
        context_dict['num_timesteps'] = self._num_timesteps
        context_dict['window_size'] = self._window_size
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
        response, action = self._model.predict(None, (context_dict, dt_input))

        # 根据 output_mode 判断是否需要转换：'price' 需要除以 cpa_constraint 得到 pacer
        if self._model._output_mode == 'price':
            pacer = action / self._cpa_constraint
        else:
            pacer = action

        # 确保 pacer 是一维 numpy array
        pacer = pacer.flatten()

        self._last_pacer = pacer
        self._history_actions[-1] = action.tolist()
        self._history_pacers.append(pacer)
        return response, pacer

    def _calc_rtg(self) -> float:
        """计算当前 rtg"""
        if not self._history_states or self._cum_reward <= 0:
            return self._model._target_rtg

        state = self._history_states[-1]
        budget_left = state[1]
        curr_cost = self._budget * (1 - budget_left)
        curr_cpa = curr_cost / (self._cum_reward + 1e-10)
        curr_coef = self._cpa_constraint / (curr_cpa + 1e-10)
        curr_penalty_squared = curr_coef ** 2
        curr_penalty = 1.0 if curr_penalty_squared > 1.0 else curr_coef
        current_score = curr_penalty * self._cum_reward
        return float(self._model._target_rtg - current_score / self._model._scale)

    def _build_model_input(self) -> Trajectory:
        """构建模型输入，padding 到 window_size"""
        T = len(self._history_states)

        valid_len = min(T, self._window_size)
        pad_size = self._window_size - valid_len

        # 将 actions 转换为 numpy array（每个元素是 action_dim 维的列表）
        actions_list = [np.array(a, dtype=np.float32).flatten() for a in self._history_actions]
        actions = np.array(actions_list, dtype=np.float32)

        if pad_size > 0:
            states = np.array(self._history_states, dtype=np.float32)
            rtgs = np.array(self._history_rtgs, dtype=np.float32).reshape(-1, 1)
            timesteps = np.arange(T, dtype=np.int64)

            pad_state = np.zeros((pad_size, self._state_dim), dtype=np.float32)
            pad_action = np.zeros((pad_size, self._action_dim), dtype=np.float32)
            pad_rtg = np.zeros((pad_size, 1), dtype=np.float32)
            pad_time = np.zeros(pad_size, dtype=np.int64)
            pad_mask = np.zeros((pad_size, self._action_dim), dtype=np.int64)

            states = np.concatenate([pad_state, states], axis=0)
            actions = np.concatenate([pad_action, actions], axis=0)
            rtgs = np.concatenate([pad_rtg, rtgs], axis=0)
            timesteps = np.concatenate([pad_time, timesteps], axis=0)
            attention_mask = np.concatenate([pad_mask, np.ones((valid_len, self._action_dim), dtype=np.int64)], axis=0)
        else:
            states = np.array(self._history_states[-self._window_size:], dtype=np.float32)
            actions = actions[-self._window_size:]
            rtgs = np.array(self._history_rtgs[-self._window_size:], dtype=np.float32).reshape(-1, 1)
            timesteps = np.arange(T - self._window_size, T, dtype=np.int64)
            attention_mask = np.ones((self._window_size, self._action_dim), dtype=np.int64)

        # 状态归一化（只对有效数据，padding 部分保持为 0）
        # 通过 np.max 现场计算 timestep_mask
        valid_mask = np.max(attention_mask, axis=-1, keepdims=True)  # (window_size, 1)
        states = np.where(valid_mask == 1, (states - self._state_mean) / (self._state_std + 1e-9), states)

        return Trajectory(states, actions, rtgs, timesteps, attention_mask)

    def _context_to_state(self, context: dict[str, Any]) -> np.ndarray:
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

    def set_episode_info(self, budget: float, cpa_constraint: float, num_timesteps: int,
                         first_pvalue_mean: float = 0.0, first_pv_num: int = 0) -> None:
        self._budget = budget
        self._cpa_constraint = cpa_constraint
        self._num_timesteps = num_timesteps
        self.cpm = first_pvalue_mean
        self.cpn = first_pv_num

    def _build_context(self) -> dict[str, Any]:
        t = len(self._history_bid_mean)
        time_left = (self._num_timesteps - t) / self._num_timesteps if self._num_timesteps > 0 else 0

        if self._budget > 0:
            budget_left = (self._budget - (self._history_total_cost[-1] if self._history_total_cost else 0)) / self._budget
        else:
            budget_left = 0

        #hack
        current_pvalue_mean = self.cpm
        current_pv_num = self.cpn

        historical_bid_mean = self._mean(self._history_bid_mean)
        last_three_bid_mean = self._mean_last_n(self._history_bid_mean, 3)
        historical_lwc_mean = self._mean(self._history_least_winning_cost_mean)
        historical_pvalue_mean = self._mean(self._history_pvalue_mean)
        historical_conversion_mean = self._mean(self._history_conversion)
        historical_xi_mean = self._mean(self._history_xi)
        last_three_lwc_mean = self._mean_last_n(self._history_least_winning_cost_mean, 3)
        last_three_pvalue_mean = self._mean_last_n(self._history_pvalue_mean, 3)
        last_three_conversion_mean = self._mean_last_n(self._history_conversion, 3)
        last_three_xi_mean = self._mean_last_n(self._history_xi, 3)
        last_three_pv_num_total = sum(self._history_pv_num[max(0, len(self._history_pv_num) - 3):])
        historical_pv_num_total = sum(self._history_pv_num)

        context = {
            'time_left': time_left,
            'budget_left': budget_left,
            'historical_bid_mean': historical_bid_mean,
            'last_three_bid_mean': last_three_bid_mean,
            'historical_LeastWinningCost_mean': historical_lwc_mean,
            'historical_pValues_mean': historical_pvalue_mean,
            'historical_conversion_mean': historical_conversion_mean,
            'historical_xi_mean': historical_xi_mean,
            'last_three_LeastWinningCost_mean': last_three_lwc_mean,
            'last_three_pValues_mean': last_three_pvalue_mean,
            'last_three_conversion_mean': last_three_conversion_mean,
            'last_three_xi_mean': last_three_xi_mean,
            'current_pValues_mean': current_pvalue_mean,
            'current_pv_num': current_pv_num,
            'last_three_pv_num_total': last_three_pv_num_total,
            'historical_pv_num_total': historical_pv_num_total,
        }

        return context

    @staticmethod
    def _mean(data: list[float]) -> float:
        if not data:
            return 0.0
        return sum(data) / len(data)

    @staticmethod
    def _mean_last_n(data: list, n: int) -> float:
        if not data:
            return 0.0
        last_n = data[max(0, len(data) - n):]
        return sum(last_n) / len(last_n)
