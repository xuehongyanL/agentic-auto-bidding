"""
AuctionNet 基础策略实现

维护历史上下文并调用 model 输出 pacer。
所有情况下的 context 都是原始 dict，traj 都是 Trajectory。
"""

from typing import Any

import numpy as np

from agb_core.data.trajectory import Trajectory
from agb_core.model.base_model import DecisionModel
from agb_core.strategy.base_strategy import BaseStrategy


class AuctionNetBaseStrategy(BaseStrategy):
    """
    AuctionNet 数据集的基础策略实现

    维护历史统计信息作为上下文，调用 model 预测 pacer。
    所有情况下的 numeral 都是 (原始 dict, Trajectory)。
    """

    def __init__(
        self,
        model: DecisionModel,
        window_size: int = 20,
    ):
        super().__init__(model)

        self._window_size = window_size

        # 从 model 获取 state_dim 和 action_dim
        self._state_dim = model._state_dim
        self._action_dim = model._action_dim

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
        context_dict, dt_input = self.pre_bidding()
        response, action = self._model.predict(context=context_dict, traj=dt_input)
        return self.post_bidding(response, action)

    def pre_bidding(self) -> tuple:
        """
        Bidding 预处理：构建 context_dict 和 Trajectory，供批量调用使用。

        Returns:
            (context_dict, dt_input): 供模型输入的二元组
        """
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
        return context_dict, dt_input

    def post_bidding(self, response, action) -> tuple[None, np.ndarray]:
        """
        Bidding 后处理：将模型输出的 action 转换为 pacer 并更新历史。

        Args:
            response: 模型文本响应（仅用于日志，忽略）
            action: 模型预测的动作

        Returns:
            (response, pacer)
        """
        if self._model._output_mode == 'price':
            pacer = action / self._cpa_constraint
        else:
            pacer = action

        pacer = pacer.flatten()
        self._last_pacer = pacer
        self._history_actions[-1] = action.tolist()
        self._history_pacers.append(pacer)
        return None, pacer

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


class AuctionNetMultiStrategy:
    """
    多环境并行策略封装，维护多个 AuctionNetBaseStrategy 上下文。

    关键优化：多个环境同时 bidding 时，所有 context 收集后通过
    model.predict_batch() 一次性调用模型，避免 N 次串行调用。
    """

    def __init__(
        self,
        model: DecisionModel,
        n_strategies: int,
        window_size: int = 20,
    ):
        """
        Args:
            model: 模型实例（所有子策略共享同一模型）
            n_strategies: 并行策略数量
            window_size: 历史窗口大小
        """
        self._model = model
        self._n = n_strategies
        self._strategies = [
            AuctionNetBaseStrategy(model, window_size=window_size)
            for _ in range(n_strategies)
        ]

    def reset(self) -> None:
        """重置所有子策略"""
        for s in self._strategies:
            s.reset()

    def set_episode_info_batch(self, reset_infos: list[dict[str, Any]]) -> None:
        """
        批量设置 episode 信息。

        Args:
            reset_infos: list of reset info dicts，与策略顺序对应
        """
        for i, info in enumerate(reset_infos):
            self._strategies[i].set_episode_info(
                budget=info['budget'],
                cpa_constraint=info['cpa_constraint'],
                num_timesteps=info['num_timesteps'],
                first_pvalue_mean=info['first_pvalue_mean'],
                first_pv_num=info['first_pv_num'],
            )

    def pre_bidding(self) -> tuple:
        """
        Bidding 预处理：收集所有子策略的上下文，合并为 batched Trajectory。

        调用子策略的 pre_bidding（含副作用：追加历史）。

        Returns:
            (context_dicts, merged_traj): 供外部多次调用 model.predict_batch 使用
        """
        context_dicts = []
        trajectories = []
        for s in self._strategies:
            context_dict, dt_input = s.pre_bidding()
            context_dicts.append(context_dict)
            trajectories.append(dt_input)

        merged_traj = Trajectory(
            states=np.stack([t.states for t in trajectories]),
            actions=np.stack([t.actions for t in trajectories]),
            rtgs=np.stack([t.rtgs for t in trajectories]),
            timesteps=np.stack([t.timesteps for t in trajectories]),
            attention_mask=np.stack([t.attention_mask for t in trajectories]),
        )
        return context_dicts, merged_traj

    def bidding(self) -> list[tuple]:
        """
        批量 bidding：调用 pre_bidding 收集所有上下文，一次模型调用，post_bidding 分布结果。

        Returns:
            list of (response, pacer) tuples
        """
        context_dicts, merged_traj = self.pre_bidding()
        _, actions_list = self._model.predict_batch(contexts=context_dicts, traj=merged_traj)
        return self.post_bidding(responses=[], actions=actions_list)

    def post_bidding(self, responses: list, actions: list) -> list[tuple[None, np.ndarray]]:
        """
        Bidding 后处理：批量将 actions 转换为 pacers 并提交到历史。

        Args:
            responses: 模型响应列表（忽略）
            actions: 模型预测的动作列表

        Returns:
            list of (response, pacer) tuples
        """
        results = []
        for s, action in zip(self._strategies, actions):
            results.append(s.post_bidding(None, action))
        return results

    def update_batch(self, env_step_results: list[dict[str, Any]]) -> None:
        """
        批量 update：更新所有子策略。

        Args:
            env_step_results: list of env step result dicts
        """
        if len(env_step_results) != self._n:
            raise ValueError(f'Expected {self._n} results, got {len(env_step_results)}')
        for i, result in enumerate(env_step_results):
            self._strategies[i].update(result)

