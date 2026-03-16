"""
AuctionNet 基础策略实现

维护历史上下文并调用 model 输出 pacer。
"""

from typing import Any, Dict, List

from agb_core.strategy.base_strategy import BaseStrategy


class AuctionNetBaseStrategy(BaseStrategy):
    """
    AuctionNet 数据集的基础策略实现

    维护历史统计信息作为上下文，调用 model 预测 pacer。
    """

    def __init__(self, model):
        super().__init__(model)

        self._history_bid_mean: List[float] = []
        self._history_pvalue_mean: List[float] = []
        self._history_pv_num: List[int] = []
        self._history_conversion: List[float] = []
        self._history_xi: List[float] = []
        self._history_value_mean: List[float] = []
        self._history_least_winning_cost_mean: List[float] = []
        self._history_total_cost: List[float] = []

        self._budget: float = 0.0
        self._cpa_constraint: float = 0.0
        self._num_timesteps: int = 0

    def reset(self) -> None:
        self._history_bid_mean = []
        self._history_pvalue_mean = []
        self._history_pv_num = []
        self._history_conversion = []
        self._history_xi = []
        self._history_value_mean = []
        self._history_least_winning_cost_mean = []
        self._history_total_cost = []

    def update(self, env_step_result: Dict[str, Any]) -> None:
        pv_num = env_step_result.get('pv_num', 1)
        conversion = env_step_result.get('conversion', 0)
        conversion_mean = conversion / pv_num if pv_num > 0 else 0.0
        self._history_bid_mean.append(env_step_result.get('bid_mean', 0))
        self._history_pvalue_mean.append(env_step_result.get('pvalue_mean', 0))
        self._history_pv_num.append(pv_num)
        self._history_conversion.append(conversion_mean)
        self._history_xi.append(env_step_result.get('win_rate', 0))
        self._history_value_mean.append(env_step_result.get('value_mean', 0))
        self._history_least_winning_cost_mean.append(env_step_result.get('least_winning_cost_mean', 0))
        self._history_total_cost.append(env_step_result.get('total_cost', 0))

    def bidding(self) -> float:
        context = self._build_context()
        return self._model.predict(context)

    def set_episode_info(self, budget: float, cpa_constraint: float, num_timesteps: int) -> None:
        self._budget = budget
        self._cpa_constraint = cpa_constraint
        self._num_timesteps = num_timesteps

    def _build_context(self) -> Dict[str, Any]:
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
    def _mean(data: List[float]) -> float:
        if not data:
            return 0.0
        return sum(data) / len(data)

    @staticmethod
    def _mean_last_n(data: List, n: int) -> float:
        if not data:
            return 0.0
        last_n = data[max(0, len(data) - n):]
        return sum(last_n) / len(last_n)
