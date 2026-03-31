"""
AuctionNet 数据集的环境实现

将 AuctionNet 数据读取和模拟出价逻辑封装在 Env 中。
注意：历史记录由 Agent 模块维护，Env 仅负责当前时间步的数据读取和竞拍模拟。
"""

import os
import pickle
import random
from typing import Any, Optional

import numpy as np

from agb_core.env.offline_env import OfflineEnv


class AuctionNetEnv(OfflineEnv):
    """
    AuctionNet 数据集的离线环境实现

    加载 AuctionNet 流量数据，模拟广告竞拍过程。
    历史记录由 Agent 模块维护。
    """

    def __init__(
        self,
        data_filenames: list[str],
        min_remaining_budget: float = 0.1,
    ):
        """
        初始化 AuctionNet 环境

        Args:
            data_filenames: 数据文件路径列表（CSV 或 PKL）
            min_remaining_budget: 最小剩余预算阈值，低于此值时停止出价
        """
        self._data_filenames = list(data_filenames)
        self._min_remaining_budget = min_remaining_budget

        # 数据存储
        self._test_dict: dict = {}
        self._keys: list[tuple] = []

        # 当前状态
        self._current_key: Optional[tuple] = None
        self._current_timestep: int = 0
        self._num_timesteps: int = 0

        # 当前episode数据
        self._pValues: list[np.ndarray] = []
        self._pValueSigmas: list[np.ndarray] = []
        self._leastWinningCosts: list[np.ndarray] = []
        self._budget: float = 0.0
        self._cpa_constraint: float = 0.0
        self._remaining_budget: float = 0.0

        # 加载数据
        self._load_data()

    def _load_data(self) -> None:
        """加载多个数据文件"""
        for path in self._data_filenames:
            if not os.path.exists(path):
                raise ValueError(f'Data file not found: {path}')
            self._load_file(path)

    def _load_file(self, file_path: str) -> None:
        """加载单个数据文件"""
        raw_data_path = file_path.replace('.csv', '.pkl')

        # 尝试使用缓存的 pkl 文件
        if os.path.exists(raw_data_path):
            with open(raw_data_path, 'rb') as f:
                data = pickle.load(f)
        else:
            import pandas as pd
            data = pd.read_csv(file_path, dtype={
                'deliveryPeriodIndex': str,
                'advertiserNumber': str,
                'timeStepIndex': int,
                'advertiserCategoryIndex': str,
            })
            # 缓存为 pkl
            with open(raw_data_path, 'wb') as f:
                pickle.dump(data, f)

        # 按 (period, advertiser) 分组
        grouped_data = data.sort_values('timeStepIndex').groupby(
            ['deliveryPeriodIndex', 'advertiserNumber']
        )

        for key, group in grouped_data:
            if key not in self._test_dict:
                self._test_dict[key] = group
                self._keys.append(key)

    def keys(self) -> list[tuple]:
        """返回所有可用的 (period_id, advertiser_id) 组合"""
        return self._keys.copy()

    def reset(self, key: Optional[tuple] = None) -> dict[str, Any]:
        """
        重置环境到指定 episode 的初始状态

        Args:
            key: 可选的 (period_id, advertiser_id) 组合，
                 如果为 None 则随机选择

        Returns:
            包含初始化信息的字典
        """
        # 选择 key
        if key is None:
            self._current_key = random.choice(self._keys)
        else:
            if key not in self._keys:
                raise ValueError(f'Invalid key: {key}')
            self._current_key = key

        # 加载该 key 对应的数据
        data = self._test_dict[self._current_key]
        self._pValues = data.groupby('timeStepIndex')['pValue'].apply(
            list
        ).apply(np.array).tolist()
        self._pValueSigmas = data.groupby('timeStepIndex')['pValueSigma'].apply(
            list
        ).apply(np.array).tolist()
        self._leastWinningCosts = data.groupby('timeStepIndex')['leastWinningCost'].apply(
            list
        ).apply(np.array).tolist()

        self._num_timesteps = len(self._pValues)
        self._budget = float(data['budget'].iloc[0])
        self._cpa_constraint = float(data['CPAConstraint'].iloc[0])

        # 重置状态
        self._current_timestep = 0
        self._remaining_budget = self._budget

        # 第一步的流量信息（供 Strategy.set_episode_info() 初始化 cpm/cpn）
        first_pvalues = self.get_current_pvalues()
        first_pvalue_mean = float(np.mean(first_pvalues)) if first_pvalues.size > 0 else 0.0
        first_pv_num = int(first_pvalues.size)

        return {
            'budget': self._budget,
            'cpa_constraint': self._cpa_constraint,
            'num_timesteps': self._num_timesteps,
            'first_pvalue_mean': first_pvalue_mean,
            'first_pv_num': first_pv_num,
        }

    def get_current_pvalues(self) -> np.ndarray:
        if self._current_timestep >= self._num_timesteps:
            return np.array([])
        return self._pValues[self._current_timestep]

    def step(self, pacer: np.ndarray) -> dict[str, Any]:
        """
        执行一步出价

        Args:
            pacer: 出价系数/步调器，一维 numpy array

        Returns:
            包含以下键的字典:
            - cost: 本次竞拍花费
            - gmv: 本次 GMV (conversion * cpa_constraint)
            - total_cost: 累计花费
            - done: 是否结束
        """
        if self._current_timestep >= self._num_timesteps:
            raise RuntimeError('Episode already finished, call reset() first')

        # 获取当前时间步的数据
        pValue = self._pValues[self._current_timestep]
        pValueSigma = self._pValueSigmas[self._current_timestep]
        leastWinningCost = self._leastWinningCosts[self._current_timestep]

        # pacer 是一维 array，取第一个元素
        pacer_value = float(pacer.flatten()[0])

        # 预算不足时不出价
        if self._remaining_budget < self._min_remaining_budget:
            bid = np.zeros(pValue.shape[0])
        else:
            # 出价 = bid * 目标CPA * pValue
            bid = pacer_value * self._cpa_constraint * pValue

        # 模拟竞拍
        tick_value, tick_cost, tick_status, tick_conversion = self._simulate_ad_bidding(
            pValue, pValueSigma, bid, leastWinningCost
        )

        # 处理超预算情况
        over_cost_ratio = max(
            (np.sum(tick_cost) - self._remaining_budget) / (np.sum(tick_cost) + 1e-4),
            0
        )
        while over_cost_ratio > 0:
            pv_index = np.where(tick_status == 1)[0]
            dropped_pv_index = np.random.choice(
                pv_index,
                int(np.ceil(pv_index.shape[0] * over_cost_ratio)),
                replace=False
            )
            bid[dropped_pv_index] = 0
            tick_value, tick_cost, tick_status, tick_conversion = self._simulate_ad_bidding(
                pValue, pValueSigma, bid, leastWinningCost
            )
            over_cost_ratio = max(
                (np.sum(tick_cost) - self._remaining_budget) / (np.sum(tick_cost) + 1e-4),
                0
            )

        # 更新预算和状态
        step_cost = float(np.sum(tick_cost))
        conversion = float(np.sum(tick_conversion))
        gmv = conversion * self._cpa_constraint
        self._remaining_budget -= step_cost
        total_cost = self._budget - self._remaining_budget

        # 准备下一时间步
        self._current_timestep += 1
        done = self._current_timestep >= self._num_timesteps

        # 获取下一步的流量信息（供 Strategy.update() 注入 cpm/cpn，供下次 bidding() 使用）
        next_pvalues = self.get_current_pvalues()
        next_pvalue_mean = float(np.mean(next_pvalues)) if next_pvalues.size > 0 else 0.0
        next_pv_num = int(next_pvalues.size)

        return {
            'cost': step_cost,
            'gmv': gmv,
            'total_cost': total_cost,
            'done': done,
            'bid_mean': float(np.mean(bid)) if bid.shape[0] > 0 else 0.0,
            'bid_sum': float(np.sum(bid)),
            'pvalue_mean': float(np.mean(pValue)) if pValue.shape[0] > 0 else 0.0,
            'pvalue_sum': float(np.sum(pValue)),
            'pv_num': pValue.shape[0],
            'next_pvalue_mean': next_pvalue_mean,
            'next_pv_num': next_pv_num,
            'conversion': conversion,
            'value_mean': float(np.mean(tick_value)) if tick_value.shape[0] > 0 else 0.0,
            'win_rate': float(np.mean(tick_status)) if tick_status.shape[0] > 0 else 0.0,
            'least_winning_cost_mean': float(np.mean(leastWinningCost)) if leastWinningCost.shape[0] > 0 else 0.0,
        }

    def _simulate_ad_bidding(
        self,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        bids: np.ndarray,
        leastWinningCosts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        模拟广告竞拍过程

        Args:
            pValues: 广告价值
            pValueSigmas: 价值标准差
            bids: 出价
            leastWinningCosts: 最低获胜成本

        Returns:
            (tick_value, tick_cost, tick_status, tick_conversion)
        """
        tick_status = bids >= leastWinningCosts
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        values = values * tick_status
        tick_value = np.clip(values, 0, 1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)

        return tick_value, tick_cost, tick_status, tick_conversion

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def current_timestep(self) -> int:
        return self._current_timestep

    @property
    def budget(self) -> float:
        return self._budget

    @property
    def remaining_budget(self) -> float:
        return self._remaining_budget

    @property
    def cpa_constraint(self) -> float:
        return self._cpa_constraint


class AuctionNetMultiEnv:
    """
    多环境并行封装，同步管理多个 AuctionNetEnv 实例。

    保持与 AuctionNetEnv 相同的 key 系统，但 reset/step 输入输出均为 list。
    数据文件只加载一次，所有子环境共享。
    """

    def __init__(
        self,
        n_envs: int,
        data_filenames: list[str],
        min_remaining_budget: float = 0.1,
    ):
        """
        Args:
            n_envs: 并行环境数量
            data_filenames: 数据文件路径列表（CSV 或 PKL）
            min_remaining_budget: 最小剩余预算阈值
        """
        self._n_envs = n_envs
        # 第一个 env 加载数据，其余 env 共享其数据字典
        self._base_env = AuctionNetEnv(data_filenames=data_filenames, min_remaining_budget=min_remaining_budget)
        self._test_dict = self._base_env._test_dict
        self._keys = self._base_env._keys

        self._envs = []
        for _ in range(n_envs):
            env = AuctionNetEnv.__new__(AuctionNetEnv)
            env._data_filenames = data_filenames
            env._min_remaining_budget = min_remaining_budget
            env._test_dict = self._test_dict
            env._keys = self._keys
            env._current_key = None
            env._current_timestep = 0
            env._num_timesteps = 0
            env._pValues = []
            env._pValueSigmas = []
            env._leastWinningCosts = []
            env._budget = 0.0
            env._cpa_constraint = 0.0
            env._remaining_budget = 0.0
            self._envs.append(env)

    def keys(self) -> list[tuple]:
        """返回所有可用的 (period_id, advertiser_id) 组合"""
        return self._keys.copy()

    def reset(self, keys: list[tuple]) -> list[dict[str, Any]]:
        """
        重置多个环境到指定 episode 的初始状态。

        Args:
            keys: list of (period_id, advertiser_id) 组合，
                  如果元素为 None 则该环境随机选择 key

        Returns:
            list of reset info dicts，与 keys 顺序对应
        """
        if len(keys) != self._n_envs:
            raise ValueError(f'Expected {self._n_envs} keys, got {len(keys)}')
        return [self._envs[i].reset(keys[i]) for i in range(self._n_envs)]

    def step(self, pacers: list[np.ndarray]) -> list[dict[str, Any]]:
        """
        同步执行多个环境的一步出价。

        Args:
            pacers: list of pacer arrays，与环境顺序对应

        Returns:
            list of step result dicts
        """
        if len(pacers) != self._n_envs:
            raise ValueError(f'Expected {self._n_envs} pacers, got {len(pacers)}')
        return [self._envs[i].step(pacers[i]) for i in range(self._n_envs)]
