"""
抽象的离线环境接口定义

定义了所有离线环境实现需要遵循的接口规范。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class OfflineEnv(ABC):
    """
    离线环境抽象基类

    用于在离线数据集上进行出价策略的评估和训练。
    Env层接收pacer参数，输出竞价结果的各种统计量。
    """

    @abstractmethod
    def keys(self) -> List[Tuple]:
        """
        返回环境中所有可用的 (period_id, advertiser_id) 组合

        Returns:
            List of (period_id, advertiser_id) tuples
        """
        pass

    @abstractmethod
    def reset(self, key: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        重置环境到初始状态

        Args:
            key: 可选的 (period_id, advertiser_id) 组合，
                 如果为 None 则随机选择

        Returns:
            包含初始化信息的字典
        """
        pass

    @abstractmethod
    def step(self, pacer: float) -> Dict[str, Any]:
        """
        执行一步出价

        Args:
            pacer: 出价系数/步调器，用于调整出价策略

        Returns:
            包含竞价结果统计量的字典，具体字段由实现类定义
        """
        pass

    @property
    @abstractmethod
    def num_timesteps(self) -> int:
        """返回总时间步数"""
        pass

    @property
    @abstractmethod
    def current_timestep(self) -> int:
        """返回当前时间步"""
        pass

    @property
    @abstractmethod
    def budget(self) -> float:
        """返回初始预算"""
        pass

    @property
    @abstractmethod
    def remaining_budget(self) -> float:
        """返回剩余预算"""
        pass

    @property
    @abstractmethod
    def cpa_constraint(self) -> float:
        """返回 CPA 约束"""
        pass
