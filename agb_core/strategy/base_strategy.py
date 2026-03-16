"""
Strategy 基类定义

所有出价策略需要继承此基类，负责维护上下文并调用 model 获取 pacer。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agb_core.model.base_model import BaseModel


class BaseStrategy(ABC):
    """
    出价策略抽象基类

    负责维护历史上下文，并调用 model 预测 pacer。
    """

    def __init__(self, model: BaseModel):
        """
        初始化策略

        Args:
            model: 出价模型实例
        """
        self._model = model

    @abstractmethod
    def reset(self) -> None:
        """
        重置策略状态
        """
        pass

    @abstractmethod
    def update(self, env_step_result: Dict[str, Any]) -> None:
        """
        更新策略上下文

        Args:
            env_step_result: 环境 step 返回的结果字典
        """
        pass

    @abstractmethod
    def bidding(self) -> float:
        """
        根据当前上下文输出 pacer

        Returns:
            pacer: 出价系数
        """
        pass

    @property
    def model(self) -> BaseModel:
        """返回关联的模型"""
        return self._model
