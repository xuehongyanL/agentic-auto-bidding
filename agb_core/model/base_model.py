"""
Model 基类定义

所有出价模型需要继承此基类，根据上下文输出 pacer。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """
    出价模型抽象基类

    接收决策上下文，输出 pacer（出价系数）。
    """

    @abstractmethod
    def predict(self, context) -> float:
        """
        根据上下文预测 pacer

        Args:
            context: 决策上下文，包含历史统计、预算余量等信息

        Returns:
            pacer: 出价系数
        """
        pass
