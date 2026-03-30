"""
Model 基类定义

所有出价模型需要继承此基类，支持三参数输入：
- prompt: 文本输入，可为 None
- context: 上下文字典，可为 None
- traj: Trajectory，可为 None

输出为 (response, action) 元组，其中 response 和 action 都可为 None。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from agb_core.data.trajectory import Trajectory


class BaseModel(ABC):
    """
    出价模型抽象基类

    支持三参数输入：
    - prompt: 文本输入，可为 None
    - context: 上下文字典，可为 None
    - traj: Trajectory，可为 None
    - 返回: (response, action) 元组，其中 response 和 action 都可为 None
    """

    @abstractmethod
    def predict(
        self,
        prompt: Optional[str] = None,
        context: Optional[dict] = None,
        traj: Optional[Trajectory] = None,
    ) -> tuple[Optional[str], Optional[Any]]:
        """
        根据文本、上下文和/或 Trajectory 输入预测响应和/或动作

        Args:
            prompt: 文本输入，可为 None
            context: 上下文字典，可为 None
            traj: Trajectory，可为 None

        Returns:
            (response, action) 元组:
            - response: 文本响应，可为 None
            - action: 动作，可为 None
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        prompts: Optional[list[str]] = None,
        contexts: Optional[list[dict]] = None,
        traj: Optional[Trajectory] = None,
    ) -> tuple[Optional[list[str]], Optional[list[Any]]]:
        pass

    _state_dim: int
    _action_dim: int
    _output_mode: str
    _target_rtg: float
    _scale: float


class DecisionModel(BaseModel):
    """
    决策模型是特殊的BaseModel

    输入: 必须包括context和traj
    输出: 必须有action
    """

    @abstractmethod
    def predict(
        self,
        context: dict,
        traj: Trajectory,
        prompt = None,
    ) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def predict_batch(
        self,
        contexts: list[dict],
        traj: Trajectory,
        prompts = None,
    ) -> tuple[Any, list[Any]]:
        pass
