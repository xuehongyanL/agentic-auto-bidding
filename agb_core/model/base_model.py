"""
Model 基类定义

所有出价模型需要继承此基类，支持双模态输入输出：
- 文本输入 (prompt): 可以为 None
- 数值输入 (numeral): 可以为 None
- 文本输出 (response): 可以为 None
- 动作输出 (action): 可以为 None

输入输出的双模态中可以留空其中一个。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseModel(ABC):
    """
    出价模型抽象基类

    支持双模态输入输出：
    - prompt: 文本输入，可为 None
    - numeral: 数值输入，可为 None
    - 返回: (response, action) 元组，其中 response 和 action 都可为 None
    """

    @abstractmethod
    def predict(
        self,
        prompt: Optional[str],
        numeral: Optional[Any] = None
    ) -> Tuple[Optional[str], Optional[Any]]:
        """
        根据文本和/或数值输入预测响应和/或动作

        Args:
            prompt: 文本输入，可为 None
            numeral: 数值输入，可为 None（具体格式由子类定义）

        Returns:
            (response, action) 元组:
            - response: 文本响应，可为 None
            - action: 动作，可为 None
        """
        pass
