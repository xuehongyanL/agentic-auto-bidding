"""
AgentModel 实现

组合 Think 和 Act 两个子模型：
1. Think 模型：输入 numeral，输出 response
2. Act 模型：输入 (response, numeral)，输出 action
"""

from typing import Any, Optional, Tuple

from agb_core.model.base_model import BaseModel


class AgentModel(BaseModel):
    """
    Agent Model - Agent 模型

    组合 Think 和 Act 两个子模型：
    1. Think 模型：输入 numeral，输出 response
    2. Act 模型：输入 (response, numeral)，输出 action

    双输入：
    - prompt: None（忽略，由 Think 模型内部构造）
    - numeral: 二元组 (context_dict, dt_tuple)

    双输出：
    - response: Think 模型的文本响应
    - action: Act 模型预测的动作
    """

    def __init__(self, think_model: BaseModel, act_model: BaseModel):
        """
        初始化 Agent Model

        Args:
            think_model: Think 子模型，负责生成文本推理
            act_model: Act 子模型，负责输出动作
        """
        self._think_model = think_model
        self._act_model = act_model

        # 继承子模型的占位符属性（用于与策略兼容）
        self._target_rtg = getattr(think_model, '_target_rtg', 0.0)
        self._scale = getattr(think_model, '_scale', 1.0)
        # 从 act_model 获取 state_dim 和 action_dim
        self._state_dim = act_model._state_dim
        self._action_dim = act_model._action_dim
        self._output_mode = act_model._output_mode

    def predict(
        self,
        prompt: Optional[str],
        numeral: Optional[Any] = None
    ) -> Tuple[Optional[str], Optional[Any]]:
        """
        两阶段预测：
        1. 调用 Think 模型获取 response
        2. 将 response 作为 prompt，与 numeral 一起传给 Act 模型获取 action

        Args:
            prompt: 忽略此参数（保留接口兼容性）
            numeral: 二元组 (context_dict, dt_tuple)

        Returns:
            (response, action): response 是 Think 模型的文本响应，action 是 Act 模型预测的动作
        """
        if numeral is None:
            raise ValueError("AgentModel requires numeral input")

        # 第一步：调用 Think 模型获取 response
        think_response, _ = self._think_model.predict(None, numeral)

        # 第二步：将 response 作为 prompt，与 numeral 一起传给 Act 模型
        _, action = self._act_model.predict(think_response, numeral)

        return think_response, action
