"""
ThinkModel 实现

对应原来的 LLMModel，输入仅有 numeral（prompt 在内部构造），输出仅有 response。
只负责推理后端的定义和调用，不负责 prompt 构造和 context 结构假设。
"""

from typing import Any, Optional

from agb_core.infer.llm_backend import BaseLLMBackend
from agb_core.model.base_model import BaseModel


class ThinkModel(BaseModel):
    """
    Think Model - 思考模型

    输入 numeral，在内部构造 prompt，输出文本响应 response。
    不输出动作（action 为 None）。

    子类需要实现 _build_prompt 方法来根据 numeral 构建 prompt。
    """

    def __init__(
        self,
        llm_backend: BaseLLMBackend,
        state_dim: int = 16,
        action_dim: int = 1,
        verbose: int = 0,
    ):
        self._llm_backend = llm_backend
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._output_mode = 'pacer'
        self._verbose = verbose


    def predict(
        self,
        prompt: Optional[str],
        numeral: Optional[Any] = None
    ) -> tuple[Optional[str], Optional[Any]]:
        """
        根据 numeral 在内部构造 prompt，调用 LLM 获取响应

        Args:
            prompt: 忽略此参数（保留接口兼容性）
            numeral: 数值输入，用于内部构造 prompt

        Returns:
            (response, None): response 是 LLM 的文本响应，action 为 None
        """
        # 在内部根据 numeral 构造 prompt
        internal_prompt = self._build_prompt(numeral)
        # 调用 LLM 获取响应
        response = self._call_llm(internal_prompt)
        return response, None

    def _build_prompt(self, numeral: Any) -> str:
        """
        根据 numeral 构造 prompt

        Args:
            numeral: 数值输入

        Returns:
            构建好的 prompt 字符串

        Note:
            子类应该重写此方法来实现具体的 prompt 构造逻辑
        """
        raise NotImplementedError

    def _get_system_prompt(self) -> str:
        """
        获取 system prompt

        Returns:
            system prompt 字符串

        Note:
            子类应该重写此方法来返回各自的 system prompt
        """
        return ""

    def _call_llm(self, prompt: str) -> str:
        system_prompt = self._get_system_prompt()
        if system_prompt:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt},
            ]
        else:
            messages = [{'role': 'user', 'content': prompt}]

        if self._verbose >= 1:
            print('=' * 80)
            print('PROMPT:')
            print('=' * 80)
            print(prompt)
            print('=' * 80)

        response = self._llm_backend.generate(messages)

        if self._verbose >= 1:
            print('MODEL OUTPUT:')
            print('=' * 80)
            print(response)
            print('=' * 80)
        return response
