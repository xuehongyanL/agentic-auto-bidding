from abc import abstractmethod
from typing import Any, Optional

from agb_core.data.trajectory import Trajectory
from agb_core.infer.llm_backend import BaseLLMBackend
from agb_core.model.base_model import BaseModel


class ThinkModel(BaseModel):
    """
    Think Model - 思考模型

    输入 context，在内部构造 prompt，输出文本响应 response。
    不输出动作（action 为 None）。

    子类需要实现 _build_prompt 方法来根据 context 构建 prompt。
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
        context: dict,
        prompt = None,
        traj = None,
    ) -> tuple[str, Any]:
        """
        根据 context 在内部构造 prompt，调用 LLM 获取响应

        Args:
            context: 上下文字典，用于内部构造 prompt
            prompt: 忽略此参数（保留接口兼容性）
            traj: Trajectory（保留，暂未使用）

        Returns:
            (response, None): response 是 LLM 的文本响应，action 为 None
        """
        responses, actions = self.predict_batch([context])
        return responses[0], actions[0]

    def predict_batch(
        self,
        contexts: list[dict],
        prompts = None,
        traj = None,
    ) -> tuple[list[str], list[Any]]:
        """
        Batch predict for multiple samples.

        Args:
            prompts: list of prompts (ignored)
            contexts: list of context dicts，用于内部构造 prompts
            traj: batched Trajectory（保留，暂未使用）

        Returns:
            (responses, actions): list of LLM text responses, list of parsed actions
        """
        messages_list = []
        for context in contexts:
            internal_prompt = self._build_prompt(context)
            system_prompt = self._get_system_prompt()
            if system_prompt:
                messages = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': internal_prompt},
                ]
            else:
                messages = [{'role': 'user', 'content': internal_prompt}]
            messages_list.append(messages)

        responses = self._llm_backend.generate_batch(messages_list)

        actions = [self._parse_response(r) for r in responses]
        return responses, actions

    @abstractmethod
    def _build_prompt(self, context: dict) -> str:
        """
        根据 context 构造 prompt

        Args:
            context: 上下文字典

        Returns:
            构建好的 prompt 字符串

        Note:
            子类应该重写此方法来实现具体的 prompt 构造逻辑
        """
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """
        获取 system prompt

        Returns:
            system prompt 字符串

        Note:
            子类应该重写此方法来返回各自的 system prompt
        """
        pass

    @abstractmethod
    def _parse_response(self, response: str):
        """解析 LLM 响应，基类返回 None，子类可覆盖"""
        pass

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
