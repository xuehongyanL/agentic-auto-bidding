"""
ThinkModel 实现

对应原来的 LLMModel，输入仅有 numeral（prompt 在内部构造），输出仅有 response。
只负责推理后端的定义和调用，不负责 prompt 构造和 context 结构假设。
"""

from typing import Any, Optional

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
        model_path: str,
        model_type: str = 'vllm',
        device: str = 'cuda',
        temperature: float = 0.0,
        max_tokens: int = 1024,
        state_dim: int = 16,
        action_dim: int = 1,
        verbose: int = 0,
    ):
        """
        初始化 Think Model

        Args:
            model_path: 本地模型路径
            model_type: 模型加载方式，可选 'vllm', 'transformers'
            device: 设备
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            state_dim: 状态维度
            action_dim: 动作维度
            verbose: 是否打印 prompt，0 不打印，1 完整打印
        """
        self._model_path = model_path
        self._model_type = model_type
        self._device = device
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._verbose = verbose
        self._model = None
        self._tokenizer = None

        self._load_model()

    def _load_model(self) -> None:
        """加载本地模型"""
        if self._model_type == 'vllm':
            from vllm import LLM
            self._model = LLM(
                model=self._model_path,
                trust_remote_code=True,
            )
        elif self._model_type == 'transformers':
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                device_map=self._device,
            )
        else:
            raise ValueError(f'Unsupported model type: {self._model_type}')

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
        """
        调用 LLM 获取响应

        Args:
            prompt: 构建好的 prompt

        Returns:
            LLM 的原始响应
        """
        if self._verbose >= 1:
            print('=' * 80)
            print('PROMPT:')
            print('=' * 80)
            print(prompt)
            print('=' * 80)

        system_prompt = self._get_system_prompt()
        if system_prompt:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
        else:
            messages = [{'role': 'user', 'content': prompt}]

        if self._model_type == 'vllm':
            from vllm import SamplingParams
            tokenizer = self._model.get_tokenizer()

            # 应用 chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            sampling_params = SamplingParams(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=['<|end|>', '<|eot|>'],
            )
            outputs = self._model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            if self._verbose >= 1:
                print('MODEL OUTPUT:')
                print('=' * 80)
                print(response)
                print('=' * 80)
            return response
        elif self._model_type == 'transformers':
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).to(self._device)
            output_ids = self._model.generate(
                inputs,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0,
            )
            # 只提取 assistant 新生成的部分（不包括 input_ids）
            input_len = inputs.shape[1]
            response_ids = output_ids[0][input_len:]
            response = self._tokenizer.decode(response_ids, skip_special_tokens=True)
            if self._verbose >= 1:
                print('MODEL OUTPUT:')
                print('=' * 80)
                print(response)
                print('=' * 80)
            return response
        else:
            raise ValueError(f'Unsupported model type: {self._model_type}')
