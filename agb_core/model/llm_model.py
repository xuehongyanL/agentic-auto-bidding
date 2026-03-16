"""
LLM Model 实现

只负责推理后端的定义和调用，不负责 prompt 构造和 context 结构假设。
"""

from typing import Any, Optional

from agb_core.model.base_model import BaseModel


class LLMModel(BaseModel):
    """
    LLM 推理后端封装

    只负责加载模型和调用 LLM 获取响应，不处理 prompt 构造和 context 结构。
    """

    def __init__(
        self,
        model_path: str,
        model_type: str = 'vllm',
        device: str = 'cuda',
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        初始化 LLM Model

        Args:
            model_path: 本地模型路径
            model_type: 模型加载方式，可选 'vllm', 'transformers', 'llamacpp'
            device: 设备
            temperature: 采样温度
            max_tokens: 最大生成 token 数
        """
        self._model_path = model_path
        self._model_type = model_type
        self._device = device
        self._temperature = temperature
        self._max_tokens = max_tokens
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
                device=self._device,
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
        elif self._model_type == 'llamacpp':
            from llama_cpp import Llama
            self._model = Llama(
                model_path=self._model_path,
                n_gpu_layers=-1,
            )
        else:
            raise ValueError(f'Unsupported model type: {self._model_type}')

    def predict(self, prompt: str) -> str:
        """
        根据 prompt 调用 LLM 获取响应

        Args:
            prompt: 完整的 prompt 字符串

        Returns:
            LLM 的原始响应文本
        """
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        """
        调用 LLM 获取响应

        Args:
            prompt: 构建好的 prompt

        Returns:
            LLM 的原始响应
        """
        print('=' * 80)
        print('PROMPT:')
        print('=' * 80)
        print(prompt)
        print('=' * 80)

        if self._model_type == 'vllm':
            from vllm import SamplingParams
            sampling_params = SamplingParams(
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stop=['<|end|>', '<|eot|>'],
            )
            outputs = self._model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            print('MODEL OUTPUT:')
            print('=' * 80)
            print(response)
            print('=' * 80)
            return response
        elif self._model_type == 'transformers':
            messages = [{'role': 'user', 'content': prompt}]
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            ).to(self._device)
            outputs = self._model.generate(
                inputs,
                max_new_tokens=self._max_tokens,
                temperature=self._temperature,
                do_sample=self._temperature > 0,
            )
            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            print('MODEL OUTPUT:')
            print('=' * 80)
            print(response)
            print('=' * 80)
            return response

        else:
            raise ValueError(f'Unsupported model type: {self._model_type}')
