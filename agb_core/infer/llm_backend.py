"""
LLM Backend

VLLMBackend, TransformersBackend, OpenAIBackend
generate(messages) -> str
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import OpenAI


class BaseLLMBackend(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """
        Args:
            messages: OpenAI format, e.g. [{'role': 'user', 'content': '...'}]
        Returns:
            generated text
        """
        raise NotImplementedError


class VLLMBackend(BaseLLMBackend):
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
    ):
        self._model_path = model_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop = stop or []
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._trust_remote_code = trust_remote_code
        self._llm: Any = None
        self._load_model()

    def _load_model(self) -> None:
        from vllm import LLM
        self._llm = LLM(
            model=self._model_path,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            trust_remote_code=self._trust_remote_code,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        from vllm import SamplingParams

        tokenizer = self._llm.get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stop=self._stop or ['<|end|>', '<|eot|>'],
        )
        outputs = self._llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text


class TransformersBackend(BaseLLMBackend):
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
        device: str = 'cuda',
        trust_remote_code: bool = True,
    ):
        self._model_path = model_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop = stop or []
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._model: Any = None
        self._tokenizer: Any = None
        self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path,
            trust_remote_code=self._trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_path,
            trust_remote_code=self._trust_remote_code,
            device_map=self._device,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        inputs = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors='pt',
        ).to(self._device)
        if self._tokenizer.pad_token_id is not None:
            pad_id = self._tokenizer.pad_token_id
        else:
            pad_id = self._tokenizer.eos_token_id
        attention_mask = inputs.ne(pad_id)
        output_ids = self._model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
            do_sample=self._temperature > 0,
        )
        input_len = inputs.shape[1]
        response_ids = output_ids[0][input_len:]
        return self._tokenizer.decode(response_ids, skip_special_tokens=True)


class OpenAIBackend(BaseLLMBackend):
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        **kwargs: Any,
    ):
        self._model_path = model_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop = stop or []
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def generate(self, messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            'model': self._model_path,
            'messages': messages,
            'max_tokens': self._max_tokens,
        }
        if self._stop:
            kwargs['stop'] = self._stop
        if self._temperature > 0:
            kwargs['temperature'] = self._temperature
        return self._client.chat.completions.create(**kwargs).choices[0].message.content
