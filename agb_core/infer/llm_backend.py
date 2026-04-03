"""
LLM Backend

VLLMBackend, TransformersBackend, OpenAIBackend
generate(messages) -> str
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
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
        pass

    @abstractmethod
    def generate_batch(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        """
        Batch generate for multiple prompts simultaneously.

        Args:
            messages_list: list of message lists, each as [{'role': ..., 'content': ...}]
        Returns:
            list of generated texts
        """
        pass


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
        top_p: float = 1.0,
    ):
        self._model_path = model_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop = stop or []
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._trust_remote_code = trust_remote_code
        self._top_p = top_p
        self._llm: Any = None
        self._load_model()

    def __del__(self):
        """销毁 vLLM 创建的 PyTorch distributed 进程组，避免程序退出时 warn."""
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()

    def _load_model(self) -> None:
        from vllm import LLM
        self._llm = LLM(
            model=self._model_path,
            tensor_parallel_size=self._tensor_parallel_size,
            gpu_memory_utilization=self._gpu_memory_utilization,
            trust_remote_code=self._trust_remote_code,
        )

    def generate(self, messages: list[dict[str, str]]) -> str:
        results = self.generate_batch([messages])
        return results[0]

    def generate_batch(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        from vllm import SamplingParams

        tokenizer = self._llm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in messages_list
        ]
        sampling_params = SamplingParams(
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            stop=self._stop or ['<|end|>', '<|eot|>'],
            top_p=self._top_p,
        )
        outputs = self._llm.generate(prompts, sampling_params)
        return [out.outputs[0].text for out in outputs]


class TransformersBackend(BaseLLMBackend):
    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop: Optional[list[str]] = None,
        device: str = 'cuda',
        trust_remote_code: bool = True,
        top_p: float = 1.0,
    ):
        self._model_path = model_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop = stop or []
        self._device = device
        self._trust_remote_code = trust_remote_code
        self._top_p = top_p
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
        results = self.generate_batch([messages])
        return results[0]

    def generate_batch(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        input_ids_list = []
        for messages in messages_list:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt',
            )
            input_ids = input_ids.to(self._device)
            input_ids_list.append(input_ids)

        # Pad to same length
        max_len = max(ids.shape[1] for ids in input_ids_list)
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        padded_input_ids = []
        padded_attention_masks = []
        for input_ids in input_ids_list:
            pad_len = max_len - input_ids.shape[1]
            if pad_len > 0:
                pad_tensor = input_ids.new_full((1, pad_len), pad_id)
                padded_input_ids.append(torch.cat([pad_tensor, input_ids], dim=1))
                padded_attention_masks.append(torch.cat([input_ids.new_zeros((1, pad_len)), torch.ones_like(input_ids)], dim=1))
            else:
                padded_input_ids.append(input_ids)
                padded_attention_masks.append(torch.ones_like(input_ids))

        batch_input_ids = torch.cat(padded_input_ids, dim=0)
        batch_attention_mask = torch.cat(padded_attention_masks, dim=0)

        output_ids = self._model.generate(
            batch_input_ids,
            attention_mask=batch_attention_mask,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            do_sample=self._temperature > 0,
        )

        results = []
        for i, input_ids in enumerate(input_ids_list):
            input_len = input_ids.shape[1]
            response_ids = output_ids[i][input_len:]
            results.append(self._tokenizer.decode(response_ids, skip_special_tokens=True))
        return results


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
        top_p: float = 1.0,
        **kwargs: Any,
    ):
        self._model_path = model_path
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._stop = stop or []
        self._top_p = top_p
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
        if self._top_p < 1.0:
            kwargs['top_p'] = self._top_p
        return self._client.chat.completions.create(**kwargs).choices[0].message.content

    def generate_batch(self, messages_list: list[list[dict[str, str]]]) -> list[str]:
        raise NotImplementedError('OpenAIBackend does not support batch generation')
