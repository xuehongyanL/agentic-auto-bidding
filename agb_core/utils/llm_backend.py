from agb_core.infer.llm_backend import (OpenAIBackend, TransformersBackend,
                                        VLLMBackend)


def build_llm_backend(bcfg: dict):
    """根据 config 构建 llm_backend，支持 transformers / vllm / openai"""
    backend_type = bcfg['type']
    if backend_type == 'transformers':
        return TransformersBackend(
            model_path=bcfg['model_path'],
            temperature=bcfg['temperature'],
            max_tokens=bcfg['max_tokens'],
            stop=bcfg.get('stop', []),
            device=bcfg.get('device', 'cuda'),
        )
    elif backend_type == 'vllm':
        return VLLMBackend(
            model_path=bcfg['model_path'],
            temperature=bcfg['temperature'],
            max_tokens=bcfg['max_tokens'],
            stop=bcfg.get('stop', []),
            tensor_parallel_size=bcfg.get('tensor_parallel_size', 1),
            gpu_memory_utilization=bcfg.get('gpu_memory_utilization', 0.9),
        )
    elif backend_type == 'openai':
        return OpenAIBackend(
            model_path=bcfg['model_path'],
            temperature=bcfg['temperature'],
            max_tokens=bcfg['max_tokens'],
            stop=bcfg.get('stop', []),
            base_url=bcfg.get('base_url'),
            api_key=bcfg.get('api_key'),
            timeout=bcfg.get('timeout', 120.0),
        )
    else:
        raise ValueError(f'Unknown llm_backend type: {backend_type}')
