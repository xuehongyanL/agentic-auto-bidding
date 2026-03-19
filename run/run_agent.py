import argparse
import pickle

import yaml

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv
from agb_auctionnet.model.think_model import AuctionNetThinkModel
from agb_auctionnet.strategy.base_strategy import AuctionNetBaseStrategy
from agb_core.infer.llm_backend import (OpenAIBackend, TransformersBackend,
                                        VLLMBackend)
from agb_core.model.act_model import ActModel
from agb_core.model.agent_model import AgentModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(config['model']['act_model']['normalize_dict_path'], 'rb') as f:
        normalize_dict = pickle.load(f)

    env = AuctionNetEnv(data_filename=config['env']['data_path'])

    # 初始化 llm_backend
    bcfg = config['model']['think_model']['llm_backend']
    backend_type = bcfg['type']
    if backend_type == 'transformers':
        llm_backend = TransformersBackend(
            model_path=bcfg['model_path'],
            temperature=bcfg['temperature'],
            max_tokens=bcfg['max_tokens'],
            stop=bcfg.get('stop', []),
            device=bcfg.get('device', 'cuda'),
        )
    elif backend_type == 'vllm':
        llm_backend = VLLMBackend(
            model_path=bcfg['model_path'],
            temperature=bcfg['temperature'],
            max_tokens=bcfg['max_tokens'],
            stop=bcfg.get('stop', []),
            tensor_parallel_size=bcfg.get('tensor_parallel_size', 1),
            gpu_memory_utilization=bcfg.get('gpu_memory_utilization', 0.9),
        )
    elif backend_type == 'openai':
        llm_backend = OpenAIBackend(
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

    # 创建 Think 子模型
    think_model = AuctionNetThinkModel(
        llm_backend=llm_backend,
        verbose=config['model']['think_model']['verbose'],
    )

    # 创建 Act 子模型
    act_model = ActModel(
        model_path=config['model']['act_model']['path'],
        model_type=config['model']['act_model']['backend'],
        state_dim=config['strategy']['state_dim'],
        action_dim=config['strategy']['action_dim'],
        device=config['device'],
        state_mean=normalize_dict['state_mean'],
        state_std=normalize_dict['state_std'],
    )
    act_model.eval()

    # 创建组合模型
    model = AgentModel(think_model, act_model)

    strategy = AuctionNetBaseStrategy(model, window_size=config['strategy']['window_size'])

    keys = env.keys()
    print(f'Available keys: {len(keys)}')

    for key in keys[:1]:
        reset_info = env.reset(key)
        strategy.reset()
        strategy.set_episode_info(
            budget=reset_info['budget'],
            cpa_constraint=reset_info['cpa_constraint'],
            num_timesteps=reset_info['num_timesteps'],
            first_pvalue_mean=reset_info['first_pvalue_mean'],
            first_pv_num=reset_info['first_pv_num'],
        )
        print(f'Episode: budget={reset_info["budget"]}, cpa={reset_info["cpa_constraint"]}, steps={reset_info["num_timesteps"]}')

        total_gmv = 0
        total_cost = 0

        for step in range(1, reset_info['num_timesteps'] + 1):
            response, pacer = strategy.bidding()

            print(f'Step#{step}, pacer={pacer}')
            print(f'CoT: {response}')
            result = env.step(pacer)
            strategy.update(result)

            total_gmv += result['gmv']
            total_cost += result['cost']

            if result['done']:
                break

        print(f'Result: gmv={total_gmv:.2f}, cost={total_cost:.2f}, CPA={total_cost/(total_gmv+1e-9):.2f}')


if __name__ == '__main__':
    main()
