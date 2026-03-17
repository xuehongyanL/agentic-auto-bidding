import argparse
import pickle

import numpy as np
import yaml

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv
from agb_auctionnet.model.think_model import AuctionNetThinkModel
from agb_auctionnet.strategy.base_strategy import AuctionNetBaseStrategy
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

    # 创建 Think 子模型
    think_model = AuctionNetThinkModel(
        model_path=config['model']['think_model']['path'],
        model_type=config['model']['think_model']['backend'],
        device=config['device'],
        temperature=config['model']['think_model']['temperature'],
        max_tokens=config['model']['think_model']['max_tokens'],
        verbose=config['model']['think_model']['verbose'],
    )

    # 创建 Act 子模型
    act_model = ActModel(
        model_path=config['model']['act_model']['path'],
        model_type=config['model']['act_model']['backend'],
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
        )
        print(f'Episode: budget={reset_info["budget"]}, cpa={reset_info["cpa_constraint"]}, steps={reset_info["num_timesteps"]}')

        total_gmv = 0
        total_cost = 0

        for step in range(1, reset_info['num_timesteps'] + 1):
            current_pvalues = env.get_current_pvalues()
            strategy.cpm = float(np.mean(current_pvalues)) if current_pvalues.size > 0 else 0.0
            strategy.cpn = int(current_pvalues.size)

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
