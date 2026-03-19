import argparse
import pickle

import yaml

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv
from agb_auctionnet.strategy.base_strategy import AuctionNetBaseStrategy
from agb_core.model.dt_model import DTModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    with open(config['model']['normalize_dict_path'], 'rb') as f:
        normalize_dict = pickle.load(f)

    env = AuctionNetEnv(data_filename=config['env']['data_path'])

    model = DTModel(
        state_dim=config['strategy']['state_dim'],
        action_dim=config['strategy']['action_dim'],
        device=config['device'],
        hidden_size=config['model']['hidden_size'],
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_inner=config['model']['n_inner'],
        scale=config['model']['scale'],
        target_rtg=config['model']['target_rtg'],
        block_config=config['model']['block_config'],
        output_mode=config['model']['output_mode'],
    ).load_model(config['model']['path'])

    strategy = AuctionNetBaseStrategy(
        model,
        window_size=config['strategy']['window_size'],
        state_mean=normalize_dict['state_mean'],
        state_std=normalize_dict['state_std'],
    )

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

        for step in range(1, reset_info["num_timesteps"] + 1):
            _, pacer = strategy.bidding()
            print(f'Step#{step}, pacer={pacer}')
            result = env.step(pacer)
            strategy.update(result)

            total_gmv += result['gmv']
            total_cost += result['cost']

            if result['done']:
                break

        print(f'Result: gmv={total_gmv:.2f}, cost={total_cost:.2f}, CPA={total_cost/(total_gmv+1e-9):.2f}')


if __name__ == '__main__':
    main()
