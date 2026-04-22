import pickle

import yaml

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv
from agb_auctionnet.strategy.base_strategy import AuctionNetBaseStrategy
from agb_core.model.dt_model import DTModel
from agb_core.utils.argparse import ArgumentParser
from agb_core.utils.path import glob_data_paths


def getScore_nips(reward, cpa_ratio):
    """cpa_ratio > 1 时以 (1 / cpa_ratio)^2 惩罚 reward"""
    beta = 2
    penalty = 1
    if cpa_ratio > 1:
        coef = 1 / (cpa_ratio + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config = parser.apply_overrides(config)

    data_filenames = [str(p) for p in glob_data_paths(config['env']['data_path'])]
    env = AuctionNetEnv(data_filenames=data_filenames,
                        use_continuous_reward=True)

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
        max_timestep_len=config['model']['max_timestep_len'],
    ).load_model(config['model']['path'])

    if normalize_dict_path := config['model'].get('normalize_dict_path'):
        with open(normalize_dict_path, 'rb') as f:
            normalize_dict = pickle.load(f)
        model.set_normalize(normalize_dict['state_mean'], normalize_dict['state_std'])

    strategy = AuctionNetBaseStrategy(
        model,
        window_size=config['strategy']['window_size'],
    )

    keys = env.keys()
    print(f'Available keys: {len(keys)}')

    all_gmvs: list[float] = []
    all_costs: list[float] = []
    all_scores: list[float] = []

    for key in keys:
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

        total_gmv = 0.0
        total_cost = 0.0

        for step in range(1, reset_info['num_timesteps'] + 1):
            _, pacer = strategy.bidding()
            # print(f'Step#{step}, pacer={pacer}')
            result = env.step(pacer)
            strategy.update(result)

            total_gmv += result['gmv']
            total_cost += result['cost']

            if result['done']:
                break

        conversions = total_gmv / (reset_info['cpa_constraint'] + 1e-9)
        cpa_ratio = total_cost / (total_gmv + 1e-9) if total_gmv > 0 else 0.0
        score = getScore_nips(conversions, cpa_ratio)

        all_gmvs.append(total_gmv)
        all_costs.append(total_cost)
        all_scores.append(score)

        print(f'Result: gmv={total_gmv:.2f}, cost={total_cost:.2f}, CPA={cpa_ratio:.2f}, score={score:.4f}')

    n_ep = len(all_gmvs)
    avg_gmv = sum(all_gmvs) / n_ep
    avg_cost = sum(all_costs) / n_ep
    avg_cpa_ratio = sum(all_costs) / (sum(all_gmvs) + 1e-9)
    avg_score = sum(all_scores) / n_ep

    print(f'\n=== Aggregated ({n_ep} episodes) ===')
    print(f'avg_gmv:      {avg_gmv:.4f}')
    print(f'avg_cost:     {avg_cost:.4f}')
    print(f'avg_cpa_ratio:{avg_cpa_ratio:.4f}')
    print(f'avg_score:    {avg_score:.4f}')


if __name__ == '__main__':
    main()
