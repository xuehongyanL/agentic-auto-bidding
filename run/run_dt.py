import pickle

import numpy as np

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv
from agb_auctionnet.strategy.dt_strategy import AuctionNetDTStrategy
from agb_core.model.dt_model import DTModel

NORMALIZE_DICT_PATH = '/DATA/xuehy/ad/AAB/aab/saved_model/DTtest_stable_20260119131013/normalize_dict.pkl'


def main():
    with open(NORMALIZE_DICT_PATH, 'rb') as f:
        normalize_dict = pickle.load(f)

    env = AuctionNetEnv(data_filename='/DATA/xuehy/ad/AAB/data/traffic/period-7.csv')

    model = DTModel(
        model_path='/DATA/xuehy/ad/AAB/aab/saved_model/DTtest_stable_20260119131013/500000.pt',
        state_dim=16,
        act_dim=1,
        device='cpu',
        hidden_size=512,
        n_layer=8,
        n_head=16,
        n_inner=2048,
        state_mean=normalize_dict['state_mean'],
        state_std=normalize_dict['state_std'],
        scale=2000.0,
    )
    model.eval()

    strategy = AuctionNetDTStrategy(model)

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

        for step in range(1, reset_info["num_timesteps"] + 1):
            current_pvalues = env.get_current_pvalues()
            # 注入当前时间步的流量信息到 strategy（必须在 bidding 之前设置）
            strategy.cpm = float(np.mean(current_pvalues)) if current_pvalues.size > 0 else 0.0
            strategy.cpn = int(current_pvalues.size)
            pacer = strategy.bidding()
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
