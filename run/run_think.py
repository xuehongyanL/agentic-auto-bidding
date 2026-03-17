import numpy as np

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv
from agb_auctionnet.model.think_model import AuctionNetThinkModel
from agb_auctionnet.strategy.base_strategy import AuctionNetBaseStrategy


def main():
    env = AuctionNetEnv(data_filename='/DATA/xuehy/ad/AAB/data/traffic/period-7.csv')

    window_size = 20

    model = AuctionNetThinkModel(
        model_path='/DATA/xuehy/agent/models/Qwen/Qwen2.5-3B-Instruct',
        model_type='vllm',
        device='cuda',
        verbose=1,
    )

    strategy = AuctionNetBaseStrategy(model, window_size=window_size)

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

            response, direction = strategy.bidding()  # response 是文本，direction 是 numpy array

            # 手动映射为真实 pacer
            direction_scalar = int(direction.flatten()[0])
            if direction_scalar == -1:
                pacer = np.array([0.8])
            elif direction_scalar == 0:
                pacer = np.array([1.0])
            else:  # direction_scalar == 1
                pacer = np.array([1.2])

            # 修改 strategy 内部记录的最近一个 pacer
            if strategy._history_pacers:
                strategy._history_pacers[-1] = pacer
            strategy._last_pacer = pacer

            print(f'Step#{step}, direction={direction}, pacer={pacer}')
            result = env.step(pacer)
            strategy.update(result)

            total_gmv += result['gmv']
            total_cost += result['cost']

            if result['done']:
                break

        print(f'Result: gmv={total_gmv:.2f}, cost={total_cost:.2f}, CPA={total_cost/(total_gmv+1e-9):.2f}')


if __name__ == '__main__':
    main()
