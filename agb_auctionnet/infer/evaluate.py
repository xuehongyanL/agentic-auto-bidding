import math
from typing import Any

import torch

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv, AuctionNetMultiEnv
from agb_auctionnet.strategy.agent_strategy import AuctionNetAgentMultiStrategy
from agb_core.model.agent_model import AgentModel
from agb_core.utils.path import glob_data_paths


def getScore_nips(reward, cpa_ratio):
    """cpa_ratio > 1 时以 (1 / cpa_ratio)^2 惩罚 reward"""
    beta = 2
    penalty = 1
    if cpa_ratio > 1:
        coef = 1 / (cpa_ratio + 1e-10)
        penalty = pow(coef, beta)
    return penalty * reward


def evaluate(
    agent_model: AgentModel,
    config: dict,
    split: str = 'valid',
    verbose: int = 0,
    think_batch_size: int = 1,
    act_batch_size: int = 1,
) -> dict[str, Any]:
    """
    在模拟环境中评估模型，返回 GMV / Cost / cpa_ratio / 综合评分等指标。
    每批 env 数量为 think_batch_size 和 act_batch_size 的最小公倍数，
    每步内 think 和 act 分别按各自的批大小分块执行。

    Args:
        agent_model: 已构造好的 AgentModel
        config: 完整配置 dict，需包含 task.window_size、infer.{split}.env
        split: 环境数据切分，可选 'valid' 或 'test'，对应 config['infer'][split]['env']
        verbose: 打印详细程度，0 不打印，1 打印每个 episode 汇总，2 额外打印每个 step 的 pacer 和 CoT
        think_batch_size: think 阶段的批大小
        act_batch_size: act 阶段的批大小
    """
    env_cfg = config['infer'][split]['env']

    # 解析 data_path，支持 glob 模式
    data_filenames = [str(p) for p in glob_data_paths(env_cfg['data_path'])]

    # 数据只加载一次，获取所有 key
    tmp_env = AuctionNetEnv(data_filenames=data_filenames)
    keys = tmp_env.keys()

    window_size = config['task']['window_size']
    all_gmvs: list[float] = []
    all_costs: list[float] = []
    all_scores: list[float] = []

    # 每批 env 数量为 think_batch_size 和 act_batch_size 的最小公倍数
    n_envs = math.lcm(think_batch_size, act_batch_size)

    # 分批并行：每批 n_envs 个 key 同时运行
    for batch_start in range(0, len(keys), n_envs):
        batch_keys = keys[batch_start:batch_start + n_envs]
        batch_size = len(batch_keys)

        # 数据只加载一次，menstrat 共享同一份底层 env 实例
        menv = AuctionNetMultiEnv(n_envs=batch_size, data_filenames=data_filenames)
        mstrategy = AuctionNetAgentMultiStrategy(agent_model, n_strategies=batch_size, window_size=window_size)

        reset_infos = menv.reset(batch_keys)
        mstrategy.reset()
        mstrategy.set_episode_info_batch(reset_infos)

        num_timesteps = reset_infos[0]['num_timesteps']

        for bi, (key, info) in enumerate(zip(batch_keys, reset_infos)):
            if verbose >= 1:
                print(
                    f'Episode [{key}]: budget={info["budget"]:.2f}, '
                    f'cpa={info["cpa_constraint"]:.2f}, steps={info["num_timesteps"]}'
                )

        batch_gmvs = [0.0] * batch_size
        batch_costs = [0.0] * batch_size

        for step_i in range(1, num_timesteps + 1):
            bidding_results = mstrategy.bidding_chunked(think_batch_size, act_batch_size)
            pacers = [r[1] for r in bidding_results]

            if verbose >= 2:
                pacer_strs = [f'{p[0]:.3f}' for p in pacers]
                print(f'  Step#{step_i}: pacers={pacer_strs}')

            results = menv.step(pacers)
            mstrategy.update_batch(results)

            for bi, result in enumerate(results):
                batch_gmvs[bi] += result['gmv']
                batch_costs[bi] += result['cost']

        # 收集结果
        for bi in range(batch_size):
            all_gmvs.append(batch_gmvs[bi])
            all_costs.append(batch_costs[bi])

            conversions = batch_gmvs[bi] / (reset_infos[bi]['cpa_constraint'] + 1e-9)
            cpa_ratio = batch_costs[bi] / (batch_gmvs[bi] + 1e-9) if batch_gmvs[bi] > 0 else 0.0
            all_scores.append(getScore_nips(conversions, cpa_ratio))

            if verbose >= 1:
                print(
                    f'  [{batch_keys[bi]}] Result: gmv={batch_gmvs[bi]:.2f}, cost={batch_costs[bi]:.2f}, '
                    f'cpa_ratio={cpa_ratio:.2f}, score={all_scores[-1]:.2f}'
                )

        # 每批处理完后清理 CUDA 缓存，避免显存碎片累积
        torch.cuda.empty_cache()

    n_ep = len(all_gmvs)
    avg_gmv = sum(all_gmvs) / n_ep if n_ep > 0 else 0.0
    avg_cost = sum(all_costs) / n_ep if n_ep > 0 else 0.0
    avg_cpa_ratio = sum(all_costs) / (sum(all_gmvs) + 1e-9) if n_ep > 0 else 0.0
    avg_score = sum(all_scores) / n_ep if n_ep > 0 else 0.0

    return {
        'avg_gmv': avg_gmv,
        'avg_cost': avg_cost,
        'avg_cpa_ratio': avg_cpa_ratio,
        'avg_score': avg_score,
        'n_episodes': n_ep,
    }
