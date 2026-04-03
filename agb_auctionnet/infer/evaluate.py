import math
from typing import Any

import numpy as np
import torch

from agb_auctionnet.env.auctionnet_env import AuctionNetEnv, AuctionNetMultiEnv
from agb_auctionnet.strategy.base_strategy import AuctionNetMultiStrategy
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
    use_continuous_reward: bool = True,
    first_try_skip_think: bool = True,
    verbose: int = 0,
) -> tuple[dict[str, Any], list[dict]]:
    """
    在模拟环境中评估模型，返回 GMV / Cost / cpa_ratio / 综合评分等指标。
    batch size 由 agent_model 构造时注入，predict_batch 内部自动分块执行。

    Args:
        agent_model: 已构造好的 AgentModel
        config: 完整配置 dict，需包含 task.window_size、infer.{split}.env
        split: 环境数据切分，可选 'valid' 或 'test'，对应 config['infer'][split]['env']
        verbose: 打印详细程度，0 不打印，1 打印每个 episode 汇总，2 额外打印每个 step 的 pacer 和 CoT
        first_try_skip_think: 若为 True，第一次采样时跳过 Think 模型推理，用空字符串代替

    Returns:
        (metrics, trajectories):
            metrics: {avg_gmv, avg_cost, avg_cpa_ratio, avg_score, n_episodes}
            trajectories: list[dict]，每 episode 一条，结构为：
                {
                    'budget': float,
                    'cpa_constraint': float,
                    'steps': list[dict],  # 每步一个，结构为：
                        # {thoughts, scores, prompt, pacers, cost, gmv}
                    'total_cost': float,
                    'total_gmv': float,
                }
    """
    env_cfg = config['infer'][split]['env']

    # 解析 data_path，支持 glob 模式
    data_filenames = [str(p) for p in glob_data_paths(env_cfg['data_path'])]

    # 数据只加载一次，获取所有 key
    tmp_env = AuctionNetEnv(data_filenames=data_filenames,
                            use_continuous_reward=use_continuous_reward)
    keys = tmp_env.keys()

    window_size = config['task']['window_size']
    all_gmvs: list[float] = []
    all_costs: list[float] = []
    all_scores: list[float] = []
    all_trajectories: list[dict] = []

    # 多次采样执行最优动作，仅探索模式可用
    n_sample = config['infer'][split].get('n_sample', 1)

    if first_try_skip_think and n_sample == 1:
        print('[WARNING] n_sample=1 with first_try_skip_think=True: Think 模型将被完全跳过')

    # 每批 env 数量为 think_batch_size 和 act_batch_size 的最小公倍数
    n_envs = math.lcm(agent_model._think_batch_size, agent_model._act_batch_size)

    # 分批并行：每批 n_envs 个 key 同时运行
    for batch_start in range(0, len(keys), n_envs):
        batch_keys = keys[batch_start:batch_start + n_envs]
        batch_size = len(batch_keys)

        # 数据只加载一次，menstrat 共享同一份底层 env 实例
        menv = AuctionNetMultiEnv(n_envs=batch_size,
                                  data_filenames=data_filenames,
                                  use_continuous_reward=use_continuous_reward)
        mstrategy = AuctionNetMultiStrategy(agent_model, n_strategies=batch_size, window_size=window_size)

        reset_infos = menv.reset(batch_keys)
        mstrategy.reset()
        mstrategy.set_episode_info_batch(reset_infos)

        num_timesteps = reset_infos[0]['num_timesteps']

        for bi, (key, info) in enumerate(zip(batch_keys, reset_infos)):
            if verbose >= 1:
                print(
                    f'Episode [{key}]: budget={info['budget']:.2f}, '
                    f'cpa={info['cpa_constraint']:.2f}, steps={info['num_timesteps']}'
                )

        batch_gmvs = [0.0] * batch_size
        batch_costs = [0.0] * batch_size

        # 初始化 batch 维度的轨迹记录
        batch_trajs: list[dict] = [
            {
                'key': key,
                'budget': info['budget'],
                'cpa_constraint': info['cpa_constraint'],
                'steps': [],
                'total_cost': 0.0,
                'total_gmv': 0.0,
            }
            for key, info in zip(batch_keys, reset_infos)
        ]

        # 记录每步采样结果的临时容器
        step_thoughts: list[list[str]] = [[] for _ in range(batch_size)]
        step_scores: list[list[float]] = [[] for _ in range(batch_size)]
        step_pacers: list[list[list[float]]] = [[] for _ in range(batch_size)]

        for step_i in range(1, num_timesteps + 1):
            # 预处理：构造一次 context 和 merged_traj（无副作用）
            contexts, merged_traj = mstrategy.pre_bidding()

            # 记录本步的 prompt（所有 sample 相同，只存一份）
            step_prompts: list[list[dict]] = [agent_model.get_prompt_messages(ctx) for ctx in contexts]

            # 多次采样：predict + step_to_end 评估，贪心选择最优 pacer
            best_pacers = [np.zeros(agent_model._action_dim)] * batch_size
            best_scores = [-float('inf')] * batch_size

            for sample_i in range(n_sample):
                skip_think = first_try_skip_think and (sample_i == 0)
                all_responses, actions = agent_model.predict_batch(contexts, merged_traj, skip_think=skip_think)
                candidate_pacers = mstrategy.post_bidding(responses=[], actions=actions)
                pacer_list = [p[1] for p in candidate_pacers]

                sim_results = menv.step_to_end(pacer_list)

                for bi in range(batch_size):
                    steps = sim_results[bi]
                    # 前缀：当前步之前已执行的历史累积
                    prefix_gmv = batch_gmvs[bi]
                    prefix_cost = batch_costs[bi]
                    # 后缀：执行该动作到 episode 结束的所有步
                    suffix_gmv = sum(s.get('gmv', 0.0) for s in steps)
                    suffix_cost = sum(s.get('cost', 0.0) for s in steps)

                    total_gmv = prefix_gmv + suffix_gmv
                    total_cost = prefix_cost + suffix_cost
                    total_conversions = total_gmv / (reset_infos[bi]['cpa_constraint'] + 1e-9)
                    cpa_ratio = total_cost / (total_gmv + 1e-9) if total_gmv > 0 else 0.0
                    score = getScore_nips(total_conversions, cpa_ratio)

                    if score > best_scores[bi]:
                        best_scores[bi] = score
                        best_pacers[bi] = pacer_list[bi]

                    # 按探索顺序记录（平分时保持先探索先记录的顺序）
                    step_thoughts[bi].append(all_responses[bi])
                    step_scores[bi].append(score)
                    step_pacers[bi].append(pacer_list[bi].tolist())

            if verbose >= 2:
                pacer_strs = [f'{p[0]:.3f}' for p in best_pacers]
                print(f'  Step#{step_i}: pacers={pacer_strs}')

            # 用最优 pacer 执行真实一步
            results = menv.step(best_pacers)
            mstrategy.update_batch(results)

            for bi, result in enumerate(results):
                batch_gmvs[bi] += result['gmv']
                batch_costs[bi] += result['cost']

                # 写入本步的采样记录和真实执行结果
                batch_trajs[bi]['steps'].append({
                    'thoughts': step_thoughts[bi],
                    'scores': step_scores[bi],
                    'prompt': step_prompts[bi],
                    'pacers': step_pacers[bi],
                    'cost': results[bi]['cost'],
                    'gmv': results[bi]['gmv'],
                })

            # 重置本步采样记录容器，供下一步使用
            step_thoughts = [[] for _ in range(batch_size)]
            step_scores = [[] for _ in range(batch_size)]
            step_pacers = [[] for _ in range(batch_size)]

        # 收集结果
        for bi in range(batch_size):
            all_gmvs.append(batch_gmvs[bi])
            all_costs.append(batch_costs[bi])

            conversions = batch_gmvs[bi] / (reset_infos[bi]['cpa_constraint'] + 1e-9)
            cpa_ratio = batch_costs[bi] / (batch_gmvs[bi] + 1e-9) if batch_gmvs[bi] > 0 else 0.0
            all_scores.append(getScore_nips(conversions, cpa_ratio))

            # 填入 episode 轨迹记录的最终结果
            batch_trajs[bi]['total_cost'] = batch_costs[bi]
            batch_trajs[bi]['total_gmv'] = batch_gmvs[bi]
            all_trajectories.append(batch_trajs[bi])

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
    }, all_trajectories
