import argparse
import pickle
import re

import numpy as np
import yaml

from agb_auctionnet.model.think_model import AuctionNetThinkModel
from agb_core.infer.llm_backend import VLLMBackend


def parse_cot_label(thought: str) -> int:
    """
    从 thought 字符串中解析 CoT 标签，返回 -1 / 0 / 1, 映射为 1 / 2 / 3 ,或解析失败时返回 0。
    参考 AuctionNetThinkModel._parse_response 的逻辑。
    """
    thought = thought.strip()
    # 尝试从 <answer> 标签中提取 -1/0/1
    answer_match = re.search(r'<answer>\s*(-?1|0)\s*</answer>', thought)
    if answer_match:
        return int(answer_match.group(1)) + 2
    # 尝试直接匹配 -1/0/1
    direction_match = re.search(r'\b(-1|0|1)\b', thought)
    if direction_match:
        return int(direction_match.group(1)) + 2
    return 0


def build_context(ep_states, ep_actions, step, budget, cpa_constraint, window_size=20):
    """
    从 episode 的 states/actions 重建 prompt 所需的 context。

    Think 的历史不包含当前步（step），与推理阶段完全对齐。
    step=0 时历史为空。
    """
    history_pacer = [np.array([float(ep_actions[i].flatten()[0])]) for i in range(step)]
    history_pv_num = [int(ep_states[i][13]) if len(ep_states[i]) > 13 else 0
                      for i in range(step)]
    history_conversion = [float(ep_states[i][6]) if len(ep_states[i]) > 6 else 0.0
                          for i in range(step)]

    # budget_left = state[1]，初始为 1.0
    history_total_cost = [
        budget * (1.0 - float(ep_states[i][1])) if len(ep_states[i]) > 1 else 0.0
        for i in range(step)
    ]

    return {
        'budget': budget,
        'cpa_constraint': cpa_constraint,
        'window_size': window_size,
        'num_timesteps': len(ep_states) - 1,  # 从数据推导
        'history_pacer': history_pacer,
        'history_pv_num': history_pv_num,
        'history_conversion': history_conversion,
        'history_total_cost': history_total_cost,
        'total_conversions': sum(history_conversion),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/agent_auctionnet.yaml')
    parser.add_argument('--part_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    llm_cfg = config['model']['think']['llm_backend']
    task_cfg = config.get('task', {})
    window_size = task_cfg.get('window_size', 20)

    # 读取 pkl
    print(f'Loading data_dict from: {args.part_path}')
    with open(args.part_path, 'rb') as f:
        data_dict = pickle.load(f)

    states_list = data_dict['states']
    actions_list = data_dict['actions']
    traj_infos_list = data_dict['traj_infos']
    step_infos = data_dict['step_infos']
    total_steps = sum(len(s) - 1 for s in states_list)
    thoughts = [''] * total_steps
    print(f'Episodes: {len(states_list)}, total steps: {total_steps}')

    # 构建推理模型
    print(f'Building ThinkModel ({llm_cfg.get("type", "vllm")}) ...')
    llm_backend = VLLMBackend(
        model_path=llm_cfg['model_path'],
        temperature=llm_cfg.get('temperature', 0.0),
        max_tokens=llm_cfg.get('max_tokens', 1024),
        tensor_parallel_size=llm_cfg.get('tensor_parallel_size', 1),
        gpu_memory_utilization=llm_cfg.get('gpu_memory_utilization', 0.9),
    )
    think_model = AuctionNetThinkModel(llm_backend=llm_backend, verbose=0)

    from vllm import SamplingParams
    tokenizer = llm_backend._llm.get_tokenizer()

    # 现场计算 episode 起始 offset
    ep_offsets = []
    offset = 0
    for s in states_list:
        ep_offsets.append(offset)
        offset += len(s) - 1

    prompts = []
    global_indices = []
    for ep_idx in range(len(states_list)):
        ep_states = states_list[ep_idx]
        ep_actions = actions_list[ep_idx]
        T = len(ep_states) - 1
        traj_info = traj_infos_list[ep_idx]
        ep_offset = ep_offsets[ep_idx]
        for step in range(T):
            context = build_context(ep_states, ep_actions, step,
                                   traj_info['budget'], traj_info['cpa_constraint'],
                                   window_size=window_size)
            user_prompt = think_model._format_user_prompt(context)
            system_prompt = think_model._get_system_prompt()
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            global_indices.append(ep_offset + step)

    print(f'Built {len(prompts)} prompts')

    # 批量推理
    for i in range(0, len(prompts), args.batch_size):
        chunk = prompts[i:i + args.batch_size]
        chunk_idx = global_indices[i:i + args.batch_size]
        outputs = llm_backend._llm.generate(chunk, SamplingParams(
            temperature=llm_cfg.get('temperature', 0.0),
            max_tokens=llm_cfg.get('max_tokens', 1024),
        ))
        for global_idx, output in zip(chunk_idx, outputs):
            thoughts[global_idx] = output.outputs[0].text
        print(f'  Batch {i // args.batch_size + 1}: {len(chunk)} prompts done')

    print(f'Processed {len(prompts)} decision points')

    # 解析 CoT 方向标签
    for global_idx, thought in enumerate(thoughts):
        if thought:
            step_infos[global_idx]['cot_label'] = parse_cot_label(thought)

    # 保存
    data_dict['thoughts'] = thoughts
    data_dict['step_infos'] = step_infos
    with open(args.output_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f'Saved to {args.output_path}')


if __name__ == '__main__':
    main()
