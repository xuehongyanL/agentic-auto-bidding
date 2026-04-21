import pickle

import yaml

from agb_auctionnet.infer.evaluate import evaluate
from agb_auctionnet.model.think_model import AuctionNetThinkModel
from agb_core.model.act_model import ActModel
from agb_core.model.agent_model import AgentModel
from agb_core.utils.argparse import ArgumentParser
from agb_core.utils.llm_backend import build_llm_backend


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--act_ckpt', type=str, required=True,
                        help='Checkpoint path for act model')
    parser.add_argument('--split', type=str, default='test',
                        choices=['valid', 'test', 'explore'],
                        help='Data split to evaluate on')
    parser.add_argument('--out', type=str, default=None,
                        help='Path to save trajectories as pkl')
    parser.add_argument('--verbose', type=int, default=0,
                        help='0: no output, 1: episode summary, 2: step details')
    parser.add_argument('--think_ckpt', type=str, default=None,
                        help='Full-parameter think model dir. If omitted, uses base model from config.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config = parser.apply_overrides(config)

    # 创建 Think 子模型
    bcfg = config['model']['think']['llm_backend']
    if args.think_ckpt:
        bcfg = {**bcfg, 'model_path': args.think_ckpt}

    # explore 模式下应用专属的采样参数
    if args.split == 'explore':
        if override_temperature := config['infer']['explore'].get('override_temperature'):
            bcfg['temperature'] = override_temperature
        if override_top_p := config['infer']['explore'].get('override_top_p'):
            bcfg['top_p'] = override_top_p

    # test 模式不需要 skip_think
    if args.split == 'test':
        first_try_skip_think = False
    else:
        first_try_skip_think = True

    llm_backend = build_llm_backend(bcfg)
    think_model = AuctionNetThinkModel(
        llm_backend=llm_backend,
        verbose=config['model']['think']['verbose'],
    )

    # 创建 Act 子模型
    act_cfg = config['model']['act']
    act_model = ActModel(
        base_model_path=act_cfg['path'],
        model_type=act_cfg['backend'],
        state_dim=config['task']['state_dim'],
        action_dim=config['task']['action_dim'],
        device=act_cfg['device'],
    )

    act_model.load_model(args.act_ckpt)
    act_model.eval()

    think_batch_size = config['infer']['think_batch_size']
    act_batch_size = config['infer']['act_batch_size']
    agent_model = AgentModel(
        think_model, act_model,
        think_batch_size=think_batch_size,
        act_batch_size=act_batch_size,
    )
    metrics, trajectories = evaluate(agent_model,
                                     config,
                                     split=args.split,
                                     first_try_skip_think=first_try_skip_think,
                                     verbose=args.verbose)
    if args.out:
        with open(args.out, 'wb') as f:
            pickle.dump(trajectories, f)
    print(
        f'=== Overall (think={think_batch_size}, act={act_batch_size}) ===\n'
        f'gmv={metrics['avg_gmv']:.2f} | '
        f'cost={metrics['avg_cost']:.2f} | '
        f'cpa_ratio={metrics['avg_cpa_ratio']:.2f} | '
        f'score={metrics['avg_score']:.2f}'
    )


if __name__ == '__main__':
    main()
