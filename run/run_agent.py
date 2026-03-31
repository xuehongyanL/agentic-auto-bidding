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
    parser.add_argument('--verbose', type=int, default=0,
                        help='0: no output, 1: episode summary, 2: step details')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    config = parser.apply_overrides(config)

    # 创建 Think 子模型
    bcfg = config['model']['think']['llm_backend']
    llm_backend = build_llm_backend(bcfg)
    think_model = AuctionNetThinkModel(
        llm_backend=llm_backend,
        verbose=config['model']['think']['verbose'],
    )

    # 创建 Act 子模型
    act_cfg = config['model']['act']
    act_model = ActModel(
        model_path=act_cfg['path'],
        model_type=act_cfg['backend'],
        state_dim=config['task']['state_dim'],
        action_dim=config['task']['action_dim'],
        device=act_cfg['device'],
    )

    act_model.load_model(args.act_ckpt)
    act_model.eval()

    # 创建组合模型
    agent_model = AgentModel(think_model, act_model)

    think_batch_size = config['infer']['test']['think_batch_size']
    act_batch_size = config['infer']['test']['act_batch_size']
    metrics = evaluate(agent_model, config, split='test', verbose=args.verbose,
                      think_batch_size=think_batch_size, act_batch_size=act_batch_size)
    print(
        f'=== Overall (think={think_batch_size}, act={act_batch_size}) ===\n'
        f'gmv={metrics['avg_gmv']:.2f} | '
        f'cost={metrics['avg_cost']:.2f} | '
        f'cpa_ratio={metrics['avg_cpa_ratio']:.2f} | '
        f'score={metrics['avg_score']:.2f}'
    )


if __name__ == '__main__':
    main()
