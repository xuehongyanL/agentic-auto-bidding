import argparse

import yaml


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_argument(
            '--set', '-S', dest='config_overrides', nargs='+', action='extend',
            metavar='KEY.PATH=VALUE', default=[],
            help='Override config values. Example: -S train.act.learning_rate=0.0001'
        )

    def apply_overrides(self, config: dict) -> dict:
        """将 --set 参数注入到 config 并返回。"""
        overrides = self.parse_args().config_overrides
        for ov in overrides:
            key_str, _, value_str = ov.partition('=')
            if not value_str:
                raise ValueError(f"Override '{ov}' missing '=' separator")
            key_path = key_str.strip().split('.')

            d = config
            for part in key_path[:-1]:
                if part not in d:
                    raise KeyError(f"Override path '{key_str}': key '{part}' not found")
                d = d[part]

            final_key = key_path[-1]
            if final_key not in d:
                raise KeyError(f"Override path '{key_str}': leaf key '{final_key}' not found")

            d[final_key] = yaml.safe_load(value_str)

        return config
