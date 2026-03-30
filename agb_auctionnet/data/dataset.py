"""AuctionNet dataset implementation."""

import ast
import bisect
import hashlib

import numpy as np
import pandas as pd
import torch

from agb_core.data.dataset import BaseDataset
from agb_core.data.trajectory import Trajectory


def safe_literal_eval(val):
    if pd.isna(val):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


class AuctionNetDataset(BaseDataset):
    """AuctionNet dataset for bidding training."""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 window_size: int = 20,
                 scale: int = 2000):
        super().__init__(state_dim, action_dim)
        self._window_size = window_size
        self._scale = scale
        self._device = 'cpu'

    @classmethod
    def build(cls, csv_path: str, continuous: bool = False,
              split: int = 1, action_mode: str = 'price') -> list[dict]:
        df = pd.read_csv(csv_path)
        df['state'] = df['state'].apply(safe_literal_eval)

        _REWARD_COLUMN = 'reward_continuous' if continuous else 'reward'

        states, actions, rtgs = [], [], []
        state, action, rtg = [], [], []
        traj_infos = []
        serial = 0
        for _, row in df.iterrows():
            state.append(row['state'])
            action.append([row['action']])
            rtg.append(row[_REWARD_COLUMN])
            serial += 1
            if row['done']:
                if len(state) > 1:
                    state.append(state[-1])  # 最后一步的next_state

                    # reward转换为rtg
                    rtg.append(0.)
                    for i in range(len(rtg) - 2, -1, -1):
                        rtg[i] += rtg[i+1]

                    states.append(np.array(state))
                    if action_mode == 'pacer':
                        cpa = float(row['CPAConstraint'])
                        actions.append(np.array(action) / cpa)
                    else:
                        actions.append(np.array(action))
                    rtgs.append(np.array(rtg))
                    traj_infos.append({
                        'budget': float(row['budget']),
                        'cpa_constraint': float(row['CPAConstraint']),
                    })
                else:
                    serial -= 1  # 单步轨迹被过滤，撤销计数
                state, action, rtg = [], [], []

        thoughts = ['' for _ in range(serial)]
        step_infos = [{} for _ in range(serial)]

        # Compute shared state normalization stats from all trajectories
        all_states = np.concatenate(states, axis=0)
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0) + 1e-6

        # 按 episode 均分
        n_ep = len(states)
        part_size = n_ep // split
        parts = []
        global_step_offset = 0
        for part_idx in range(split):
            start = part_idx * part_size
            end = n_ep if part_idx == split - 1 else (part_idx + 1) * part_size

            part_states = states[start:end]
            part_actions = actions[start:end]
            part_rtgs = rtgs[start:end]
            part_traj_infos = traj_infos[start:end]
            part_thoughts = []
            part_step_infos = []
            for ep_idx in range(start, end):
                ep_len = len(states[ep_idx]) - 1
                part_thoughts.extend(thoughts[global_step_offset:global_step_offset + ep_len])
                part_step_infos.extend(step_infos[global_step_offset:global_step_offset + ep_len])
                global_step_offset += ep_len

            parts.append({
                'states': part_states,
                'actions': part_actions,
                'rtgs': part_rtgs,
                'thoughts': part_thoughts,
                'traj_infos': part_traj_infos,
                'step_infos': part_step_infos,
                'state_mean': state_mean,
                'state_std': state_std,
            })
        return parts

    def load(self, data_dicts: list[dict]) -> 'AuctionNetDataset':
        self.states = []
        self.actions = []
        self.rtgs = []
        self.thoughts = []
        self.traj_infos = []
        self.step_infos = []
        for part in data_dicts:
            self.states.extend(part['states'])
            self.actions.extend(part['actions'])
            self.rtgs.extend(part['rtgs'])
            self.thoughts.extend(part['thoughts'])
            self.traj_infos.extend(part['traj_infos'])
            self.step_infos.extend(part['step_infos'])
        self.state_mean = data_dicts[0]['state_mean']
        self.state_std = data_dicts[0]['state_std']

        # 根据 states 长度现场计算全局 indices
        self._traj_offsets = []
        offset = 0
        for states_arr in self.states:
            self._traj_offsets.append(offset)
            offset += len(states_arr) - 1  # 每条轨迹有 T+1 行（含 next_state），实际决策步数为 T
        self._total_steps = offset

        assert len(self.thoughts) == self._total_steps, (
            f'thoughts length {len(self.thoughts)} != total steps {self._total_steps}'
        )
        return self

    def _locate(self, index: int) -> tuple[int, int]:
        # 全局 index -> (轨迹 index, 时间步 index)
        traj_idx = bisect.bisect_right(self._traj_offsets, index) - 1
        step_idx = index - self._traj_offsets[traj_idx]
        return traj_idx, step_idx

    def __getitem__(self, index: int) -> tuple[Trajectory, str, dict, dict]:
        traj_idx, step_idx = self._locate(index)
        W = self._window_size

        # 计算padding长度
        pad_len = max(0, W - step_idx - 1)
        start = step_idx - W + 1 + pad_len

        states = self.states[traj_idx][start:step_idx + 2]
        rtgs = self.rtgs[traj_idx][start:step_idx + 2][:, np.newaxis]  # [T] -> [T, 1]
        actions = self.actions[traj_idx][start:step_idx + 1]
        timesteps = np.arange(start, step_idx + 1)  # 连续时间步
        attention_mask = np.ones((W, self._action_dim))
        thought = self.thoughts[index]

        if pad_len > 0:
            states = np.concatenate([np.zeros((pad_len, self._state_dim)), states])
            actions = np.concatenate([np.zeros((pad_len, self._action_dim)), actions])
            rtgs = np.concatenate([np.zeros((pad_len, 1)), rtgs])
            timesteps = np.concatenate([np.zeros(pad_len), timesteps])
            attention_mask[:pad_len, :] = 0

        traj = Trajectory(
            states=torch.from_numpy(states).float(),
            actions=torch.from_numpy(actions).float(),
            rtgs=torch.from_numpy(rtgs).float(),
            timesteps=torch.from_numpy(timesteps).long(),
            attention_mask=torch.from_numpy(attention_mask).long(),
        )
        return traj, thought, self.step_infos[index], self.traj_infos[traj_idx]

    def filter_thoughts(self, mode: str, default_action: float = 1.0, eps: float = 1e-6) -> tuple[int, int]:
        """
        根据 cot_label 与实际调价方向的一致性过滤 thoughts。

        参数:
            mode: 过滤模式
                - 'match': 方向必须完全一致 (actual_label == cot_label)
                - 'nonconflict': 方向不冲突 (正/负 与 中性 不冲突，正向≠负向)
                - 'any': 不过滤方向，只过滤解析失败 (cot_label != 0)
            default_action: 第-1步（第一步前）的默认基准价格
            eps: 浮点数比较的容差

        返回:
            (n_keep, n_filtered): 保留和被过滤的样本数
        """
        assert mode in ('match', 'nonconflict', 'any')

        n_keep = 0
        n_filtered = 0

        for index in range(self._total_steps):
            traj_idx, step_idx = self._locate(index)
            actions = self.actions[traj_idx]  # [T]

            # 确定基准价格：第0步用default_action，否则用前一步的action
            if step_idx == 0:
                prev_action = default_action
            else:
                prev_action = float(actions[step_idx - 1])

            current_action = float(actions[step_idx])

            # 计算实际调价方向: 1=负向, 2=中性, 3=正向
            if current_action < prev_action - eps:
                actual_label = 1
            elif current_action > prev_action + eps:
                actual_label = 3
            else:
                actual_label = 2

            cot_label = self.step_infos[index].get('cot_label', 0)
            if cot_label == 0:
                self.thoughts[index] = ''
                n_filtered += 1
                continue

            if mode == 'match':
                keep = (actual_label == cot_label)
            elif mode == 'nonconflict':
                # 中性(2)与任何方向都不冲突；正向(3)与负向(1)冲突
                if actual_label == 2 or cot_label == 2:
                    keep = True
                else:
                    keep = (actual_label == cot_label)
            else:  # 'any'
                keep = True

            if not keep:
                self.thoughts[index] = ''
                n_filtered += 1
            else:
                n_keep += 1

        return n_keep, n_filtered

    def __len__(self) -> int:
        return self._total_steps
