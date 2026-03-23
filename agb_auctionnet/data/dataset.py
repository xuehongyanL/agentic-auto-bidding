"""AuctionNet dataset implementation."""

import ast
import bisect

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
    def build(cls, csv_path: str, continuous: bool = False) -> dict:
        df = pd.read_csv(csv_path)
        df['state'] = df['state'].apply(safe_literal_eval)

        _REWARD_COLUMN = 'reward_continuous' if continuous else 'reward'

        states, actions, rtgs = [], [], []
        state, action, rtg = [], [], []
        metadata = []
        serial = 0
        for _, row in df.iterrows():
            state.append(row['state'])
            action.append([row['action']])
            rtg.append(row[_REWARD_COLUMN])
            serial += 1
            if row['done']:
                state.append(state[-1])  # 最后一步的next_state

                # reward转换为rtg
                rtg.append(0.)
                for i in range(len(rtg) - 2, -1, -1):
                    rtg[i] += rtg[i+1]

                states.append(np.array(state))
                actions.append(np.array(action))
                rtgs.append(np.array(rtg))
                metadata.append({
                    'budget': float(row['budget']),
                    'cpa_constraint': float(row['CPAConstraint']),
                })
                state, action, rtg = [], [], []

        thoughts = ['' for _ in range(serial)]

        # Compute state normalization stats from all trajectories
        all_states = np.concatenate(states, axis=0)
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0) + 1e-6

        data_dict = {
            'states': states,
            'actions': actions,
            'rtgs': rtgs,
            'thoughts': thoughts,
            'metadata': metadata,
            'state_mean': state_mean,
            'state_std': state_std,
        }
        return data_dict

    def load(self, data_dict: dict) -> 'AuctionNetDataset':
        self.states = data_dict['states']
        self.actions = data_dict['actions']
        self.rtgs = data_dict['rtgs']
        self.thoughts = data_dict['thoughts']
        self.metadata = data_dict['metadata']
        self.state_mean = data_dict['state_mean']
        self.state_std = data_dict['state_std']

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

    def __getitem__(self, index: int) -> tuple[Trajectory, str, dict]:
        traj_idx, step_idx = self._locate(index)
        W = self._window_size

        # 计算padding长度
        pad_len = max(0, W - step_idx - 1)
        start = step_idx - W + 1 + pad_len

        states = self.states[traj_idx][start:step_idx + 2]
        rtgs = self.rtgs[traj_idx][start:step_idx + 2]
        actions = self.actions[traj_idx][start:step_idx + 1]
        timesteps = np.arange(start, step_idx + 1)  # 连续时间步
        attention_mask = np.ones((W, self._action_dim))
        thought = self.thoughts[index]

        if pad_len > 0:
            states = np.concatenate([np.zeros((pad_len, self._state_dim)), states])
            actions = np.concatenate([np.zeros((pad_len, self._action_dim)), actions])
            rtgs = np.concatenate([np.zeros(pad_len), rtgs])
            timesteps = np.concatenate([np.zeros(pad_len), timesteps])
            attention_mask[:pad_len, :] = 0

        traj = Trajectory(
            states=torch.from_numpy(states).float(),
            actions=torch.from_numpy(actions).float(),
            rtgs=torch.from_numpy(rtgs).float(),
            timesteps=torch.from_numpy(timesteps).long(),
            attention_mask=torch.from_numpy(attention_mask).long(),
        )
        return traj, thought, self.metadata[traj_idx]

    def __len__(self) -> int:
        return self._total_steps
