"""Dataset基类定义"""

from abc import abstractmethod

from torch.utils.data import Dataset

from agb_core.data.trajectory import Trajectory


class BaseDataset(Dataset):
    """Base dataset class for trajectory data."""

    def __init__(self, state_dim: int, action_dim: int):
        self._state_dim = state_dim
        self._action_dim = action_dim

    @abstractmethod
    def __getitem__(self, index: int) -> tuple[Trajectory, str]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
