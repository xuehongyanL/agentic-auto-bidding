"""轨迹数据结构定义"""

from collections import namedtuple

Trajectory = namedtuple('Trajectory', ['states', 'actions', 'rtgs', 'timesteps', 'attention_mask'])
