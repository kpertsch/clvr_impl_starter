import numpy as np


class Reward:
    """A simple wrapper class for reward computation."""
    def __call__(self, trajectories, shapes):
        raise NotImplementedError

    @property
    def name(self):
        return self.NAME


class ZeroReward(Reward):
    """Returns zero reward for every shape and step in trajectory."""
    NAME = 'zero'

    def __call__(self, trajectories, shapes):
        return np.zeros((trajectories.shape[0],), dtype=np.float32)


class VertPosReward(Reward):
    """Returns reward proportional to the vertical position of the first object."""
    NAME = 'vertical_position'

    def __call__(self, trajectories, shapes):
        return trajectories[:, 0, 1]


class HorPosReward(Reward):
    """Returns reward proportional to the horizontal position of the first object."""
    NAME = 'horizontal_position'

    def __call__(self, trajectories, shapes):
        return trajectories[:, 0, 0]


class AgentXReward(Reward):
    """Returns reward proportional to the horizontal position of the agent. Assumes that agent is the first object."""
    NAME = 'agent_x'

    def __call__(self, trajectories, shapes):
        return trajectories[:, 0, 1]


class AgentYReward(Reward):
    """Returns reward proportional to the vertical position of the agent. Assumes that agent is the first object."""
    NAME = 'agent_y'

    def __call__(self, trajectories, shapes):
        return trajectories[:, 0, 0]


class TargetXReward(Reward):
    """Returns reward proportional to the horizontal position of the target. Assumes that target is second object."""
    NAME = 'target_x'

    def __call__(self, trajectories, shapes):
        return trajectories[:, 1, 1]


class TargetYReward(Reward):
    """Returns reward proportional to the vertical position of the target. Assumes that target is the second object."""
    NAME = 'target_y'

    def __call__(self, trajectories, shapes):
        return trajectories[:, 1, 0]

