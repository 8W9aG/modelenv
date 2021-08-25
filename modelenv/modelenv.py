"""The model mutation environment class."""
import typing
import enum
import math

import gym
from gym import spaces
import numpy as np

from .model_factory import create_model
from .constants import ACTION_PARAMETER_COUNT, LAYER_PARAMETER_COUNT


MAXIMUM_LAYER_COUNT = 100


class NASAction(enum.IntEnum):
    """Actions to perform on the model."""
    Hold = 0
    Add = 1
    Remove = 2


class ModelEnv(gym.Env):
    """The environment representing a model and its mutations."""
    metadata = {'render.modes': ['human']}

    def __init__(self, example_data: np.array, example_output: np.array):
        super(ModelEnv, self).__init__()
        self.action_space = spaces.Box(
            low=np.ones(ACTION_PARAMETER_COUNT + LAYER_PARAMETER_COUNT) * -1.0,
            high=np.ones(ACTION_PARAMETER_COUNT + LAYER_PARAMETER_COUNT),
        )
        self.observation_space = spaces.Box(
            low=np.ones(MAXIMUM_LAYER_COUNT * LAYER_PARAMETER_COUNT) * -1.0,
            high=np.ones(MAXIMUM_LAYER_COUNT * LAYER_PARAMETER_COUNT),
        )
        self.model = None
        self.example_data = example_data
        self.example_output = example_output
        self.reset()

    def step(self, action: np.array) -> typing.Tuple[np.array, float, bool, typing.Dict]:
        """Take an action in the environment and advance to the next state."""
        nas_action = int(math.floor(((action[0] + 1.0) / 2.0) * (NASAction.Remove + 1)))
        if nas_action == NASAction.Hold:
            pass
        elif nas_action == NASAction.Add:
            if self.model is None:
                self.model = create_model(action)
            else:
                self.model.add_layer(action)
        elif nas_action == NASAction.Remove:
            self.model.remove_layer()
        self.state = self.model.state()
        reward = self.model.train()
        return self.state, reward, False, {}

    def reset(self) -> np.array:
        """Reset the state of the environment."""
        self.state = np.ones(MAXIMUM_LAYER_COUNT * LAYER_PARAMETER_COUNT) * -1.0
        return self.state

    def render(self, mode = "human", close = False):
        """Render the current state."""
        if self.model is None:
            print("No Model")
            return
        self.model.print()
