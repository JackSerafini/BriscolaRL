import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class RandomAgent:
    def __init__(self, action_dim=3):
        self.action_dim = action_dim

    def act(self, obs, action_mask):
        """
        obs: environment observation (not used here)
        action_mask: array like [1,1,0] for valid moves
        """
        valid_actions = np.where(action_mask == 1)[0]
        return np.random.choice(valid_actions)

    def learn(self, *args, **kwargs):
        """
        Random agent does not learn.
        This exists just for compatibility with training loops.
        """
        pass