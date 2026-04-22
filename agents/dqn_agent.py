import random
from collections import namedtuple, deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):
    def __init__(self, n_obs, n_actions = 40):
        """
        n_obs: it is the number of observations at each state (40 (hand) + 40 (table) + 4 (briscola suit) + 40 (played) + 1 (turn) = 125)
        n_actions: it is the card to play from the hand -> hand represented as 40-array of 0s and 1s
        """
        super(QNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent():
    def __init__(self, n_obs, n_actions, lr = 1e-3):
        self.n_actions = n_actions

        self.policy_net = QNet(n_obs, n_actions)
        self.target_net = QNet(n_obs, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
    
class DQN_Agent():
    def __init__(self, n_observations, n_actions, device,
                 lr = 3e-4, buffer_size = 100_000,
                 epsilon_start = 0.9, epsilon_end = 0.01, epsilon_decay = 50_000):
        self.device = device
        self.eps_start = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(buffer_size)

    def select_action(self, state, action_mask):
        global steps_done
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * steps_done / self.eps_decay)
        steps_done += 1
        
        # EXPLOIT (use model)
        if random.random() > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(state)  # shape: [1, 3]

                # Convert mask to tensor
                mask = torch.tensor(action_mask, device=self.device).unsqueeze(0)

                # Mask invalid actions
                q_values[mask == 0] = -1e9

                return q_values.max(1).indices.view(1, 1)

        # EXPLORE (random valid)
        else:
            valid_actions = np.where(action_mask == 1)[0]
            action = np.random.choice(valid_actions)

            return torch.tensor([[action]], device=self.device, dtype=torch.long)