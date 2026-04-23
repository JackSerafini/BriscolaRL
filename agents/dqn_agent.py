import random
from collections import namedtuple, deque
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from briscola import Briscola

BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.999 # discount factor
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.01 # final value of epsilon
# EPS_DECAY = 50_000 # the rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 0.995 # the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # update rate of the target network
LR = 3e-4 # the learning rate of the ``AdamW`` optimizer

# Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_mask'))

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
    
def state_to_tensor(state):
    return torch.cat([
        torch.tensor(state["hand"], dtype = torch.float32),
        torch.tensor(state["table_card"], dtype = torch.float32),
        torch.tensor(state["briscola"], dtype = torch.float32),
        torch.tensor(state["played_cards"], dtype = torch.float32),
        torch.tensor(state["is_first"], dtype = torch.float32),
    ]).unsqueeze(0)

class QNet(nn.Module):
    def __init__(self, n_obs: int, n_actions: int = 40):
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

class DQN_Agent():
    def __init__(self, env: Briscola, device: torch.device,
                 lr: float = LR,
                 batch_size: int = BATCH_SIZE,
                 buffer_size: int = 100_000,
                 epsilon_start: float = EPS_START, epsilon_end: float = EPS_END, epsilon_decay: float = EPS_DECAY,
                 gamma: float = GAMMA,
                 tau: float = TAU):
        self.env = env
        self.n_actions = self.env.action_space.n
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau

        self.device = device

        n_obs = gym.spaces.flatdim(self.env.observation_space)
        self.policy_net = QNet(n_obs, self.n_actions).to(self.device)
        self.target_net = QNet(n_obs, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is for evaluation only of course

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.buffer = ReplayMemory(buffer_size)

    def select_action(self, state):
        """
        After all, hand is already by itself a mask of the possible actions (available cards)
        """
        # self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        # EXPLOIT
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = state_to_tensor(state).to(self.device)
                q_values = self.policy_net(state_tensor)
                mask = torch.tensor(state["hand"], dtype = torch.bool, device = self.device).unsqueeze(0)
                q_values[mask == 0] = -1e9 # Mask invalid actions

                # .max(1) gets the maximum value by row
                # [0] is the actual value, [1] is the index of such value -> the action
                return q_values.max(1)[1].view(1, 1)
        # EXPLORE
        else:
            action = self.env.action_space.sample(mask = state["hand"])
            return torch.tensor([[action]], device=self.device, dtype=torch.long)
        
    def soft_update(self):
        for target, policy in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target.data.lerp_(policy.data, self.tau)
            
    def learn(self):
        """
        Perform a learning step:  
        `if` experience buffer < batch size: just explore  
        `else`: sample transitions from the buffer, calculate q-values (predicted and target), compute bellmann,
        calculate loss, perform a gradient descent step
        """
        if len(self.buffer) < self.batch_size:
            return # Do not learn if the replay buffer is not at least == to the batch size: just explore

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))      # Transition( (s1, s2), (a1, a2), ... )

        # Compute a mask of non-final states and concatenate the batch elements (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device, dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)
        non_final_next_masks = torch.cat([m for s, m in zip(batch.next_state, batch.next_mask) if s is not None]).to(self.device)

        # To tensor
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)

        # Predicted and target values:
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = torch.zeros(self.batch_size, device = self.device)
        with torch.no_grad():
            # next_actions = self.policy_net(non_final_next_states)
            # next_actions[non_final_next_masks == 0] = -1e9
            # next_actions = next_actions.max(1)[1].unsqueeze(1)

            # next_q = self.target_net(non_final_next_states)
            # next_q_values[non_final_mask] = next_q.gather(1, next_actions).squeeze(1)
            next_q = self.target_net(non_final_next_states)
            next_q[non_final_next_masks == 0] = -1e9
            next_q_values[non_final_mask] = next_q.max(1)[0]

        # Compute the expected Q values
        expected_q_values = reward_batch + (next_q_values * self.gamma)

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        self.soft_update()