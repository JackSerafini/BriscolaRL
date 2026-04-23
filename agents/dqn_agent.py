import random
from collections import namedtuple, deque
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
EPS_DECAY = 0.99 # the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # update rate of the target network
LR = 3e-4 # the learning rate of the ``AdamW`` optimizer

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
    def __init__(self, env: Briscola, n_obs: int, device: torch.device,
                 lr: float = LR,
                 batch_size: int = BATCH_SIZE,
                 buffer_size: int = 100_000,
                 epsilon_start: float = EPS_START, epsilon_end: float = EPS_END, epsilon_decay: float = EPS_DECAY):
        self.env = env
        self.n_actions = self.env.action_space.n
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.eps_end = epsilon_end
        self.eps_decay = epsilon_decay

        self.device = device

        self.policy_net = QNet(n_obs, self.n_actions).to(self.device)
        self.target_net = QNet(n_obs, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # self.target_net.eval() # Target network is for evaluation only of course

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.buffer = ReplayMemory(buffer_size)

    def select_action(self, hand):
        """
        After all, hand is already by itself a mask of the possible actions (available cards)
        """
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        # EXPLOIT
        if random.random() > self.epsilon:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
            
                # .max(1) gets the maximum value by row
                # [0] is the actual value, [1] is the index of such value, i.e. our action.
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
            
                q_values = self.policy_net(state)  # shape: [1, 3]

                # Convert mask to tensor
                mask = torch.tensor(action_mask, device=self.device).unsqueeze(0)

                # Mask invalid actions
                q_values[mask == 0] = -1e9

                return q_values.max(1).indices.view(1, 1)
        # EXPLORE
        else:
            return torch.tensor([[self.env.action_space.sample(mask = hand)]], device=self.device, dtype=torch.long)
            
    def learn(self):
        """
        Perform a learning step:

        `if` experience buffer < batch size: just explore

        `else`: sample transitions from the buffer, calculate q-values (predicted and target), compute bellmann,
        calculate loss, perform a gradient descent step.
        """
        if len(self.buffer) < self.batch_size:
            return # Do not learn if the replay buffer is not at least == to the batch size: just explore

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))      # Transition( (s1, s2), (a1, a2), ... )

        # To tensor
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)

        # Predicted and target values:
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Compute the expected Q values with the usual formula
        # If done=True, the value is 0
        expected_q_values =  reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # Loss and gradient descent
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()