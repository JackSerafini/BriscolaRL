import random
from collections import namedtuple, deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.999 # discount factor
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.01 # final value of epsilon
EPS_DECAY = 50_000 # the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # update rate of the target network
LR = 3e-4 # the learning rate of the ``AdamW`` optimizer

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

class DQN_Agent():
    def __init__(self, n_obs, n_actions, device,
                 lr = LR,
                 batch_size = BATCH_SIZE,
                 buffer_size = 100_000,
                 epsilon_start = EPS_START, epsilon_end = EPS_END, epsilon_decay = EPS_DECAY):
        # TODO: aggiungere env?
        self.n_actions = n_actions
        self.batch_size = batch_size

        self.device = device

        self.policy_net = QNet(n_obs, n_actions)
        self.target_net = QNet(n_obs, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.buffer = ReplayMemory(buffer_size)

    def select_action(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if random.random() > self.epsilon: # EXPLOIT
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else: # EXPLORE
            return torch.tensor([[env.action_space.sample()]], device=self.device, dtype=torch.long)

        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                # .max(1) gets the maximum value by row
                # [0] is the actual value, [1] is the index of such value, i.e. our action.
                return self.policy_net(state_tensor).max(1)[1].view(1, 1)
            
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