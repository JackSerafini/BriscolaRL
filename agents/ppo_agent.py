import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from briscola import Briscola

LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
EPSILON_CLIP = 0.2
EPOCHS = 10
BATCH_SIZE = 64
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
MAX_GRAD_NORM = 0.5

class RolloutBuffer:
    """
    Stores one rollout, then discards after update.
    """
    def __init__(self):
        self.states = [] # Batch observations
        self.actions = [] # Batch actions
        self.masks = [] # Batch masks
        self.log_probs = [] # Log-probabilities of each action
        self.rewards = [] # Batch rewards
        self.values = [] # Batch values
        self.dones = []

    def push(self, state, action, mask, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.masks.append(mask)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)

# We want to keep track of the state, action log probability of the action, reward and value of the critic of the action
# Then compute Reward to go: R_tg = R_t + gamma R_t+1 + gamma^2 R_t+2 + ...
# Then compute Advantage: A(s, a) = Q(s, a) - V(s), where Q is R_tg
# Advantage used to decrease % of taking bad moves and viceversa

# To optimize the critic net, we compute the loss (e.g., MSE): loss = 1/N sum_i (y_i - y^_i)^2,
# where y_i is the R_tg and y^_i is the value of the critic net
# At the end of training the estimated value V should approach the best possible action reward

# To optimize the actor net, use the PPO clip objective:
# L^CLIP(theta) = E_t[min[r_t(theta)A_t, clip(r_t(theta), 1 - eps, 1 + eps)A_t]],
# where r is the probability ratio (function of the parameters) = action prob_current / action prob_old, A is the advantage
# The expected value means taking all the trajectories and taking the average
# We want to maximize the objective -> gradient ascent for the actor, instead of gradient descent

# Repeat the optimization for n times and then repeat the whole process for T times

def state_to_tensor(state):
    return torch.cat([
        torch.tensor(state["hand"], dtype = torch.float32),
        torch.tensor(state["table_card"], dtype = torch.float32),
        torch.tensor(state["briscola"], dtype = torch.float32),
        torch.tensor(state["played_cards"], dtype = torch.float32),
        torch.tensor(state["is_first"], dtype = torch.float32),
    ]).unsqueeze(0)

class ActorCritic(nn.Module):
    def __init__(self, n_obs= 125, n_actions = 40):
        super(ActorCritic, self).__init__()

        # TODO: understand if tanh is better than relu
        self.shared = nn.Sequential(
            nn.Linear(n_obs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, n_actions) # Policy or Model
        self.critic = nn.Linear(128, 1) # The output is the value of the total expected return
        # -> at the end of training we expect this value to reach the best reward from that state
        # Basically, given the state s, we compute the estimation of the total expected return,
        # and each episode should improve this estimation until reach of the best value
        # (at the beginning of training the value will be random, as are the weights)
        # -> value used to improve the actor network

        # TODO: understand if this is useful or not
        # Orthogonal init — standard for PPO
        # for layer in self.trunk:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
        #         nn.init.zeros_(layer.bias)
        # nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        # nn.init.orthogonal_(self.value_head.weight,  gain=1.0)
        # nn.init.zeros_(self.policy_head.bias)
        # nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        x = self.shared(x)
        # TODO: understand what is the difference
        return self.actor(x), self.critic(x)
        # return self.policy_head(x), self.value_head(x).squeeze(-1)
    
    
class PPO_Agent():
    def __init__(self, env: Briscola, device: torch.device,
                 savefile = None,
                 lr: float = LR,
                 gamma: float = GAMMA,
                 gae_lambda: float = GAE_LAMBDA,
                 eps_clip = EPSILON_CLIP,
                 epochs = EPOCHS,
                 batch_size: int = BATCH_SIZE,
                 entropy_coef: float = ENTROPY_COEF,
                 value_coef: float = VALUE_COEF,
                 max_grad_norm: float = MAX_GRAD_NORM):
        self.env = env
        n_actions = env.action_space.n
        n_obs = gym.spaces.flatdim(env.observation_space)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.device = device

        self.policy_net = ActorCritic(n_obs, n_actions).to(device)
        if savefile:
            self.policy_net.load_state_dict(savefile)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, state):
        state_tensor = state_to_tensor(state).to(self.device)

        with torch.no_grad():
            logits, value = self.policy_net(state_tensor)

        # Mask invalid actions
        mask = torch.tensor(state["hand"], dtype=torch.bool, device=self.device).unsqueeze(0)
        logits = logits.masked_fill(~mask, float('-inf'))

        dist = torch.distributions.Categorical(logits = logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), mask, log_prob.item(), entropy, value.item()

    def _compute_GAE_rewardstg_and_advantages(self, last_value: float):
        """Generalized Advantage Estimation over the current rollout.  
        GAE: How much better was this action than expected, taking into account future corrections?"""
        length = len(self.buffer)
        advantages = torch.zeros(length, device = self.device)
        last_gae = 0.0

        rewards = self.buffer.rewards
        values = self.buffer.values
        dones = self.buffer.dones

        for t in reversed(range(length)):
            next_val = last_value if t == length - 1 else values[t + 1]
            not_done = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * not_done - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        # Advantage: A(s, a) = Q(s, a) - V(s), where Q is R_tg -> Q(s, a) = A(s, a) + V(s)
        rtgs = advantages + torch.tensor(values, device = self.device)
        return advantages, rtgs
    
    def learn(self, last_value: float = 0.0):
        """
        Run K epochs of minibatch PPO updates on the collected rollout.
        Call this once per rollout (every `rollout_steps` steps), not every step.
        """
        # YOU START BY TAKING THE EXPERIENCE FROM THE TRAINING, DOING ROLLOUT_STEPS AND PUSHING EACH STEP IN THE BUFFER
        steps = len(self.buffer)
        states = torch.cat(self.buffer.states).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype = torch.long, device = self.device)
        masks = torch.cat(self.buffer.masks).to(self.device)
        old_logprobs = torch.tensor(self.buffer.log_probs, dtype = torch.float32, device = self.device)

        with torch.no_grad():
            advantages, rewards_tg = self._compute_GAE_rewardstg_and_advantages(last_value)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        indices = np.arange(steps)
        # N epochs of minibatch updates
        # EPOCHS = the number of times we reuse the SAME rollout data
        for _ in range(self.epochs):
            # np.random.shuffle(indices)
            # for start in range(0, steps, self.batch_size):
                # idx = indices[start : start + self.batch_size]

                # logits, state_values = self.policy_net(states[idx])
            logits, state_values = self.policy_net(states)
                # mask = masks[idx]
                # logits = logits.masked_fill(~mask, float('-inf'))
            logits = logits.masked_fill(~masks, float('-inf'))

                # probs = torch.softmax(logits, dim=1)
                # dist = torch.distributions.Categorical(logits = logits)
            dist = torch.distributions.Categorical(logits = logits)
                # logprobs = dist.log_prob(actions[idx])
            logprobs = dist.log_prob(actions)
                # entropy = dist.entropy()
            entropy = dist.entropy()

                # Clipped surrogate loss
                # ratio = torch.exp(logprobs - old_logprobs[idx])
            ratio = torch.exp(logprobs - old_logprobs)

                # surr1 = ratio * advantages[idx]
            surr1 = ratio * advantages
                # surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # policy_loss = -torch.min(surr1, surr2).mean()
            policy_loss = -torch.min(surr1, surr2).mean()
                # value_loss = 0.5 * (state_values.squeeze() - rewards_tg[idx]).pow(2).mean()
            value_loss = 0.5 * (state_values.squeeze() - rewards_tg).pow(2).mean()
                # entropy_loss = -entropy.mean()
            entropy_loss = -entropy.mean()

                # loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # self.optimizer.zero_grad()
            self.optimizer.zero_grad()
                # loss.backward()
            loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                # self.optimizer.step()
            self.optimizer.step()

        self.buffer.clear()