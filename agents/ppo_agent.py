import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from briscola import Briscola

LR = 3e-4
GAMMA = 0.99
EPSILON_CLIP = 0.2
EPOCHS = 10
BATCH_SIZE = 64

# ── Hyperparameters ───────────────────────────────────────────
GAE_LAMBDA    = 0.95
VALUE_COEF    = 0.5
ENTROPY_COEF  = 0.01
MAX_GRAD_NORM = 0.5
ROLLOUT_STEPS = 1024
# ─────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one rollout, then discards after update.
    """
    def __init__(self):
        self.states = [] # Batch observations
        self.actions = [] # Batch actions
        self.log_probs = [] # Log-probabilities of each action
        self.rewards = [] # Batch rewards
        self.values = []
        self.dones = []

    def push(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
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
        return self.actor(x), self.critic(x)
        # TODO: understand what is the difference
        # return self.policy_head(x), self.value_head(x).squeeze(-1)
    
    
class PPO_Agent():
    def __init__(self, env: Briscola, device: torch.device,
                 savefile = None,
                 lr: float = LR,
                 gamma: float = GAMMA,
                 eps_clip = EPSILON_CLIP,
                 epochs = EPOCHS,
                 batch_size: int = BATCH_SIZE,
                 gae_lambda:    float = GAE_LAMBDA,
                 value_coef:    float = VALUE_COEF,
                 entropy_coef:  float = ENTROPY_COEF,
                 max_grad_norm: float = MAX_GRAD_NORM,
                 rollout_steps: int   = ROLLOUT_STEPS):
        self.env = env
        n_actions = env.action_space.n
        n_obs = gym.spaces.flatdim(env.observation_space)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.epochs = epochs
        self.batch_size = batch_size
        # self.gae_lambda    = gae_lambda
        # self.value_coef    = value_coef
        # self.entropy_coef  = entropy_coef
        # self.max_grad_norm = max_grad_norm
        # self.rollout_steps = rollout_steps

        self.device = device

        self.policy = ActorCritic(n_obs, n_actions).to(device)
        if savefile:
            self.policy.load_state_dict(savefile)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # buffer
        self.buffer = RolloutBuffer()
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []


    def select_action(self, state):
        state_tensor = state_to_tensor(state).to(self.device)

        logits, value = self.policy(state_tensor)

        # Mask invalid actions
        mask = torch.tensor(state["hand"], dtype=torch.bool, device=self.device).unsqueeze(0)
        logits[~mask] = -1e9

        probs = torch.softmax(logits, dim=1)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        # entropy = dist.entropy() # TODO: add the entropy?

        self.buffer.push(state_tensor, action, log_prob)
        self.states.append(state_tensor)
        self.actions.append(action)
        self.logprobs.append(log_prob)
        # TODO: add value?

        # # Stash these so the training loop can push them to the buffer
        # self._last_log_prob = log_prob.item()
        # self._last_value    = value.item()

        # return action.view(1, 1)

        return action.item()

    def compute_returns(self):
        returns = []
        G = 0

        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                G = 0
            G = r + self.gamma * G
            returns.insert(0, G)

        return torch.tensor(returns, dtype=torch.float32, device=self.device)
    
    def _compute_rewards_tg(self):
        rtgs = []
        last_rtg = 0

        # TODO: understand whether to use the buffer or not
        rewards = self.buffer.rewards
        dones = self.buffer.dones

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                last_rtg = 0
            last_rtg = reward + self.gamma * last_rtg
            rtgs.insert(0, last_rtg)

        return torch.tensor(rtgs, dtype=torch.float32, device=self.device)
    
    def _compute_gae(self, last_value: float):
        """Generalized Advantage Estimation over the current rollout."""
        T          = len(self.buffer)
        advantages = torch.zeros(T, device=self.device)
        last_gae   = 0.0
  
        rewards = self.buffer.rewards
        values  = self.buffer.values
        dones   = self.buffer.dones

        for t in reversed(range(T)):
            next_val  = last_value if t == T - 1 else values[t + 1]
            not_done  = 1.0 - dones[t]
            delta     = rewards[t] + self.gamma * next_val * not_done - values[t]
            last_gae  = delta + self.gamma * self.gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + torch.tensor(values, device=self.device)
        return advantages, returns
    
    def update(self):
        returns = self.compute_returns()

        states = torch.cat(self.states)
        actions = torch.stack(self.actions)
        old_logprobs = torch.stack(self.logprobs).detach()

        # EPOCHS = the number of times we reuse the SAME rollout data
        for _ in range(self.epochs):
            logits, state_values = self.policy(states)

            # Recompute mask
            masks = torch.cat([
                torch.tensor(s["hand"], dtype=torch.bool, device=self.device).unsqueeze(0)
                for s in self.states_raw  # you may want to store raw states too
            ])
            logits[~masks] = -1e9

            probs = torch.softmax(logits, dim=1)
            dist = torch.distributions.Categorical(probs)

            logprobs = dist.log_prob(actions.squeeze())
            entropy = dist.entropy()

            ratios = torch.exp(logprobs - old_logprobs)

            advantages = returns - state_values.squeeze().detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * (returns - state_values.squeeze())**2
                - 0.01 * entropy
            ).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # clear buffer
        self.states.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.dones.clear()


    def learn(self, last_value: float = 0.0):
        """
        Run K epochs of minibatch PPO updates on the collected rollout.
        Call this once per rollout (every `rollout_steps` steps), not every step.
        """
        T = len(self.buffer)
        if T == 0:
            return {}

        # Convert buffer to tensors
        states    = torch.cat(self.buffer.states).to(self.device)
        actions   = torch.tensor(self.buffer.actions,   dtype=torch.long,    device=self.device)
        old_lp    = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        masks     = torch.cat(self.buffer.masks).to(self.device)

        with torch.no_grad():
            advantages, returns = self._compute_gae(last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # K epochs of minibatch updates
        indices = np.arange(T)
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                new_lp, entropy, new_val = self.net.evaluate(
                    states[idx], masks[idx], actions[idx]
                )

                # Clipped surrogate loss
                ratio  = torch.exp(new_lp - old_lp[idx])
                surr1  = ratio * advantages[idx]
                surr2  = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                policy_loss  = -torch.min(surr1, surr2).mean()
                value_loss   =  0.5 * (new_val - returns[idx]).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":    -entropy_loss.item(),
        }
    
    def learn(self):
        states = torch.cat(self.buffer.states).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype = torch.long, device = self.device)
        old_logprobs = torch.tensor(self.buffer.log_probs, dtype = torch.float32, device = self.device)

        rewards_to_go = self._compute_rewards_tg()

        # N epochs of minibatch updates
        for _ in range(self.epochs):
            # TODO: shuffle indexes
            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                ratio = 1

        self.buffer.clear()