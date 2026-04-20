from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch
import torch.nn as nn

class BriscolaNet(nn.Module):
    """
    Shared trunk → separate policy head and value head.
    Input: flattened obs vector.
    Output: action logits (40,) and state value (1,).
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head  = nn.Linear(hidden, 1)

        # Orthogonal init — standard practice for PPO
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)  # small init for policy
        nn.init.orthogonal_(self.value_head.weight,  gain=1.0)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

    def get_action(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Sample an action, masking invalid ones.
        Returns: action, log_prob, entropy, value
        """
        logits, value = self.forward(x)

        # Safety: if a row is all-invalid (terminal state), unmask everything
        # so Categorical doesn't receive all-inf logits
        all_masked = ~mask.any(dim=-1, keepdim=True)  # shape [B, 1]
        safe_mask = mask | all_masked.expand_as(mask)

        # Apply mask: set invalid logits to -inf before softmax
        # logits = logits.masked_fill(~mask.bool(), float('-inf'))
        logits = logits.masked_fill(~safe_mask, float('-inf'))

        dist = torch.distributions.Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        # Entropy only over valid actions (finite logits)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate(self, x: torch.Tensor, mask: torch.Tensor, action: torch.Tensor):
        """
        Re-evaluate stored actions under the current policy.
        Used during the update phase.
        """
        logits, value = self.forward(x)

        # Safety: if a row is all-invalid (terminal state), unmask everything
        # so Categorical doesn't receive all-inf logits
        all_masked = ~mask.any(dim=-1, keepdim=True)  # shape [B, 1]
        safe_mask = mask | all_masked.expand_as(mask)

        # logits = logits.masked_fill(~mask.bool(), float('-inf'))
        logits = logits.masked_fill(~safe_mask, float('-inf'))

        dist     = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        return log_prob, entropy, value
    


@dataclass
class RolloutBuffer:
    """Stores one rollout, then discards after update."""
    states:    List[np.ndarray] = field(default_factory=list)
    actions:   List[int]        = field(default_factory=list)
    log_probs: List[float]      = field(default_factory=list)
    rewards:   List[float]      = field(default_factory=list)
    values:    List[float]      = field(default_factory=list)
    masks:     List[np.ndarray] = field(default_factory=list)
    dones:     List[bool]       = field(default_factory=list)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class PPO:
    def __init__(
        self,
        net,
        lr:           float = 3e-4,
        gamma:        float = 0.99,
        gae_lambda:   float = 0.95,
        clip_eps:     float = 0.2,
        value_coef:   float = 0.5,
        entropy_coef: float = 0.01,
        n_epochs:     int   = 10,
        batch_size:   int   = 64,
        max_grad_norm:float = 0.5,
        device:       str   = "cpu",
    ):
        self.net          = net
        self.optimizer    = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.max_grad_norm = max_grad_norm
        self.device       = device

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values:  torch.Tensor,
        dones:   torch.Tensor,
        last_value: float,
    ):
        """
        Generalized Advantage Estimation.
        Produces advantages and discounted returns.
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_val  = last_value if t == T - 1 else values[t + 1].item()
            next_done = dones[t]                    # 1.0 if episode ended at t
            delta     = rewards[t] + self.gamma * next_val * (1.0 - next_done) - values[t]
            last_gae  = delta + self.gamma * self.gae_lambda * (1.0 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, buffer: RolloutBuffer, last_value: float):
        """Run K epochs of minibatch PPO updates on the collected rollout."""
        T = len(buffer)

        # Convert buffer to tensors
        states    = torch.tensor(np.array(buffer.states),    dtype=torch.float32, device=self.device)
        actions   = torch.tensor(buffer.actions,             dtype=torch.long,    device=self.device)
        old_lp    = torch.tensor(buffer.log_probs,           dtype=torch.float32, device=self.device)
        rewards   = torch.tensor(buffer.rewards,             dtype=torch.float32, device=self.device)
        values    = torch.tensor(buffer.values,              dtype=torch.float32, device=self.device)
        masks     = torch.tensor(np.array(buffer.masks),     dtype=torch.bool,    device=self.device)
        dones     = torch.tensor(buffer.dones,               dtype=torch.float32, device=self.device)

        with torch.no_grad():
            advantages, returns = self.compute_gae(rewards, values, dones, last_value)

        # Normalize advantages — critical for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- K epochs of minibatch updates ---
        indices = np.arange(T)
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                idx = indices[start : start + self.batch_size]

                new_lp, entropy, new_val = self.net.evaluate(
                    states[idx], masks[idx], actions[idx]
                )

                # PPO clipped surrogate loss
                ratio        = torch.exp(new_lp - old_lp[idx])
                surr1        = ratio * advantages[idx]
                surr2        = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[idx]
                policy_loss  = -torch.min(surr1, surr2).mean()

                # Value loss (clipped, same idea as policy clip)
                value_loss   = 0.5 * (new_val - returns[idx]).pow(2).mean()

                # Entropy bonus — encourages exploration
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss":  value_loss.item(),
            "entropy":     -entropy_loss.item(),
        }