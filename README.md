# BriscolaRL

Reinforcement Learning agents trained to play **Briscola**, a classic Italian trick-taking card game.
The project implements and compares two approaches, **DQN** (Deep Q-Network) and **PPO**
(Proximal Policy Optimization), built from scratch in PyTorch against a random opponent.

## Game

Briscola is a 2-player card game played with a 40-card Italian deck. Each player holds 3 cards,
takes turns playing one, and the trick winner is determined by suit and rank strength. A trump suit
(*briscola*) is revealed at the start and beats all other suits. The player with more than 60 points
at the end wins.

Key challenges for RL:
- **Imperfect information:** opponent's hand is hidden
- **Delayed reward:** the winning condition is only known at the end of the game
- **Stochastic transitions:** opponent plays randomly, deck is shuffled

## Project Structure

```
BriscolaRL/
├── briscola.py        # Gymnasium environment
├── agents/
│   ├── dqn_agent.py   # DQN agent (Double DQN, replay buffer, action masking)
│   └── ppo_agent.py   # PPO agent (GAE, clipped objective, rollout buffer)
├── models/            # Saved model weights (.pt)
├── outputs/           # Training logs and plots
├── train.ipynb        # Training notebook
├── evaluate.ipynb     # Evaluation notebook (win rate, score breakdown)
└── plots.ipynb        # Evaluation notebook (win rate, score breakdown)
```

## Agents

### DQN
- Experience replay buffer (100k transitions)
- Soft target network updates (τ = 0.005)
- Action masking via hand encoding (invalid cards → Q = −∞)
- ε-greedy exploration with multiplicative decay

### PPO
- Actor-Critic architecture
- Generalized Advantage Estimation (GAE, λ = 0.95)
- Clipped surrogate objective (ε = 0.2)
- Action masking via `masked_fill` before `Categorical(logits=...)`
- Rollout buffer cleared after each update

## Installation

```bash
git clone https://github.com/JackSerafini/BriscolaRL.git
cd BriscolaRL
python3.14 -m venv venv
source venv/bin/activate
pip install torch gymnasium numpy
```

## Usage

**Train:** open and run `train.ipynb`, or adapt the training loop from the notebook.

**Evaluate:** open and run `evaluate.ipynb`, or adapt the evaluation loop from the notebook.

## Results

| Agent   | Random |   DQN | PPO (500K) | PPO (1000K) | DQN (Auto-500K) | PPO (Auto-1000K) |
| ------- | -----: | ----: | ---------: | ----------: | --------------: | ---------------: |
| **DQN** |  89.1% |  ~50% |      58.7% |       53.6% |           66.8% |            50.8% |
| **PPO** |  87.5% | 44.7% |      53.8% |        ~50% |           64.6% |            47.5% |

*Evaluated over 10,000 episodes.*
