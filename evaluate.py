import torch
import numpy as np
from collections import defaultdict
from briscola import Briscola
from agents.ppo_agent import BriscolaNet

# ── Config ───────────────────────────────────────────────────
MODEL_PATH   = "briscola_ppo.pt"
N_EPISODES   = 1000
HIDDEN       = 128
DETERMINISTIC = True   # False = sample from policy (stochastic eval)
# ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")

def flatten_obs(obs):
    return np.concatenate([
        obs["hand"].astype(np.float32),
        obs["table_card"].astype(np.float32),
        # obs["briscola_suit"].astype(np.float32),
        obs["briscola"].astype(np.float32),
        obs["played_cards"].astype(np.float32),
        obs["is_first"].astype(np.float32),
    ])

env = Briscola()
obs_dim   = len(flatten_obs(env.reset()[0]))
n_actions = env.action_space.n

net = BriscolaNet(obs_dim, n_actions, hidden=HIDDEN).to(device)
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.eval()

# ── Trackers ─────────────────────────────────────────────────
results = []          # "win" / "loss" / "tie"
score_diffs = []      # player_score - opponent_score per episode
reward_totals = []    # cumulative shaped reward per episode
tricks_won = []       # how many tricks the agent won per episode
points_breakdown = defaultdict(list)  # points scored in wins vs losses

def select_action(state_t, mask_t, deterministic=True):
    with torch.no_grad():
        logits, _ = net(state_t)
        logits = logits.masked_fill(~mask_t.bool(), float('-inf'))
        if deterministic:
            return logits.argmax(dim=-1).item()
        else:
            return torch.distributions.Categorical(logits=logits).sample().item()

# ── Evaluation loop ──────────────────────────────────────────
for ep in range(N_EPISODES):
    obs, info = env.reset()
    state       = flatten_obs(obs)
    action_mask = info["action_masks"]

    ep_reward  = 0.0
    ep_tricks  = 0
    done       = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t  = torch.tensor(action_mask, dtype=torch.bool, device=device).unsqueeze(0)

        action = select_action(state_t, mask_t, DETERMINISTIC)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        ep_reward += reward
        if reward > 0:
            ep_tricks += 1

        if not done:
            state       = flatten_obs(obs)
            action_mask = info["action_masks"]

    # Final scores are on the env after termination
    p_score = env.player_score
    o_score = env.opponent_score
    diff    = p_score - o_score

    if diff > 0:
        result = "win"
    elif diff < 0:
        result = "loss"
    else:
        result = "tie"

    results.append(result)
    score_diffs.append(diff)
    reward_totals.append(ep_reward)
    tricks_won.append(ep_tricks)
    points_breakdown[result].append(p_score)

# ── Report ───────────────────────────────────────────────────
wins   = results.count("win")
losses = results.count("loss")
ties   = results.count("tie")
total  = len(results)

print(f"\n{'='*50}")
print(f"  Evaluation over {total} episodes")
print(f"{'='*50}")
print(f"  Win rate:   {wins/total*100:5.1f}%  ({wins})")
print(f"  Loss rate:  {losses/total*100:5.1f}%  ({losses})")
print(f"  Tie rate:   {ties/total*100:5.1f}%  ({ties})")
print(f"{'─'*50}")
print(f"  Avg score diff  (agent - opponent): {np.mean(score_diffs):+.1f}")
print(f"  Avg agent score:                    {np.mean([env.player_score for _ in range(1)]):.1f}")
print(f"  Avg shaped reward per episode:      {np.mean(reward_totals):+.2f}")
print(f"{'─'*50}")
print(f"  Avg tricks won per episode:         {np.mean(tricks_won):.1f} / 10")
print(f"{'─'*50}")
for outcome in ("win", "loss", "tie"):
    pts = points_breakdown[outcome]
    if pts:
        print(f"  Avg agent points in {outcome}s:  {np.mean(pts):.1f}")
print(f"{'='*50}\n")