import torch

from agents import RandomAgent
from briscola import Briscola

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def play_game(env, agent, n_games=100):
    wins = 0

    for _ in range(n_games):
        obs, _ = env.reset()
        done = False

        while not done:
            mask = env._get_action_mask()
            action = agent.act(obs, mask)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if env.player_score > env.opponent_score:
            wins += 1

    return wins / n_games


agent = RandomAgent()
env = Briscola()
win_rate = play_game(env, agent, 1000000)

print("Win rate:", win_rate)