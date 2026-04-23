import numpy as np

class RandomAgent():
    def act(self, obs):
        """
        obs: environment observation (not used here)
        action_mask: array like [1,1,0] for valid moves
        """
        valid_actions = np.where(obs['hand'] == 1)[0]
        return np.random.choice(valid_actions)

    def learn(self):
        """
        Random agent does not learn.
        This exists just for compatibility with training loops.
        """
        pass