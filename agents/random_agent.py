import numpy as np

class RandomAgent():
    def act(self, state):
        """
        state: environment observation (not used here)
        action_mask: array like [1,1,0] for valid moves
        """
        valid_actions = np.where(state['hand'] == 1)[0]
        return np.random.choice(valid_actions)

    def learn(self):
        """
        Random agent does not learn.
        This exists just for compatibility with training loops.
        """
        pass