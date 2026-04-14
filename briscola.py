import random
import gymnasium as gym

SUITS = [0, 1, 2, 3] # 0: "Danari", 1: "Coppe", 2: "Spade", 3: "Bastoni"
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 1: "Asso", 8: "Fante", 9: "Cavallo", 10: "Re"

POINTS = {
    1: 11,
    3: 10,
    8: 2,
    9: 3,
    10: 4,
}

class Briscola(gym.Env):
    def __init__(self):
        super().__init__()

        self.action_space = gym.spaces.Discrete(3)  # play 1 of 3 cards

        self.observation_space = gym.spaces.Dict({
            "hand": gym.spaces.MultiBinary(40),
            "table_card": gym.spaces.MultiBinary(40),
            "briscola": gym.spaces.MultiBinary(4),
            "played_cards": gym.spaces.MultiBinary(40),
        })

        self.deck = []
    
    def _create_deck(self):
        self.deck = []
        for suit in SUITS:
            for rank in RANKS:
                self.deck.append((suit, rank))
        return self.deck

    def _shuffle(self):
        random.shuffle(self.deck)

    def _draw(self, number):
        # TODO: check the len of the deck?
        cards = []
        for i in range(number):
            cards.append(self.deck.pop())
        return cards
    
    def _get_obs(self):
        # TODO
        pass

    def _opponent_policy(self):
        return random.choice(self.opponent_hand)

    def _evaluate_trick(self, player_card, opponent_card):
        # TODO: change player and opponent to first and second (as in order of play)
        # TODO: fix value of cards with strenght
        if player_card[0] == self.briscola_suit and opponent_card[0] == self.briscola_suit: # if both are briscola, take higher card
            if player_card[1] > opponent_card[1]:
                winner = "player"
            else:
                winner = "opponent"
        elif player_card[0] == self.briscola_suit and opponent_card[0] != self.briscola_suit: # if one is briscola and the other is not, briscola wins
            winner = "player"
        elif player_card[0] != self.briscola_suit and opponent_card[0] == self.briscola_suit: 
            winner = "opponent"
        elif player_card[0] == opponent_card[0]: # if the second player matches the suit, the higher card wins
            if player_card[1] > opponent_card[1]:
                winner = "player"
            else:
                winner = "opponent"
        elif player_card[0] != opponent_card[0]: # otherwise wins the first player
            winner = "player"

        points = 0
        if player_card[1] in POINTS:
            points += POINTS[player_card[1]]
        if opponent_card[1] in POINTS:
            points += POINTS[opponent_card[1]]

        return winner, points
    
    def _draw_phase(self, winner):
        # TODO: handle the last turn of drawing
        if len(self.deck) > 0:
            if winner == "player":
                self.player_hand.append(self.deck.pop())
                self.opponent_hand.append(self.deck.pop())
            else:
                self.opponent_hand.append(self.deck.pop())
                self.player_hand.append(self.deck.pop())
        

    def reset(self, seed = None):
        super().reset(seed=seed)
        self.deck = self._create_deck()
        self._shuffle()

        self.player_hand = self._draw(3)
        self.opponent_hand = self._draw(3)

        self.briscola_card = self._draw(1)[0]
        self.briscola_suit = self.briscola_card[0]

        self.played_cards = []
        self.table = []

        self.terminated = False
        self.truncated = False

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        assert not self.terminated
        assert not self.truncated

        # TODO: MAKE THE PLAY DEPEND ON THE WINNER
        # 1. Player plays a card
        player_card = self.player_hand.pop(action)

        # 2. Opponent plays (simple policy for now)
        opponent_card = self._opponent_policy() # TODO: choose whether to pop the card in the policy or right after
        # self.opponent_hand.remove(opponent_card)

        # Determine trick winner
        winner, points = self._evaluate_trick(player_card, opponent_card)

        reward = points if winner == "player" else -points

        # Draw new cards
        self._draw_phase(winner)

        # Check end of game
        if len(self.player_hand) == 0 and len(self.deck) == 0:
            self.terminated = True

        obs = self._get_obs()
        return obs, reward, self.terminated, self.truncated, {}