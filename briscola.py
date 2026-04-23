import random
import numpy as np
import gymnasium as gym

SUITS = [0, 1, 2, 3] # 0: "Danari", 1: "Coppe", 2: "Spade", 3: "Bastoni"
RANKS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # 1: "Asso", 8: "Fante", 9: "Cavallo", 10: "Re"

POINTS = {1: 11, 3: 10, 8: 2, 9: 3, 10: 4}

STRENGTH = {1: 10, 3: 9, 10: 8, 9: 7, 8: 6, 7: 5, 6: 4, 5: 3, 4: 2, 2: 1}

PLAYER = 0
OPPONENT = 1

class Briscola(gym.Env):
    def __init__(self):
        super().__init__()

        # self.action_space = gym.spaces.Discrete(3)  # play 1 of 3 cards
        self.action_space = gym.spaces.Discrete(40)

        # TODO: add the points that the agest has
        self.observation_space = gym.spaces.Dict({
            "hand": gym.spaces.MultiBinary(40),
            "table_card": gym.spaces.MultiBinary(40),
            "briscola": gym.spaces.MultiBinary(4),
            "played_cards": gym.spaces.MultiBinary(40),
            "is_first": gym.spaces.MultiBinary(1),
        })

        self.deck = []
        self.player_score = 0
        self.current_player = None
        self.table = []
        self.played_cards = []
        self.briscola_card = None
        self.briscola_suit = None
    
    def _create_deck(self):
        return [(suit, rank) for suit in SUITS for rank in RANKS]

    def _shuffle(self):
        random.shuffle(self.deck)

    def _draw(self, n):
        return [self.deck.pop() for _ in range(n)]
    
    def _reveal_briscola(self):
        self.briscola_card = self.deck[0]
        self.briscola_suit = self.briscola_card[0]

    def _opponent_policy(self):
        return random.choice(self.opponent_hand)

    def _evaluate_trick(self, first_card, second_card):
        suit1, rank1 = first_card
        suit2, rank2 = second_card

        strength1 = STRENGTH[rank1]
        strength2 = STRENGTH[rank2]

        if suit1 == suit2:
            winner = "first" if strength1 > strength2 else "second"
        # elif suit1 == self.briscola_suit: # TODO: understand if it's optional
        #     winner = "first"
        elif suit2 == self.briscola_suit:
            winner = "second"
        else:
            winner = "first"

        points = POINTS.get(rank1, 0) + POINTS.get(rank2, 0)

        return winner, points
    
    def _draw_phase(self, winner):
        if len(self.deck) > 0:
            if winner == "player":
                self.player_hand.append(self.deck.pop())
                self.opponent_hand.append(self.deck.pop())
            else:
                self.opponent_hand.append(self.deck.pop())
                self.player_hand.append(self.deck.pop())

    def _encode_cards(self, cards):
        vec = np.zeros(40, dtype=np.int8)
        for suit, rank in cards:
            idx = suit * 10 + (rank - 1)
            vec[idx] = 1
        return vec
    
    def _encode_suit(self, suit):
        vec = np.zeros(4, dtype=np.int8)
        vec[suit] = 1
        return vec
    
    # def _get_action_mask(self):
    #     mask = np.zeros(40, dtype=np.int8)
    #     for suit, rank in self.player_hand:
    #         idx = suit * 10 + (rank - 1)
    #         mask[idx] = 1

    #     # terminal state: hand is empty, make mask valid (arbitrary — never sampled)
    #     if mask.sum() == 0:
    #         mask[0] = 1
    #     return mask
    
    def _get_obs(self):
        # Encode player's hand
        hand = self._encode_cards(self.player_hand)

        # Encode table (max 1 card visible to player)
        table_card = self._encode_cards(self.table)

        # Encode briscola suit
        briscola = self._encode_suit(self.briscola_suit)

        # Encode played cards
        played_cards = self._encode_cards(self.played_cards)

        # Encode the order of play (1 is first, 0 is second)
        is_first = np.array([1 if len(self.table) == 0 else 0], dtype=np.int8)

        return {
            "hand": hand,
            "table_card": table_card,
            "briscola": briscola,
            "played_cards": played_cards,
            "is_first": is_first,
        }
        

    def reset(self, seed = None):
        super().reset(seed=seed)
        # Create the deck, shuffle it, and reset the score
        self.deck = self._create_deck()
        self._shuffle()
        self.player_score = 0
        # Reset both the table and the played cards
        self.table = []
        self.played_cards = []
        self.terminated = False
        self.truncated = False

        # Choose the first player
        self.current_player = random.choice(["player", "opponent"])

        # Deal the cards based on the order
        if self.current_player == "player":
            self.player_hand = self._draw(3)
            self.opponent_hand = self._draw(3)
        else:
            self.opponent_hand = self._draw(3)
            self.player_hand = self._draw(3)

        # Draw the Briscola
        self._reveal_briscola()

        # If the opponent is first, play its turn
        if self.current_player == "opponent":
            first_card = self._opponent_policy()
            self.opponent_hand.remove(first_card)
            self.table.append(first_card)

        # Get the observations of the initial state
        obs = self._get_obs()
        # info = {"action_masks": self._get_action_mask()}
        return obs, {}
    

    def step(self, action):
        assert not self.terminated
        # assert not self.truncated

        card = (action // 10, (action % 10) + 1)
        if card not in self.player_hand:
            raise ValueError("Invalid action")

        self.player_hand.remove(card)

        # The player is first
        if self.current_player == "player":
            first_card = card
            self.table.append(first_card)

            second_card = self._opponent_policy()
            self.opponent_hand.remove(second_card)
            self.table.append(second_card)

            first = "player"
            second = "opponent"
        # The player is second and the opponent has played
        else:
            first_card = self.table[0]

            second_card = card
            self.table.append(second_card)

            first = "opponent"
            second = "player"

        self.played_cards.extend(self.table)

        trick_winner, points = self._evaluate_trick(first_card, second_card)

        # Map winner to actual player
        winner = first if trick_winner == "first" else second

        # Assign reward based on win or loss
        reward = points if winner == "player" else -points
        if winner == "player":
            self.player_score += points

        # Update the current_player for the next turn
        self.current_player = winner
        # Draw phase
        self._draw_phase(winner)

        # Clear table
        self.table = []

        # Check if both the cards in the deck and in hand are finished
        if len(self.player_hand) == 0 and len(self.deck) == 0:
            self.terminated = True

            # TODO: choose best reward for winning the game
            sign = np.sign(self.player_score - 60)
            reward += 200 * sign

            obs = self._get_obs()
            # info = {"action_masks": self._get_action_mask()}
            return obs, reward, self.terminated, self.truncated, {}

        # If opponent starts next → play immediately
        if self.current_player == "opponent":
            first_card = self._opponent_policy()
            self.opponent_hand.remove(first_card)
            self.table.append(first_card)

        obs = self._get_obs()
        # info = {"action_masks": self._get_action_mask()}
        return obs, reward, self.terminated, self.truncated, {}