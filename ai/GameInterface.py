from typing import List, Any, Tuple

import torch

from shared.model.GameState import GameStatus
from shared.model.BoardState import BoardState
from shared.model.PlayerState import PlayerState
from shared.setup.all_moves import MOVE_INDEX
from shared.gameplay.Game import Game
from shared.setup.GameFactory import OnePlayerGameFactory

class GameInterface:
    game:Game

    """An interface for the Splendor single-player game environment."""
    def __init__(self):
        self.reset()
    
    def reset(self) -> None:
        """
        Resets the game to an initial starting state.
        """
        self.game = OnePlayerGameFactory().make_game("ai_game46546")
        self.game.add_player("ai_bot")
    
    def current_state(self) -> Any:
        """
        Returns a representation of the current game state suitable for the neural network.
        This could be a numpy array, a PyTorch tensor, or something else that can easily be
        converted to the model’s input shape.
        """
        state:BoardState = self.game.state.board_state
        me:PlayerState = self.game.state.players[0]

        # cards per tier x num tiers x num tokens + 2
        cards = state.cards_tensor.view(1, -1) 
        print(cards.shape)
        
        #  num tokens
        tokens = state.tokens_tensor.view(1, -1)
        print(tokens.shape)
        
        # nobles in play x num tokens + 1
        nobles = state.nobles_tensor.view(1, -1)
        print(nobles.shape)
        
        # cards per tier x num tiers x num tokens + 2
        me = me.tensor.view(1, -1)
        print(me.shape)
        return torch.cat([cards, tokens, nobles, me], dim=1)

    def legal_actions(self) -> List[int]:
        """
        Returns a list of legal actions (integer indices) for the current state.
        Action space indices might range over all possible moves, but only a subset
        are legal at this step.
        """
        return [i for i in range(len(MOVE_INDEX)) if self.game.test_move(MOVE_INDEX[i])]
    
    def step(self, action_index: int) -> None:
        """
        Takes the given action (which must be legal), and updates the game state.
        """
        move = MOVE_INDEX[action_index] ## do converting magic
        self.game.apply_move(move)

    def is_terminal(self) -> bool:
        """
        Returns True if the game is over (no further moves).
        """
        return self.game.state == GameStatus.COMPLETED

    def final_score(self) -> float:
        """
        Returns the final score/outcome of the game once it is terminal.
        This should be a scalar, possibly normalized to a certain range.
        """
        me:PlayerState = self.game.state.players[0]
        return me.score

    def get_state_representation(self) -> Any:
        """
        Returns a representation of the state as used for storage in the replay buffer.
        Sometimes the state representation for NN input and for storing might be the same.
        If they are the same, this can just call current_state().
        """
        return self.current_state()
