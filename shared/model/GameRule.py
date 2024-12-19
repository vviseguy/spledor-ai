from shared.model.Move import Move
from shared.model.GameState import GameState


from pydantic import BaseModel


from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

def default_callback(className):
    raise ValueError(f"Game rule triggered: {className}")

class GameRule(BaseModel):
    callback: Optional[Callable[[], None]] = lambda: default_callback("-")

    @abstractmethod
    def check(self, game_state: GameState, previous_move: Optional["Move"] = None) -> None:
        """
        Check if the rule is violated given a GameState and optionally the previous move.
        Call the callback if the rule is triggered.
        """
        pass


    @classmethod
    def get_subclass_by_name(cls, class_name):
        # Check direct subclasses
        for subclass in cls.__subclasses__():
            if subclass.__name__ == class_name:
                return subclass
            # Recursively check subclasses of this subclass
            nested_subclass = subclass.get_subclass_by_name(class_name)
            if nested_subclass:
                return nested_subclass
        return None