import random
from shared.model.TokenType import TokenType


import torch
from pydantic import BaseModel, model_validator


from typing import Dict, List


class Noble(BaseModel):
    noble_id: str
    requirements: Dict[TokenType, int]
    points: int

    
    # make sure the cost has an item for each TokenType
    @model_validator(mode="after")
    def ensure_all_token_types(cls, model):
        # Initialize missing TokenTypes to 0
        for token in TokenType:
            if token not in model.requirements:
                model.requirements[token] = 0
        return model

    @property
    def array(self):
        return [
            self.points,
            self.requirements[TokenType.DIAMOND],
            self.requirements[TokenType.SAPPHIRE],
            self.requirements[TokenType.EMERALD],
            self.requirements[TokenType.RUBY],
            self.requirements[TokenType.ONYX],
        ]


class NobleDeck(BaseModel):
    nobles: List[Noble]

    def pop(self):
        return self.nobles.pop()

    def len(self):
        return len(self.nobles)

    def shuffle(self):
        random.shuffle(self.nobles)
