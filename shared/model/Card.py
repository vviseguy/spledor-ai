from enum import StrEnum
import random
from shared.model.TokenType import TokenType


import torch
from pydantic import BaseModel, model_validator


from typing import Dict, List


# Enums for predefined types
class CardTier(StrEnum):
    TIER1 = "I"
    TIER2 = "II"
    TIER3 = "III"


class Card(BaseModel):
    card_id: str
    tier: CardTier
    cost: Dict[TokenType, int]
    points: int
    bonus: TokenType

    # make sure the cost has an item for each TokenType
    @model_validator(mode="after")
    def ensure_all_token_types(cls, model):
        # Initialize missing TokenTypes to 0
        for token in TokenType:
            if token not in model.cost:
                model.cost[token] = 0
        return model

    @property
    def array(self):
        print(self.cost)
        return [
            self.points,
            self.cost[TokenType.GOLD],
            self.cost[TokenType.DIAMOND],
            self.cost[TokenType.SAPPHIRE],
            self.cost[TokenType.EMERALD],
            self.cost[TokenType.RUBY],
            self.cost[TokenType.ONYX],
            list(TokenType).index(self.bonus),
        ]


class Deck(BaseModel):
    tiers: Dict[CardTier, List[Card]]

    def pop(self, tier: CardTier):
        return self.tiers[tier].pop()

    def len(self, tier: CardTier):
        return len(self.tiers[tier])

    def shuffle(self):
        for tier in self.tiers:
            random.shuffle(tier)
