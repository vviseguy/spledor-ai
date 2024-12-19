from pydantic import BaseModel
from shared.model import Card, Noble, TokenType


import torch


from typing import Dict, List


class PlayerStateDto(BaseModel):
    player_id: str
    resources: Dict[TokenType, int]={tokenType:0 for tokenType in TokenType}
    reservedCards: List[str]=[]
    ownedCards: List[str]=[]
    noblesOwned: List[str]=[]
    score: int=0


class PlayerState(PlayerStateDto):
    reservedCards: List[Card]=[]
    ownedCards: List[Card]=[]
    noblesOwned: List[Noble]=[]

    @property
    def tensor(self):
        return torch.tensor(
            [
                self.score,
                self.noblesOwned,
                self.noblesOwned[TokenType.DIAMOND],
                self.noblesOwned[TokenType.SAPPHIRE],
                self.noblesOwned[TokenType.EMERALD],
                self.noblesOwned[TokenType.RUBY],
                self.noblesOwned[TokenType.ONYX],
                self.resources[TokenType.DIAMOND],
                self.resources[TokenType.SAPPHIRE],
                self.resources[TokenType.EMERALD],
                self.resources[TokenType.RUBY],
                self.resources[TokenType.ONYX],
                len(self.reservedCards),
            ]
        )
    @property
    def dto(self):
        reservedCards = [card.card_id for card in self.reservedCards]
        ownedCards = [card.card_id for card in self.ownedCards]
        noblesOwned = [noble.noble_id for noble in self.noblesOwned]
        return PlayerStateDto(self.player_id, self.resources, reservedCards, ownedCards, noblesOwned)