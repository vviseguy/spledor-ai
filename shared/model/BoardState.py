from pydantic import BaseModel
from shared.model.Card import Deck, Card, CardTier
from shared.model.Noble import Noble, NobleDeck
from shared.model.TokenType import TokenType


import torch


from typing import Dict, List


class BoardStateDto(BaseModel):
    availableCards: Dict[CardTier, list[str]]
    nobles: List[str]
    tokenBank: Dict[TokenType, int]


class BoardState(BoardStateDto):
    availableCards: Dict[CardTier, list[Card]]
    nobles: List[Noble]
    deck: Deck
    noble_deck: NobleDeck

    @property
    def cards_tensor(self):
        return torch.tensor([[card.array for card in tier] for teir_name, tier in self.availableCards.items()])
    @property
    def nobles_tensor(self):
        return torch.tensor([noble.array for noble in self.nobles] )
    @property
    def tokens_tensor(self):
        return torch.tensor([self.tokenBank[tokenType] for tokenType in list(TokenType)])

    @property
    def dto(self):
        availableCards = {key:[card.card_id for card in values] for key, values in self.availableCards.items()}
        nobles = [tile.noble_id for tile in self.nobles.nobles]
        return BoardStateDto(availableCards, nobles, self.tokenBank)
