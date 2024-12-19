from enum import StrEnum
from pydantic import BaseModel


from typing import Dict, Optional, Union

from shared.model import CardTier, TokenType


class MoveType(StrEnum):
    BUY = "buy_card"
    RESERVE = "reserve_card"
    TAKE_TOKENS = "take_tokens"


class MoveDtoDetailsBuyCard(BaseModel):
    cardId: str
    # payment: Dict[str, int]
    # nobleChoice: Optional[str]


class MoveDtoDetailsTakeTokens(BaseModel):
    tokens: Dict[str, int]
    # selectedTokens: Optional[Dict[TokenType, int]]


class MoveDtoDetailsReserveCard(BaseModel):
    cardId: str
    # tier: Optional[str]


class MoveDto(BaseModel):
    action: MoveType
    details: Union[MoveDtoDetailsBuyCard, MoveDtoDetailsReserveCard, MoveDtoDetailsTakeTokens]
    playerIndex: Optional[int] = None


class Move(BaseModel):
    type: MoveType
    card_tier: Optional[CardTier]=None
    card_index: Optional[int]=None
    tokens: Optional[Dict[str, int]]=None
    playerIndex: Optional[int]=None

    @staticmethod
    def take_two_tokens(type: TokenType):
        return Move(type=MoveType.TAKE_TOKENS, tokens={type: 2})

    @staticmethod
    def take_three_tokens(type1: TokenType, type2: TokenType, type3: TokenType):
        return Move(type=MoveType.TAKE_TOKENS, tokens={type1: 1, type2: 1, type3: 2})

    @staticmethod
    def reserve_from_deck(tier: CardTier):
        return Move(type=MoveType.RESERVE, tokens={TokenType.GOLD: 1}, card_tier=tier)

    @staticmethod
    def reserve_from_field(tier: CardTier, card_index: int):
        return Move(type=MoveType.RESERVE, tokens={TokenType.GOLD: 1}, card_tier=tier, card_index=card_index)

    @staticmethod
    def buy_from_field(tier: CardTier, card_index: int):
        return Move(type=MoveType.BUY, card_tier=tier, card_index=card_index)

    @staticmethod
    def buy_from_hand(card_index: int):
        return Move(type=MoveType.BUY, card_index=card_index)

    @property
    def dto(self):
        raise NotImplementedError
        # reservedCards = [card.card_id for card in self.reservedCards]
        # ownedCards = [card.card_id for card in self.ownedCards]
        # noblesOwned = [noble.noble_id for noble in self.noblesOwned]
        # return MoveDto(self.action, self.resources, reservedCards, ownedCards, noblesOwned)
