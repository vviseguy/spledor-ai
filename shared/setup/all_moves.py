import itertools

from shared.model.Card import CardTier
from shared.model.Move import Move
from shared.model.TokenType import TokenType



tiers = list(CardTier)
gem_types = list(TokenType)[1:]
HARDCODED_MAX_RESERVED_CARDS = 3
HARDCODED_CARDS_PER_ROW = 4

MOVE_INDEX: list[Move] = [
    # SELECT GEMS
    *[Move.take_three_tokens(*combination) for combination in itertools.combinations(gem_types, 3)],
    *[Move.take_two_tokens(gem) for gem in gem_types],

    # RESERVE A CARD 
    *[Move.reserve_from_field(tier, card_i) for card_i in range(HARDCODED_CARDS_PER_ROW) for tier in tiers],
    *[Move.reserve_from_deck(tier) for tier in tiers],

    # PURCHASE A CARD 
    *[Move.buy_from_field(tier, card_i) for card_i in range(HARDCODED_CARDS_PER_ROW) for tier in tiers],
    *[Move.buy_from_hand(card_i) for card_i in range(HARDCODED_MAX_RESERVED_CARDS)],
]

print(len(MOVE_INDEX))
