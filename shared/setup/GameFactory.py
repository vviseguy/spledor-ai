from abc import ABC, abstractmethod
import json
from typing import Any
from uuid import uuid4
from shared.gameplay.Game import Game
from shared.model.Card import Card, Deck
from shared.model.GameRules import *
from shared.model.Noble import Noble, NobleDeck
from shared.model.Ruleset import Ruleset, RulesetDto


class GameFactory(ABC):

    card_deck: Deck
    noble_deck: NobleDeck
    rules: Dict[str, Any]
    
    def _load_deck_from_file(self, fileName):
        with open(fileName, "r") as json_str:
            # Parse JSON string into Python list of dicts
            data = json.load(json_str)

            # Create list of Card instances
            card_list = [Card(**card) for card in data]
            # Sort into tiers
            tiers = set(card.tier for card in card_list)
            self.card_deck = Deck(tiers={tier:[card for card in card_list if card.tier == tier] for tier in tiers})

    def _load_noble_deck_from_file(self, fileName):
        with open(fileName, "r") as json_str:
            # Parse JSON string into Python list of dicts
            data = json.load(json_str)

            # Create list of Card instances
            noble_list = [Noble(**noble) for noble in data]
            self.noble_deck = NobleDeck(nobles=noble_list)

    @abstractmethod
    def get_ruleset(self) -> Ruleset:
        pass

    def make_game(self, game_id:str =str(uuid4())) -> Game:
        return Game(game_id, self.get_ruleset())


class OnePlayerGameFactory(GameFactory):
    def __init__(self):
        rules = [
            GameEndRule(rule_value=15),
            TokenWiothdrawlLimitRule(),
            MaximallyFilledTeirsRule(rule_value=4),
            MaxReservationsRule(rule_value=3),
            MaxPlayerTokenHoldingRule(rule_value=10),
            NumTokensInStartingSupplyRule(
                rule_value={
                    TokenType.GOLD: 5,
                    TokenType.DIAMOND: 8,
                    TokenType.SAPPHIRE: 8,
                    TokenType.EMERALD: 8,
                    TokenType.RUBY: 8,
                    TokenType.ONYX: 8,
                }
            ),
            RulesSelectNobleForPlayerRule(),
            MinTokensToRequest2Rule(rule_value=4),
            Request3DifferentTokensRule(),
            RequestOnly2Or3TokensRule(),
            NumNoblesToStartRule(rule_value=6),
            NumPlayersRule(rule_value=1),
        ]
        self.rules = {rule.__class__.__name__: rule for rule in rules}
        self._load_deck_from_file("shared/decks/cards.json")
        self._load_noble_deck_from_file("shared/decks/nobles.json")

    def get_ruleset(self):
        return Ruleset(deck=self.card_deck, noble_deck=self.noble_deck, rules=self.rules)
