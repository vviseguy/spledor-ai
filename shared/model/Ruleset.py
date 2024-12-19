from shared.model.GameRule import GameRule
from shared.model.Card import Deck
from shared.model.GameState import GameStateDto
from shared.model.Noble import NobleDeck
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class RulesetDto(BaseModel):
    deck: Deck
    noble_deck: NobleDeck
    rules: Dict[str, Any]


class Ruleset(RulesetDto):
    _rules: Dict[str, GameRule] = {}

    @property
    def rules(self):
        # This acts as the getter
        return self._rules
    
    @rules.setter
    def rules(self, rules: Dict[str, Any]):
        self._rules = {}

        # Rule name is the class of the rule
        for rule_name, rule in rules:
            try:
                rule_obj = GameRule.get_subclass_by_name(rule_name)(rule)
                self._rules[rule_name] = rule_obj
            except:
                pass

    class Config:
        arbitrary_types_allowed = True

    @property
    def dto(self):
        rules = {rule.__class__.__name__: rule.rule_value for rule_name, rule in self.rules.items()}
        return GameStateDto(self.deck, self.noble_deck, rules)
    
    def getRuleValue(self, ruleName: type):
        return self.rules[ruleName.__name__].rule_value
