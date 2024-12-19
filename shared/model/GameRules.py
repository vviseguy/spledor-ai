from typing import Dict, Optional
from shared.model.Card import CardTier
from shared.model.GameRule import GameRule
from shared.model.Move import Move, MoveType
from shared.model.GameState import GameState
from shared.model.Noble import Noble
from shared.model.PlayerState import PlayerState
from shared.model.TokenType import TokenType


class GameEndRule(GameRule):
    rule_value: int

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if game_state.current_turn == 0:
            winner = max(game_state.players, key=lambda p: (p["score"], -len(p.get("ownedCards", []))))
            if winner["score"] >= self.rule_value:
                self.callback()


class TokenWiothdrawlLimitRule(GameRule):
    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        token_bank = game_state.board_state.tokenBank

        # Player token stacks
        for player in game_state.players:
            for tokenType, amount in player.resources.items():
                # balance with Gold
                if tokenType != TokenType.GOLD and amount < 0:
                    player.resources[TokenType.GOLD] += amount
                    token_bank[TokenType.GOLD] -= amount

                    player.resources[tokenType] -= amount
                    token_bank[tokenType] += amount

            # throw error if there isnt enought Gold
            if player.resources[TokenType.GOLD] < 0:
                self.callback()

        # Board token stacks
        for token, count in token_bank.items():
            if count < 0:
                self.callback()


class MaximallyFilledTeirsRule(GameRule):

    rule_value: int

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        for tier, visible_cards in game_state.board_state.availableCards.items():
            # flag if we should have delt cards but we didnt
            if len(visible_cards) < self.rule_value and game_state.board_state.deck.len(tier):
                self.callback()


class MaxReservationsRule(GameRule):

    rule_value: int

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        for player in game_state.players:
            if len(player.reservedCards) > self.rule_value:
                self.callback()


class MaxPlayerTokenHoldingRule(GameRule):

    rule_value: int

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        for player in game_state.players:
            total_tokens = sum(player.resources.values())
            if total_tokens > self.rule_value:
                self.callback()


class NumTokensInStartingSupplyRule(GameRule):

    rule_value: Dict[TokenType, int]

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if not previous_move:
            token_bank = game_state.board_state.tokenBank
            for token, count in token_bank.items():
                if count != self.rule_value.get(token, 0):
                    self.callback()


class RulesSelectNobleForPlayerRule(GameRule):
    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if previous_move and previous_move.type == MoveType.BUY:
            player: PlayerState = game_state.players[previous_move.playerIndex]

            nobles = game_state.board_state.nobles
            for noble_i in range(len(nobles)):
                noble:Noble = nobles[noble_i]
                if all(
                    sum(card.bonus == tokenType for card in player.ownedCards) >= amount
                    for tokenType, amount in noble.requirements.items()
                ):
                    nobles.pop(noble_i)
                    player.noblesOwned.append(noble)
                    
                    player.score += noble.points
                    # self.callback()


# class InvalidNobleOrCardRequestRule(GameRule):
#     def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
#         if previous_move:
#             if previous_move.type == "buy_card" or previous_move.type == "reserve_card":
#                 card_id = previous_move.details.get("cardId")
#                 tier = previous_move.details.get("tier")

#                 if card_id and card_id not in game_state.board_state.get("availableCards", {}).get(tier, []):
#                     self.callback()

#             if previous_move.type == "claim_noble":
#                 noble_id = previous_move.details.get("nobleId")
#                 if noble_id and noble_id not in [noble.noble_id for noble in game_state.ruleset.noble_deck.nobles]:
#                     self.callback()


class MinTokensToRequest2Rule(GameRule):
    rule_value: int

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if previous_move and previous_move.type == MoveType.TAKE_TOKENS:
            tokens = previous_move.tokens
            if len(tokens) == 1:
                token_type, count = list(tokens.items())[0]
                if count != 2 or game_state.board_state.tokenBank[token_type] < self.rule_value:
                    self.callback()


class Request3DifferentTokensRule(GameRule):
    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if previous_move and previous_move.type == MoveType.TAKE_TOKENS:
            tokens = previous_move.tokens
            if len(tokens) == 3:
                if any(count != 1 for count in tokens.values()):
                    self.callback()


class RequestOnly2Or3TokensRule(GameRule):
    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if previous_move and previous_move.type == MoveType.TAKE_TOKENS:
            tokens_sum = sum(previous_move.tokens.values())
            if not tokens_sum in [2, 3]:
                self.callback()


class NumNoblesToStartRule(GameRule):
    rule_value: int

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if not previous_move and len(game_state.board_state.nobles) != self.rule_value:
            self.callback()

class NumPlayersRule(GameRule):
    rule_value: int

    def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
        if not previous_move and len(game_state.players) != self.rule_value:
            self.callback()

# class InvalidCardCostRule(GameRule):
#     def check(self, game_state: GameState, previous_move: Optional[Move] = None) -> None:
#         if previous_move and previous_move.type == "buy_card":
#             card_id = previous_move.details.get("cardId")
#             payment = previous_move.details.get("payment", {})
#             card = next(
#                 (
#                     card
#                     for card in game_state.ruleset.deck.tiers[previous_move.details.get("tier", "")]
#                     if card.card_id == card_id
#                 ),
#                 None,
#             )

#             if card:
#                 total_payment = payment.copy()
#                 for resource, cost in card.cost.items():
#                     available = total_payment.get(resource, 0)
#                     if cost > available:
#                         total_payment["gold"] = total_payment.get("gold", 0) - (cost - available)
#                         total_payment[resource] = 0
#                     else:
#                         total_payment[resource] -= cost

#                 if any(value < 0 for value in total_payment.values()):
#                     self.callback()
