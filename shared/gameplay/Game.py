import copy
from tkinter import Place
from typing import Dict
from shared.model.Noble import NobleDeck
from shared.model.BoardState import BoardState
from shared.model.Card import Card, Deck
from shared.model.Move import MoveType
from shared.model.Move import Move
from shared.model.PlayerState import PlayerState
from shared.model.Ruleset import Ruleset
from shared.model.GameState import GameState, GameStatus
from shared.model.TokenType import TokenType
from shared.model.GameRules import *


class Game:
    def __init__(self, game_id: str, ruleset: Ruleset):
        self.state = self._initialize_game_state(game_id, ruleset)
        self.ruleset = ruleset
        self.move_history = []

    def add_player(self, player_name: str) -> str:
        if len(self.state.players) >= self.ruleset.getRuleValue(NumPlayersRule):
            raise ValueError("This game is already full.")
        if self.state.status != GameStatus.PENDING:
            raise ValueError("Cannot add players after the game has started.")
        player_id = f"player_{len(self.state.players) + 1}"
        self.state.players.append(PlayerState(player_id=player_id, name=player_name))
        return player_id

    def remove_player(self, player_id: str):
        if self.state.status != GameStatus.PENDING:
            raise ValueError("Cannot remove players after the game has started.")
        self.state.players = [p for p in self.state.players if p.player_id != player_id]

    def start_game(self):
        if len(self.state.players) < self.ruleset.getRuleValue(NumPlayersRule):
            raise ValueError(f"{self.ruleset.getRuleValue(NumPlayersRule)} players are required to start the game.")
        self.state.status = GameStatus.ACTIVE
        self.state.current_turn = 0

    def apply_move(self, move: Move):
        save_state = copy.deepcopy(self.state)
        try:
            move = copy.deepcopy(move)
            move.playerIndex = move.playerIndex or self.state.current_turn
            self._process_move(move)
            self._validate_state(move)
        except Exception as e:
            self.state = save_state
            return Exception(f"Unable to perform move: {str(e)}")

        self.move_history.append(move)
        self.state.current_turn = (self.state.current_turn + 1) % len(self.state.players)

    def test_move(self, move: Move):
        save_state = copy.deepcopy(self.state)
        try:
            self._process_move(move)
            self._validate_state(move)
            return True
        except Exception as e:
            return False
        finally:
            self.state = save_state

    def undo_move(self):
        # if not self.move_history:
        #     raise ValueError("No moves to undo.")
        # last_move = self.move_history.pop()
        # self._rollback_move(last_move)
        # self.state.current_turn = (self.state.current_turn - 1) % len(self.state.players)
        pass

    def _validate_state(self, previous_move: Move):
        for rule in self.ruleset.rules.values():
            rule.check(self.state, previous_move)

    def _process_move(self, move: Move):
        moving_player_index = move.playerIndex or self.state.current_turn
        moving_player: PlayerState = self.state.players[moving_player_index]

        if move.type == MoveType.BUY:
            card: Card

            if move.card_tier:  # buy from field
                cardList = self.state.board_state.availableCards[move.card_tier]
                card = cardList.pop(move.card_index)
                if self.state.board_state.deck.len(move.card_tier):  # redeal
                    cardList.append(self.state.board_state.deck.pop(move.card_tier))
            else:  # but from reserve
                card = moving_player.reservedCards.pop(move.card_index)

            moving_player.ownedCards.append(card)

            for tokenType, cost in card.cost:
                # evaluate cost to the player
                card_buy_power = sum([card.bonus == tokenType for card in moving_player.ownedCards])
                num_tokens_transferred = max(0, cost - card_buy_power)

                # move tokens back to bank
                moving_player.resources[tokenType] -= num_tokens_transferred
                self.state.board_state.tokenBank[tokenType] += num_tokens_transferred
            
            moving_player.score += card.points

        elif move.type == MoveType.RESERVE:
            card = None

            if move.card_index:  # reserve from field
                cardList = self.state.board_state.availableCards[move.card_tier]
                card = cardList.pop(move.card_index)
                if self.state.board_state.deck.len(move.card_tier):  # redeal
                    cardList.append(self.state.board_state.deck.pop(move.card_tier))
            else:  # reserve from deck
                card = self.state.board_state.deck.pop(move.card_tier)

            moving_player.reservedCards.append(card)

            moving_player.resources[TokenType.GOLD] += 1
            self.state.board_state.tokenBank[TokenType.GOLD] -= 1

        elif move.type == MoveType.TAKE_TOKENS:

            # move tokens to player
            for tokenType, amount in move.tokens:
                moving_player.resources[tokenType] += amount
                self.state.board_state.tokenBank[tokenType] -= amount
        else:
            raise Exception(f"Invalid move type: {move.type}")

    def _rollback_move(self, move: Move):
        # Reverse the effects of the last move
        pass

    def _initialize_game_state(self, game_id: str, ruleset: Ruleset) -> GameState:
        

        #      GameEndRule(15),
        # TokenWiothdrawlLimitRule(),
        # MaximallyFilledTeirsRule(4),
        # MaxReservationsRule(3),
        # MaxPlayerTokenHoldingRule(10),
        # NumTokensInStartingSupplyRule({
        #     TokenType.GOLD:5,
        #     TokenType.DIAMOND:8,
        #     TokenType.SAPPHIRE:8,
        #     TokenType.EMERALD:8,
        #     TokenType.RUBY:8,
        #     TokenType.ONYX:8,
        # }),
        # RulesSelectNobleForPlayerRule(),
        # MinTokensToRequest2Rule(4),
        # Request3DifferentTokensRule(),
        # RequestOnly2Or3TokensRule(),
        # NumNoblesToStartRule(6),
        # NumPlayersRule(1)
        deck: Deck = copy.deepcopy(ruleset.deck)
        noble_deck: NobleDeck = copy.deepcopy(ruleset.noble_deck)
        available_cards = {
            tier: [deck.pop(tier) for _ in range(ruleset.getRuleValue(MaximallyFilledTeirsRule))] for tier in deck.tiers
        }
        nobles = noble_deck.nobles[:ruleset.getRuleValue(NumNoblesToStartRule)]
        print(*nobles)
        token_bank = copy.deepcopy(ruleset.getRuleValue(NumTokensInStartingSupplyRule))
        board_state = BoardState(availableCards=available_cards, nobles=nobles, noble_deck=noble_deck, deck=deck, tokenBank=token_bank)
        return GameState(
            game_id=game_id,
            players=[],
            board_state=board_state,
            current_turn=0,
            status=GameStatus.PENDING,
        )

    def get_summary(self) -> Dict:
        return {
            "gameId": self.state.game_id,
            "status": self.state.status.value,
            "players": [
                {
                    "playerId": p.player_id,
                    "name": p.name,
                    "score": p.score,
                    "resources": p.resources,
                    "reservedCards": p.reserved_cards,
                    "ownedCards": p.owned_cards,
                    "nobles": p.nobles,
                }
                for p in self.state.players
            ],
            "currentTurn": self.state.current_turn,
        }


# (bad/old) Example Usage
# ruleset_config = {
#     "deck": {
#         "tiers": {
#             "tier1": [{"card_id": "c1", "tier": "tier1", "cost": {"ruby": 2}, "points": 1}],
#             "tier2": [],
#             "tier3": [],
#         }
#     },
#     "noble_deck": {"nobles": [{"noble_id": "n1", "requirements": {"ruby": 4}, "points": 3}]},
#     "rules": {
#         "max_tokens": {"gold": 5, "diamond": 7, "sapphire": 7, "emerald": 7, "ruby": 7, "onyx": 7},
#         "points_to_win": 15,
#     },
# }

# ruleset = Ruleset(ruleset_config)
# game = Game(game_id="game_1", ruleset=ruleset)
# player1_id = game.add_player("Alice")
# player2_id = game.add_player("Bob")
# game.start_game()
# print(game.get_summary())
