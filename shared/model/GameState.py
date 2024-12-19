from enum import StrEnum
from pydantic import BaseModel
from shared.model.PlayerState import PlayerState, PlayerStateDto
from shared.model.BoardState import BoardState, BoardStateDto
from typing import List, Optional

class GameStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"

class GameStateDto(BaseModel):
    game_id: str
    players: List[PlayerStateDto]
    board_state: BoardStateDto
    current_turn: int
    status: GameStatus


class GameState(GameStateDto):
    game_id: str
    players: List[PlayerState]
    board_state: BoardState
    current_turn: int
    status: GameStatus

    @property
    def dto(self):
        players = [player.dto for player in self.players]
        return GameStateDto(self.game_id, players, self.board_state.dto, self.current_turn, self.status)

