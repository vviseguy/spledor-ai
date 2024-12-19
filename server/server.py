from fastapi import FastAPI, HTTPException, Path, Header, Body
from typing import Dict, Optional
from uuid import uuid4
from pydantic import BaseModel
from shared.model.Move import Move
from shared.model.Ruleset import Ruleset
from shared.setup.GameFactory import DEFAULT_RULES, OnePlayerGameFactory
from shared.model import GameStateDto

from shared.gameplay.Game import Game
from shared.model.GameState import GameState

# Initialize FastAPI
app = FastAPI(title="Splendor Game API", version="1.0.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Game Manager: Holds active game instances
game_manager: Dict[str, Game] = {}

# --- Helper Classes ---
class PlayerAuth(BaseModel):
    player_id: str
    auth_token: str

# --- Endpoints ---
@app.post("/game", summary="Create a new game")
def create_game(body: Dict) -> GameStateDto:
    player_names = body.get("playerNames")
    ruleset_data = body.get("ruleset", None)

    if not player_names or not isinstance(player_names, list):
        raise HTTPException(status_code=400, detail="Invalid player names")

    game_id = str(uuid4())
    game_manager[game_id] = OnePlayerGameFactory().make_game(game_id)

    for name in player_names:
        game_manager[game_id].add_player(name)

    return game_manager[game_id].state.dto


@app.post("/game/{game_id}/start", summary="Start the game")
def start_game(game_id: str = Path(...)):
    if game_id not in game_manager:
        raise HTTPException(status_code=404, detail="Game not found")

    game_manager[game_id].start_game()
    return {"message": "Game started successfully"}


@app.get("/game/{game_id}", summary="Retrieve game state")
def get_game_state(game_id: str = Path(...)):
    if game_id not in game_manager:
        raise HTTPException(status_code=404, detail="Game not found")

    return game_manager[game_id].get_summary()


@app.get("/game/{game_id}/current-turn", summary="Get current turn")
def current_turn(game_id: str = Path(...)):
    if game_id not in game_manager:
        raise HTTPException(status_code=404, detail="Game not found")

    game = game_manager[game_id]
    current_player = game.state.players[game.state.current_turn]
    return {"playerId": current_player.player_id, "playerIndex": game.state.current_turn}


@app.post("/game/{game_id}/player", summary="Register a player")
def add_player(game_id: str = Path(...), body: Dict = Body(...)):
    player_name = body.get("playerName")
    if not player_name:
        raise HTTPException(status_code=400, detail="Player name is required")

    if game_id not in game_manager:
        raise HTTPException(status_code=404, detail="Game not found")

    player_id = game_manager[game_id].add_player(player_name)
    return {"playerId": player_id, "authToken": str(uuid4())}  # Simple auth


@app.post("/game/{game_id}/move", summary="Submit a move")
def submit_move(game_id: str = Path(...), Authorization: str = Header(...), move: Move = Body(...)):
    if game_id not in game_manager:
        raise HTTPException(status_code=404, detail="Game not found")

    # Simple auth validation (replace with real auth logic)
    if not Authorization:
        raise HTTPException(status_code=401, detail="Unauthorized")

    game = game_manager[game_id]
    try:
        game.apply_move(move)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid move: {str(e)}")
    
    return {"message": "Move accepted"}


@app.get("/game/{game_id}/summary", summary="Get game summary")
def game_summary(game_id: str = Path(...)):
    if game_id not in game_manager:
        raise HTTPException(status_code=404, detail="Game not found")

    return game_manager[game_id].get_summary()


# --- Specific Endpoints for Deck and Nobles ---
@app.get("/deck/{card_id}", summary="Get card details")
def get_card(card_id: str = Path(...)):
    for game in game_manager.values():
        for tier in game.ruleset.deck.tiers.values():
            for card in tier:
                if card.card_id == card_id:
                    return card
    raise HTTPException(status_code=404, detail="Card not found")


@app.get("/nobles/{noble_id}", summary="Get noble details")
def get_noble(noble_id: str = Path(...)):
    for game in game_manager.values():
        for noble in game.ruleset.noble_deck.nobles:
            if noble.noble_id == noble_id:
                return noble
    raise HTTPException(status_code=404, detail="Noble not found")


# --- Main Function ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
