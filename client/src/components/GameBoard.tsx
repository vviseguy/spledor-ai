import React from 'react';
import { useLocation } from 'react-router-dom';
import { useGamePolling } from '../hooks/useGamePolling';
import { GameState } from '../types/GameState';
import { TokenBank } from './TokenBank';
import { CardGrid } from './CardGrid';
import { NobleArea } from './NobleArea';
import { PlayerDashboard } from './PlayerDashboard';
import { TurnIndicator } from './TurnIndicator';
import { ActionLog } from './ActionLog';

interface GameBoardProps {
    gameId: string;
}

export const GameBoard: React.FC<GameBoardProps> = ({ gameId }) => {
    const location = useLocation();
    const urlParams = new URLSearchParams(location.search);
    const playerId = urlParams.get('playerId') || undefined;

    const gameState: GameState | null = useGamePolling(gameId);

    if (!gameState) {
        return <div>Loading game state...</div>;
    }

    const isPlayer = Boolean(playerId);
    const activePlayer = gameState.players[gameState.current_turn];
    const isPlayerTurn = activePlayer.playerId === playerId;

    return (
        <div className="game-container">
            <TurnIndicator
                currentTurn={gameState.current_turn}
                players={gameState.players}
            />

            <NobleArea nobles={gameState.nobles} />

            <div className="board-section">
                <CardGrid
                    visibleCards={gameState.visible_cards}
                    isInteractive={isPlayer && isPlayerTurn}
                    playerId={playerId}
                    gameId={gameId}
                />
            </div>

            <TokenBank
                tokens={gameState.board_tokens}
                isInteractive={isPlayer && isPlayerTurn}
                gameId={gameId}
                playerId={playerId}
            />

            <div className="players-section">
                {gameState.players.map(player => (
                    <PlayerDashboard
                        key={player.playerId}
                        player={player}
                        isCurrentPlayer={player.playerId === playerId}
                        gameId={gameId}
                    />

                ))}
            </div>

            <ActionLog />

            {!isPlayer && (
                <div className="spectator-banner">
                    You are watching this game as a spectator.
                </div>
            )}
        </div>
    );
};
