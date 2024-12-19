import React from 'react';
import { PlayerState } from '../types/GameState';

interface TurnIndicatorProps {
  currentTurn: number;
  players: PlayerState[];
}

export const TurnIndicator: React.FC<TurnIndicatorProps> = ({ currentTurn, players }) => {
  const currentPlayer = players[currentTurn];
  return (
    <div className="turn-indicator">
      <h2>Current Turn: {currentPlayer.playerId}</h2>
    </div>
  );
};
