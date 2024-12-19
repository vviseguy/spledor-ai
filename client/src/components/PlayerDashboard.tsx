import React, { useCallback } from 'react';
import { PlayerState } from '../types/GameState';
import { API_BASE_URL } from '../config';

interface PlayerDashboardProps {
  player: PlayerState;
  isCurrentPlayer: boolean;
  gameId: string;
}


export const PlayerDashboard: React.FC<PlayerDashboardProps> = ({ player, isCurrentPlayer, gameId }) => {

  const onDropToReserve = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const cardId = e.dataTransfer.getData('cardId');
    const tier = parseInt(e.dataTransfer.getData('tier'), 10);

    const body = {
      playerId: player.playerId,
      cardId,
      tier
    };

    const res = await fetch(`${API_BASE_URL}/game/${gameId}/reserve_card`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      console.error("Error reserving card");
    }
  }, [player.playerId, gameId]);

  const onDropToBuy = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    const cardId = e.dataTransfer.getData('cardId');
    const tier = parseInt(e.dataTransfer.getData('tier'), 10);

    const body = {
      playerId: player.playerId,
      cardId,
      tier
    };

    const res = await fetch(`${API_BASE_URL}/game/${gameId}/buy_card`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      console.error("Error buying card");
    }
  }, [player.playerId, gameId]);

  return (
    <div className="player-dashboard">
      <h3>{player.playerId} {isCurrentPlayer && '(You)'}</h3>
      <div>Score: {player.score}</div>

      <h4>Your Resources:</h4>
      <div className="resource-list">
        {Object.entries(player.resources).map(([res, val]) => (
          <div key={res}>{res}: {val}</div>
        ))}
      </div>

      <h4>Reserved Cards (drop here to reserve):</h4>
      <div 
        className="reserved-area" 
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDropToReserve}
      >
        {player.reservedCards.map(card => (
          <div className="card reserved" key={card.cardId}>
            {card.cardId} - {card.points} points
          </div>
        ))}
      </div>

      <h4>Owned Cards (drop here to buy):</h4>
      <div 
        className="owned-area" 
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDropToBuy}
      >
        {player.ownedCards.map(card => (
          <div className="card owned" key={card.cardId}>
            {card.cardId} - {card.points} points (Bonus: {card.bonus})
          </div>
        ))}
      </div>
    </div>
  );
};
