import React from 'react';
import { Card } from '../types/GameState';

interface CardGridProps {
  visibleCards: Record<number, Card[]>;
  isInteractive: boolean;
  playerId?: string;
  gameId: string;
}

export const CardGrid: React.FC<CardGridProps> = ({ visibleCards, isInteractive }) => {
  return (
    <div className="card-grid">
      {Object.entries(visibleCards).map(([tier, cards]:any) => (
        <div className="card-tier" key={tier}>
          <h3>Tier {tier}</h3>
          <div className="card-row">
            {cards.map(card => (
              <div 
                key={card.cardId} 
                className={`card ${isInteractive ? 'draggable' : 'static'}`}
                draggable={isInteractive}
                onDragStart={(e) => {
                  e.dataTransfer.setData('cardId', card.cardId);
                  e.dataTransfer.setData('tier', tier);
                }}
              >
                <div className="card-points">Points: {card.points}</div>
                <div className="card-cost">
                  Cost: {Object.entries(card.cost).map(([res, val]) => `${res}: ${val}`).join(', ')}
                </div>
                <div className="card-bonus">Bonus: {card.bonus}</div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};
