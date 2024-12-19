import React from 'react';
import { Noble } from '../types/GameState';

interface NobleAreaProps {
  nobles: Noble[];
}

export const NobleArea: React.FC<NobleAreaProps> = ({ nobles }) => {
  return (
    <div className="noble-area">
      <h3>Nobles</h3>
      <div className="noble-list">
        {nobles.map(noble => (
          <div key={noble.nobleId} className="noble-tile">
            <div className="noble-id">{noble.nobleId}</div>
            <div>Points: {noble.points}</div>
            <div>
              Requirements: {Object.entries(noble.requirements).map(([res, val]) => `${res}: ${val}`).join(', ')}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
