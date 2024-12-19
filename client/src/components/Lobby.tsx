import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

export const Lobby: React.FC = () => {
  const [gameId, setGameId] = useState('');
  const [playerId, setPlayerId] = useState('');
  const navigate = useNavigate();

  const joinAsPlayer = () => {
    if (!gameId || !playerId) return;
    navigate(`/game/${gameId}?playerId=${playerId}`);
  };

  const watchGame = () => {
    if (!gameId) return;
    navigate(`/game/${gameId}`);
  };

  return (
    <div className="lobby">
      <h1>Splendor Lobby</h1>
      <input 
        placeholder="Enter Game ID" 
        value={gameId} 
        onChange={(e) => setGameId(e.target.value)} 
      />
      <input 
        placeholder="Enter Player ID (optional if joining as player)" 
        value={playerId} 
        onChange={(e) => setPlayerId(e.target.value)} 
      />
      <button onClick={joinAsPlayer} disabled={!gameId || !playerId}>Join as Player</button>
      <button onClick={watchGame} disabled={!gameId}>Watch Game</button>
    </div>
  );
};
