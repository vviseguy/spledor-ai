import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Lobby } from './components/Lobby';
import { GameBoard } from './components/GameBoard';

export const App: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Lobby />} />
      <Route path="/game/:gameId" element={<GameBoardWrapper />} />
    </Routes>
  );
};

// Wrapper for GameBoard to extract gameId from URL
import { useParams } from 'react-router-dom';
const GameBoardWrapper: React.FC = () => {
  const { gameId } = useParams();
  if (!gameId) return <div>Missing Game ID</div>;
  return <GameBoard gameId={gameId} />;
};
