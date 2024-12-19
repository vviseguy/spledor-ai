import { useEffect, useState } from 'react';
import { GameState } from '../types/GameState';
import { API_BASE_URL } from '../config';

export function useGamePolling(gameId: string) {
  const [gameState, setGameState] = useState<GameState | null>(null);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const fetchGameState = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/game/${gameId}`);
        const data = await res.json();
        setGameState(data);
      } catch (error) {
        console.error("Error fetching game state:", error);
      }
    };

    fetchGameState();
    intervalId = setInterval(fetchGameState, 500); // Poll every 0.5s

    return () => clearInterval(intervalId);
  }, [gameId]);

  return gameState;
}
