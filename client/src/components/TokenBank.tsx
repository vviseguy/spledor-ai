import React, { useState } from 'react';
import { API_BASE_URL } from '../config';

interface TokenBankProps {
  tokens: Record<string, number>;
  isInteractive: boolean;
  gameId: string;
  playerId?: string;
}

export const TokenBank: React.FC<TokenBankProps> = ({ tokens, isInteractive, gameId, playerId }) => {
  const [selectedTokens, setSelectedTokens] = useState<Record<string, number>>({});

  const handleTokenClick = (tokenType: string) => {
    if (!isInteractive) return;
    setSelectedTokens((prev) => {
      const nextCount = (prev[tokenType] || 0) + 1;
      return { ...prev, [tokenType]: nextCount };
    });
  };

  const takeTokens = async () => {
    if (!playerId) return;
    const body = { playerId, tokens: selectedTokens };

    const res = await fetch(`${API_BASE_URL}/game/${gameId}/take_tokens`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });

    if (!res.ok) {
      console.error("Error taking tokens");
    } else {
      // Reset selected tokens or re-fetch game state
      setSelectedTokens({});
    }
  };

  return (
    <div className="token-bank">
      <h3>Token Bank</h3>
      <div className="token-list">
        {Object.entries(tokens).map(([tokenType, count]) => (
          <div 
            key={tokenType} 
            className={`token ${isInteractive ? 'clickable' : ''}`} 
            title={`Token: ${tokenType}`}
            onClick={() => handleTokenClick(tokenType)}
          >
            {tokenType}: {count}
          </div>
        ))}
      </div>
      {isInteractive && (
        <div>
          <h4>Selected Tokens:</h4>
          <pre>{JSON.stringify(selectedTokens, null, 2)}</pre>
          <button onClick={takeTokens}>Confirm Tokens</button>
        </div>
      )}
    </div>
  );
};
