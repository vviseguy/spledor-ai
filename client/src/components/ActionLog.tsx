import React from 'react';

export const ActionLog: React.FC = () => {
  // For demonstration, we don't have an actual action log.
  // In a real scenario, you'd fetch the action log from the server or from the game state.
  // For now, just display a placeholder.

  return (
    <div className="action-log">
      <h4>Action Log</h4>
      <ul>
        <li>Player1 reserved a Tier 1 card.</li>
        <li>Player2 took 2 red tokens.</li>
        <li>Player1 bought a card from Tier 2.</li>
      </ul>
    </div>
  );
};
