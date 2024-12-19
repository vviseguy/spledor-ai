export interface Card {
    cardId: string;
    tier: number;
    cost: Record<string, number>;
    points: number;
    bonus: string;
  }
  
  export interface Noble {
    nobleId: string;
    requirements: Record<string, number>;
    points: number;
  }
  
  export interface PlayerState {
    playerId: string;
    resources: Record<string, number>;
    reservedCards: Card[];
    ownedCards: Card[];
    score: number;
  }
  
  export interface GameState {
    players: PlayerState[];
    board_tokens: Record<string, number>;
    visible_cards: Record<number, Card[]>;
    nobles: Noble[];
    current_turn: number;
    status: "pending" | "active" | "completed";
  }
  