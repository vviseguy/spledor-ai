# main.py

import torch
from .GameInterface import GameInterface
from .training import NeuralNetwork, MCTS, ReplayBuffer, training_loop

def main():
    # Define game parameters
    state_shape = (9,)  # For Splendor, a flattened 3x3 board
    num_actions = 45  # 9 possible actions in Splendor

    # Initialize training
    training_loop(
        game_class=GameInterface,
        state_shape=state_shape,
        num_actions=num_actions,
        initial_model_path=None,  # Path to a pre-trained model if available
        max_replay_size=500000,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_cycles=10000,            # Define the number of training cycles
        batch_size=512,              # Adjust based on your hardware
        evaluation_interval=1000,
        improvement_threshold=0.1,   # Adjust based on empirical results
        save_path="best_model.pth"   # Define where to save the best model
    )

if __name__ == "__main__":
    # game = 
    print(GameInterface().current_state().shape)
    main()
