import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Any

from ai.GameInterface import GameInterface

device = 'cuda' if torch.cuda.is_available() else 'cpu'


############################################
# GameInterface (Must be provided by user) #
############################################
#
# class GameInterface:
#     def reset(self) -> None: ...
#     def current_state(self) -> Any: ...
#     def legal_actions(self) -> List[int]: ...
#     def step(self, action: int) -> None: ...
#     def is_terminal(self) -> bool: ...
#     def final_score(self) -> float: ...
#     def get_state_representation(self) -> Any: ...
#
# Please implement this interface according to your game logic.
#

############################################
# Neural Network Implementation (PyTorch)  #
############################################

class SplendorNet(nn.Module):
    def __init__(self, state_shape: Tuple[int, ...], num_actions: int):
        super(SplendorNet, self).__init__()
        # Example: If state_shape is something like (state_dim,) i.e. a flat vector.
        # Adjust architecture as needed for your representation.
        
        # Flatten input dimension
        state_dim = 1
        for d in state_shape:
            state_dim *= d

        # A simple MLP
        hidden_dim = 256
        self.fc_body = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, num_actions)
            # We will apply softmax outside when needed
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Assuming value range is [-1,1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, *state_shape]
        # Flatten
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)
        out = self.fc_body(out)
        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        return policy_logits, value


class NeuralNetwork:
    def __init__(self, state_shape: Tuple[int, ...], num_actions: int, lr=1e-3, device=device):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.device = device
        self.model = SplendorNet(state_shape, num_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Losses: 
        # Policy loss: cross-entropy
        # Value loss: MSE
        self.policy_loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.value_loss_fn = nn.MSELoss(reduction='mean')

    def predict(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # states: [batch, *state_shape]
        self.model.eval()
        with torch.no_grad():
            states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
            policy_logits, values = self.model(states_t)
            # Convert to numpy
            policy_logits = policy_logits.cpu().numpy()
            values = values.cpu().numpy()
        return policy_logits, values

    def train_on_batch(self, states: np.ndarray, target_policies: np.ndarray, target_values: np.ndarray) -> float:
        # states: [batch, *state_shape]
        # target_policies: [batch, num_actions], this is a distribution (probabilities)
        # target_values: [batch, 1], scalar values
        self.model.train()
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        target_policies_t = torch.tensor(target_policies, dtype=torch.float32, device=self.device)
        target_values_t = torch.tensor(target_values, dtype=torch.float32, device=self.device)

        # Forward
        policy_logits, values = self.model(states_t)  # policy_logits: [batch, num_actions], values: [batch,1]

        # The target_policies are probabilities. We use cross-entropy:
        # CrossEntropyLoss expects class indices by default. To handle distribution targets,
        # one method is to use -sum(p * log_softmax) = KL divergence-like measure.
        # Alternatively, we can sample from the distribution or approximate.
        # A simpler approach: treat target_policies as fixed distributions and
        # minimize KL-div(P||Q). That can be done by: loss = - sum(p * log_softmax)
        log_probs = torch.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policies_t * log_probs).sum(dim=1).mean()

        value_loss = self.value_loss_fn(values.view(-1), target_values_t.view(-1))
        loss = policy_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


############################################
# MCTS Implementation                      #
############################################

class MCTSNode:
    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False

    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def u_value(self, parent_visit_count, c_puct):
        return c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)


class MCTS:
    def __init__(self, neural_network: NeuralNetwork, num_actions: int, c_puct=1.0, num_simulations=800):
        self.nn = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.num_actions = num_actions
        self.root = MCTSNode()

    def run(self, game: GameInterface):
        for _ in range(self.num_simulations):
            self._simulate(copy.deepcopy(game), self.root)

    def _simulate(self, game: GameInterface, node: MCTSNode):
        if game.is_terminal():
            return game.final_score()

        if not node.is_expanded:
            # Expand node
            policy_logits, value = self.nn.predict(np.array([game.current_state()]))
            policy_logits = policy_logits[0]
            value = value[0][0]

            legal_actions = game.legal_actions()
            mask = np.zeros(self.num_actions, dtype=bool)
            mask[legal_actions] = True

            exp_logits = np.exp(policy_logits - np.max(policy_logits[mask])) * mask
            priors = exp_logits / np.sum(exp_logits[mask])

            for a in legal_actions:
                node.children[a] = MCTSNode(prior=priors[a])

            node.is_expanded = True
            return value

        # If node is expanded, choose action with maximum Q+U
        total_visits = sum(child.visit_count for child in node.children.values())
        best_action, best_value = None, -float('inf')
        for a, child in node.children.items():
            u = child.q_value() + child.u_value(total_visits, self.c_puct)
            if u > best_value:
                best_value = u
                best_action = a

        game.step(best_action)
        value = self._simulate(game, node.children[best_action])
        node.children[best_action].value_sum += value
        node.children[best_action].visit_count += 1
        return value

    def get_action_probabilities(self, temperature=1.0):
        actions = sorted(self.root.children.keys())
        visits = np.array([self.root.children[a].visit_count for a in actions])
        if temperature != 1.0:
            visits = visits ** (1/temperature)
        probs = visits / np.sum(visits)
        return actions, probs


############################################
# Replay Buffer                            #
############################################

class ReplayBuffer:
    def __init__(self, max_size=500000):
        self.max_size = max_size
        self.data = []

    def store(self, state, policy, value):
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append((state, policy, value))

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        states, policies, values = zip(*batch)
        return np.array(states), np.array(policies), np.array(values)


############################################
# Self-Play and Training Loop              #
############################################

def self_play_episode(game: 'GameInterface', mcts: MCTS):
    trajectory = []
    while not game.is_terminal():
        mcts.run(game)
        actions, probs = mcts.get_action_probabilities(temperature=1.0)

        full_policy = np.zeros(mcts.num_actions)
        for a, p in zip(actions, probs):
            full_policy[a] = p

        trajectory.append((game.get_state_representation(), full_policy))
        action = np.random.choice(actions, p=probs)
        game.step(action)

    final_value = game.final_score()
    data = [(s, p, final_value) for (s, p) in trajectory]
    return data

def generate_data(neural_net: NeuralNetwork, 
                  game_class, 
                  num_games=25000, 
                  num_simulations=800, 
                  c_puct=1.0, 
                  num_actions=5000):
    data = []
    for _ in range(num_games):
        game = game_class()
        game.reset()
        mcts = MCTS(neural_net, num_actions=num_actions, c_puct=c_puct, num_simulations=num_simulations)
        ep_data = self_play_episode(game, mcts)
        data.extend(ep_data)
    return data

def evaluate_models(candidate_net: NeuralNetwork, 
                    best_net: NeuralNetwork, 
                    game_class, 
                    num_eval_games=1000, 
                    num_actions=5000):
    """
    Evaluate candidate_net vs best_net by comparing their average final scores.
    This is a single-player scenario, so we just compare final scores.
    """
    candidate_scores = []
    best_scores = []

    for _ in range(num_eval_games):
        # Candidate
        game_c = game_class()
        game_c.reset()
        candidate_mcts = MCTS(candidate_net, num_actions=num_actions)
        # Run one full episode
        for s, p, v in self_play_episode(game_c, candidate_mcts):
            pass
        candidate_scores.append(game_c.final_score())

        # Best
        game_b = game_class()
        game_b.reset()
        best_mcts = MCTS(best_net, num_actions=num_actions)
        for s, p, v in self_play_episode(game_b, best_mcts):
            pass
        best_scores.append(game_b.final_score())

    return np.mean(candidate_scores) - np.mean(best_scores)


def training_loop(game_class, 
                  state_shape, 
                  num_actions=5000, 
                  initial_model_path=None,
                  max_replay_size=500000,
                  device=device):
    best_net = NeuralNetwork(state_shape, num_actions, device=device)
    if initial_model_path is not None:
        best_net.load(initial_model_path)

    replay_buffer = ReplayBuffer(max_size=max_replay_size)
    old_best_net = NeuralNetwork(state_shape, num_actions, device=device)
    if initial_model_path is not None:
        old_best_net.load(initial_model_path)
    else:
        # Copy initial weights
        old_best_net.model.load_state_dict(best_net.model.state_dict())

    training_cycles = 0
    evaluation_interval = 1000
    improvement_threshold = 1.0  # Example threshold

    while True:
        # 1. Self-play to generate 25,000 new samples
        new_data = generate_data(best_net, game_class, num_games=25000, num_actions=num_actions)
        for d in new_data:
            replay_buffer.store(*d)

        # 2. Train the network for 1000 steps
        for _ in range(evaluation_interval):
            states, policies, values = replay_buffer.sample(batch_size=2048)
            loss = best_net.train_on_batch(states, policies, values)

        training_cycles += evaluation_interval

        # 3. Evaluate candidate vs old best
        candidate_net = NeuralNetwork(state_shape, num_actions, device=device)
        candidate_net.model.load_state_dict(best_net.model.state_dict())

        result = evaluate_models(candidate_net, old_best_net, game_class, num_actions=num_actions)
        if result > improvement_threshold:
            # Candidate is better
            old_best_net.model.load_state_dict(candidate_net.model.state_dict())
            old_best_net.save("best_model_path")
            print(f"New best model found with improvement {result} over old best!")
        else:
            print(f"No improvement. Candidate vs Best: {result}")

        # Potential stopping criterion or continue indefinitely
        # break if desired.
