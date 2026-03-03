"""
MCTS (Monte Carlo Tree Search) implementation for the 2048 AI.
Handles the search tree, node expansion, dirichlet noise for exploration, 
and backpropagation of values. References `ai.model.NUM_TILE_CHANNELS`.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from ai.model import NUM_TILE_CHANNELS

class Node:
    def __init__(self, prior_prob, parent=None, action_taken=None):
        self.prior_prob = prior_prob
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        for action, prob in action_probs.items():
            if action not in self.children:
                self.children[action] = Node(prior_prob=prob, parent=self, action_taken=action)

    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, c_puct=1.5, dirichlet_alpha=0.03, dirichlet_epsilon=0.25):
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @staticmethod
    def encode_state(grid):
        """
        Transform 2048 grid into a one-hot encoding across tile powers.
        Output shape: (NUM_TILE_CHANNELS, 4, 4) unbatched
        """
        flat = np.asarray(grid, dtype=np.int64).ravel()
        one_hot = np.zeros((NUM_TILE_CHANNELS, 16), dtype=np.float32)
        nonzero_mask = flat > 0
        # Empty cells map to channel 0
        one_hot[0, ~nonzero_mask] = 1.0
        # Nonzero cells: compute power of 2 and use as channel index
        powers = np.zeros(16, dtype=np.int64)
        powers[nonzero_mask] = np.log2(flat[nonzero_mask]).astype(np.int64)
        valid = powers < NUM_TILE_CHANNELS
        indices = np.where(nonzero_mask & valid)[0]
        one_hot[powers[indices], indices] = 1.0
        return one_hot.reshape((NUM_TILE_CHANNELS, 4, 4))

    @staticmethod
    def encode_states_batch(grids):
        """
        Vectorized batch encoding of multiple grids at once.
        grids: list of flat grids (each a list/array of 16 ints)
        Output shape: (batch, NUM_TILE_CHANNELS, 4, 4)
        """
        n = len(grids)
        flat = np.array(grids, dtype=np.int64).reshape(n, 16)
        one_hot = np.zeros((n, NUM_TILE_CHANNELS, 16), dtype=np.float32)
        nonzero_mask = flat > 0
        # Empty cells -> channel 0
        one_hot[:, 0, :][~nonzero_mask] = 1.0
        # Nonzero cells: compute power of 2
        powers = np.zeros_like(flat)
        powers[nonzero_mask] = np.log2(flat[nonzero_mask]).astype(np.int64)
        valid = nonzero_mask & (powers < NUM_TILE_CHANNELS)
        batch_idx, cell_idx = np.where(valid)
        one_hot[batch_idx, powers[batch_idx, cell_idx], cell_idx] = 1.0
        return one_hot.reshape((n, NUM_TILE_CHANNELS, 4, 4))

    def search_root(self, game, model, device):
        """
        Evaluates the root node directly using the model to expand the first layer.
        Returns the root node.
        """
        root = Node(prior_prob=1.0)
        
        # Fast path: evaluate root
        state_tensor = torch.from_numpy(self.encode_state(game._flat_grid)).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, _ = model(state_tensor)
            
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return root
            
        policy = F.softmax(policy_logits[0], dim=0).cpu().numpy()
        action_probs = {m: policy[m] for m in valid_moves}
        
        # Normalize prior probabilities for valid moves
        sum_probs = sum(action_probs.values())
        if sum_probs > 0:
            action_probs = {m: p / sum_probs for m, p in action_probs.items()}
        else:
            action_probs = {m: 1.0 / len(valid_moves) for m in valid_moves}
            
        root.expand(action_probs)
        return root

    def apply_dirichlet_noise(self, root):
        """Applies Dirichlet noise to the root node for exploration."""
        if not root.is_expanded():
            return
            
        valid_moves = list(root.children.keys())
        if self.dirichlet_epsilon > 0 and len(valid_moves) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_moves))
            for i, action in enumerate(valid_moves):
                child = root.children[action]
                child.prior_prob = (1 - self.dirichlet_epsilon) * child.prior_prob + self.dirichlet_epsilon * noise[i]

    def get_action_prob(self, root, temperature=1.0):
        """Returns the action probabilities based on visit counts at the root."""
        action_visits = {}
        valid_moves = list(root.children.keys())
        
        for action, child in root.children.items():
            action_visits[action] = child.visit_count
            
        sum_visits = sum(action_visits.values())
        if sum_visits == 0:
            if not valid_moves:
                return {}
            return {m: 1.0/len(valid_moves) for m in valid_moves}
            
        if temperature == 0:
            # Deterministic, argmax
            best_action = max(action_visits, key=action_visits.get)
            return {m: 1.0 if m == best_action else 0.0 for m in valid_moves}
            
        # Normally scale by temperature, but in standard AlphaZero temp=1 just normalizes
        # For our 2048, we mainly use temp=1 (proportional) or temp=0 (argmax) handled by caller
        return {m: v / sum_visits for m, v in action_visits.items()}

    def search_leaf(self, game, root_node):
        """
        Traverses MCTS tree to a leaf node using a simulated game instance.
        Returns the (search_path, leaf_game) where leaf_game is at the target state.
        This does NOT run neural net evaluation.
        """
        node = root_node
        
        # Fast deep copy of game state
        sim_game = game.clone()
        
        search_path = [node]
        
        while node.is_expanded():
            action, node = self.select_child(node)
            sim_game.move(action)
            search_path.append(node)
            
        return search_path, sim_game

    def select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        total_node_visits = node.visit_count
        
        for action, child in node.children.items():
            u = self.c_puct * child.prior_prob * math.sqrt(total_node_visits) / (1 + child.visit_count)
            score = child.value + u 
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def backpropagate_leaf(self, search_path, leaf_game, value, policy):
        """
        Expands the leaf node (if not terminal) and backpropagates the value.
        """
        leaf_node = search_path[-1]
        
        if not leaf_game.game_over and policy is not None:
            valid_moves = leaf_game.get_valid_moves()
            if valid_moves:
                action_probs = {m: policy[m] for m in valid_moves}
                sum_probs = sum(action_probs.values())
                if sum_probs > 0:
                    action_probs = {m: p / sum_probs for m, p in action_probs.items()}
                else:
                    action_probs = {m: 1.0 / len(valid_moves) for m in valid_moves}
                    
                leaf_node.expand(action_probs)
                
        # Backpropagate
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1