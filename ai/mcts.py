import math
import numpy as np
import torch
import torch.nn.functional as F

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
    def __init__(self, model, num_simulations=100, c_puct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def encode_state(self, grid):
        """ Transform 2048 grid using log2 representation for NN input """
        tensor = torch.tensor(grid, dtype=torch.float32)
        tensor = torch.where(tensor > 0, torch.log2(tensor), torch.zeros_like(tensor))
        return tensor.unsqueeze(0).unsqueeze(0) # (1, 1, 4, 4)

    def search(self, game):
        """
        Runs MCTS simulations starting from the given game state.
        Returns to root and evaluates action probabilities.
        """
        root = Node(prior_prob=1.0)
        
        # Fast path: evaluate root
        state_tensor = self.encode_state(game.grid)
        device = next(self.model.parameters()).device
        state_tensor = state_tensor.to(device)
        with torch.no_grad():
            policy_logits, _ = self.model(state_tensor)
            
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None
            
        policy = F.softmax(policy_logits[0], dim=0).cpu().numpy()
        action_probs = {m: policy[m] for m in valid_moves}
        
        # Normalize prior probabilities for valid moves
        sum_probs = sum(action_probs.values())
        if sum_probs > 0:
            action_probs = {m: p / sum_probs for m, p in action_probs.items()}
        else:
            action_probs = {m: 1.0 / len(valid_moves) for m in valid_moves}
            
        root.expand(action_probs)

        # MCTS simulations loop
        for _ in range(self.num_simulations):
            node = root
            # Make a copy of the game engine to simulate
            import copy
            sim_game = copy.deepcopy(game)
            sim_game.spawn_tile = lambda: _spawn_tile(sim_game) # handle lambda safely if needed
            
            # 1. Selection
            search_path = [node]
            while node.is_expanded():
                action, node = self.select_child(node)
                sim_game.move(action) # Applies move AND random spawn
                search_path.append(node)
                
            # 2. Evaluation / Expansion
            if not sim_game.check_game_over():
                state_tensor = self.encode_state(sim_game.grid)
                device = next(self.model.parameters()).device
                state_tensor = state_tensor.to(device)
                with torch.no_grad():
                    policy_logits, value = self.model(state_tensor)
                
                value = value.item()
                valid_moves = sim_game.get_valid_moves()
                
                # Mask invalid moves
                policy = F.softmax(policy_logits[0], dim=0).cpu().numpy()
                action_probs = {m: policy[m] for m in valid_moves}
                sum_probs = sum(action_probs.values())
                if sum_probs > 0:
                    action_probs = {m: p / sum_probs for m, p in action_probs.items()}
                else:
                    action_probs = {m: 1.0 / len(valid_moves) for m in valid_moves}
                    
                node.expand(action_probs)
            else:
                # Terminal node
                value = -1.0 # Penalize for hitting terminal state (game over)
                
            # 3. Backpropagation
            self.backpropagate(search_path, value)

        # Return action probabilities based on visit counts at root
        action_visits = {}
        for action, child in root.children.items():
            action_visits[action] = child.visit_count
            
        sum_visits = sum(action_visits.values())
        if sum_visits == 0:
            return {m: 1.0/len(valid_moves) for m in valid_moves}
            
        return {m: v / sum_visits for m, v in action_visits.items()}

    def select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            # PUCT logic
            u = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            # 2048 is single-player, so no need to negate child.value like in two-player games
            score = child.value + u 
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_action, best_child

    def backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            
def _spawn_tile(game):
    # Hack to allow deepcopy to work with spawn_tile method
    import random
    empty_cells = list(zip(*np.where(game.grid == 0)))
    if empty_cells:
        row, col = random.choice(empty_cells)
        game.grid[row, col] = 2 if random.random() < 0.9 else 4
