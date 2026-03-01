"""
Self-play simulation loop for generating training data.
Handles concurrent batched game playing, MCTS simulations, data augmentation 
(rotations and reflections), and terminal value calculation.
References `ai.mcts.MCTS` and `engine.game.GameEngine`.
"""
import torch
import torch.nn.functional as F
import numpy as np
import math
from ai.mcts import MCTS
from engine.game import GameEngine

# Action remapping tables for 8-fold symmetry augmentation.
# Directions: 0=Left, 1=Up, 2=Right, 3=Down
# rot90 ccw:  Left->Down, Up->Left, Right->Up, Down->Right  => [3, 0, 1, 2]
# flip_lr:    Left->Right, Up->Up, Right->Left, Down->Down   => [2, 1, 0, 3]
ROT90_ACTION = [3, 0, 1, 2]
FLIP_LR_ACTION = [2, 1, 0, 3]

def apply_rotation(grid, policy, k):
    """Rotate grid k times (90° CCW each) and remap policy accordingly."""
    rotated_grid = np.rot90(grid, k)
    rotated_policy = policy.copy()
    for _ in range(k):
        rotated_policy = [rotated_policy[ROT90_ACTION[i]] for i in range(4)]
    return rotated_grid, rotated_policy

def apply_flip(grid, policy):
    """Flip grid left-right and remap policy accordingly."""
    flipped_grid = np.fliplr(grid)
    flipped_policy = [policy[FLIP_LR_ACTION[i]] for i in range(4)]
    return flipped_grid, flipped_policy

def augment_data(grids, policies):
    """Apply 8-fold symmetry augmentation (4 rotations x 2 reflections)."""
    aug_grids = []
    aug_policies = []
    for grid, policy in zip(grids, policies):
        for k in range(4):
            rg, rp = apply_rotation(grid, policy, k)
            aug_grids.append(rg.copy())
            aug_policies.append(rp)
            fg, fp = apply_flip(rg, rp)
            aug_grids.append(fg.copy())
            aug_policies.append(fp)
    return aug_grids, aug_policies

def evaluate_board(grid):
    """
    Returns a heuristic score representing the board structure, from 0.0 to 1.0.
    Values clean boards (monotonic patterns, heavy tiles in corners).
    """
    monotonic_left = 0
    monotonic_right = 0
    monotonic_up = 0
    monotonic_down = 0

    for i in range(4):
        for j in range(3):
            val1 = grid[i, j]
            val2 = grid[i, j + 1]
            if val1 != 0 and val2 != 0:
                if val1 >= val2: monotonic_left += 1
                if val1 <= val2: monotonic_right += 1
            
            val3 = grid[j, i]
            val4 = grid[j + 1, i]
            if val3 != 0 and val4 != 0:
                if val3 >= val4: monotonic_up += 1
                if val3 <= val4: monotonic_down += 1
                
    max_monotonicity = max(monotonic_left, monotonic_right) + max(monotonic_up, monotonic_down)
    # Perfect score is 24 (12 horizontal pairs + 12 vertical pairs aligned)
    monotonic_score = max_monotonicity / 24.0
    
    # Empty cells logic (bonus for 4+ empty cells)
    empty_cells = np.count_nonzero(grid == 0)
    empty_score = min(1.0, empty_cells / 8.0) 
    
    return (monotonic_score + empty_score) / 2.0


def play_games_concurrently(model, target_games, num_concurrent=256, num_simulations=200, c_puct=1.5, temperature_moves=30):
    device = next(model.parameters()).device
    mcts = MCTS(c_puct=c_puct)
    
    # Initialize the concurrent track
    active_games = [GameEngine() for _ in range(num_concurrent)]
    roots = [mcts.search_root(game, model, device) for game in active_games]
    for root in roots:
        mcts.apply_dirichlet_noise(root)
        
    histories = [{"raw_grids": [], "states": [], "policies": [], "moves": 0} for _ in range(num_concurrent)]
    
    completed_games = 0
    
    all_final_states = []
    all_final_policies = []
    all_final_values = []
    
    # We loop until we have finished exactly `target_games`
    while completed_games < target_games:
        
        # MCTS simulation loops for all active games
        for _ in range(num_simulations):
            search_paths = []
            leaf_games = []
            eval_states = []
            eval_indices = []
            
            # Phase 1: Selection to a leaf for each game
            for i, (game, root) in enumerate(zip(active_games, roots)):
                path, leaf_game = mcts.search_leaf(game, root)
                search_paths.append(path)
                leaf_games.append(leaf_game)
                
                if leaf_game.game_over:
                    eval_states.append(None)
                else:
                    eval_states.append(MCTS.encode_state(leaf_game._flat_grid))
                    eval_indices.append(i)
                    
            # Phase 2: Massive GPU evaluation
            values_batch = np.zeros(num_concurrent, dtype=np.float32)
            policies_batch = [None] * num_concurrent
            
            if eval_indices:
                tensors = [eval_states[idx] for idx in eval_indices]
                batch_tensor = torch.from_numpy(np.stack(tensors)).to(device)
                
                with torch.no_grad():
                    p_logits, v_out = model(batch_tensor)
                    
                p_out = F.softmax(p_logits, dim=1).cpu().numpy()
                v_out = v_out.cpu().numpy()
                
                for idx_in_batch, original_idx in enumerate(eval_indices):
                    values_batch[original_idx] = v_out[idx_in_batch][0]
                    policies_batch[original_idx] = p_out[idx_in_batch]
                    
            # Special case for finished games at the leaf
            for i in range(num_concurrent):
                if eval_states[i] is None:
                    values_batch[i] = -1.0
                    
            # Phase 3: Backpropagation distribution
            for i in range(num_concurrent):
                mcts.backpropagate_leaf(search_paths[i], leaf_games[i], values_batch[i], policies_batch[i])

        # After MCTS simulations, pick a move and advance all games
        for i in range(num_concurrent):
            game = active_games[i]
            root = roots[i]
            history = histories[i]
            
            temp = 1.0 if history["moves"] < temperature_moves else 0.0
            action_probs = mcts.get_action_prob(root, temperature=temp)
            
            if not action_probs:
                # Should not happen unless forced game over internally, but safeguard
                game.game_over = True
                continue
                
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            
            policy = [0.0] * 4
            for a, p in action_probs.items():
                policy[a] = p
                
            history["raw_grids"].append(game.grid.copy())
            history["states"].append(torch.from_numpy(MCTS.encode_state(game._flat_grid)))
            history["policies"].append(policy)
            
            if temp > 0:
                action = np.random.choice(actions, p=probs)
            else:
                action = max(action_probs, key=action_probs.get)
                
            game.move(action)
            history["moves"] += 1
            
            if not game.game_over:
                # Discard tree after every real move and re-root from exact new state due to stochasticity
                roots[i] = mcts.search_root(game, model, device)
                mcts.apply_dirichlet_noise(roots[i])
                
        # Handle finished games
        i = 0
        while i < num_concurrent:
            game = active_games[i]
            if game.game_over:
                max_tile = np.max(game.grid)
                moves = histories[i]["moves"]
                completed_games += 1
                
                if completed_games % 50 == 0 or completed_games == 1:
                    print(f"Finished Game {completed_games}/{target_games} - Final Score: {game.score} - Max Tile: {max_tile} - Total Moves: {moves}")

                # Calculate specific terminal values
                SCORE_TARGET = 150000.0
                score_val = min(1.0, game.score / SCORE_TARGET)
                tile_val = (math.log2(max(2, max_tile)) / 16.0) 
                board_val = evaluate_board(game.grid)
                base_val = (0.5 * score_val + 0.3 * tile_val + 0.2 * board_val)
                terminal_value = max(-1.0, min(1.0, base_val * 2.0 - 1.0))
                
                # Apply discounting
                gamma = 0.99
                n_states = len(histories[i]["states"])
                values = [0.0] * n_states
                if n_states > 0:
                    values[-1] = terminal_value
                    for t in range(n_states - 2, -1, -1):
                        values[t] = gamma * values[t + 1]
                        
                aug_grids, aug_policies = augment_data(histories[i]["raw_grids"], histories[i]["policies"])
                
                aug_states = []
                for g in aug_grids:
                    aug_states.append(torch.from_numpy(MCTS.encode_state(g)))
                    
                aug_values = []
                for v in values:
                    aug_values.extend([v] * 8)
                    
                all_final_states.extend(aug_states)
                all_final_policies.extend(aug_policies)
                all_final_values.extend(aug_values)
                
                # Instantiate a new game in this slot to keep batch size constant, unless we are done
                if completed_games + (num_concurrent - 1 - i) < target_games:
                    # We still need another game in this slot
                    active_games[i] = GameEngine()
                    roots[i] = mcts.search_root(active_games[i], model, device)
                    mcts.apply_dirichlet_noise(roots[i])
                    histories[i] = {"raw_grids": [], "states": [], "policies": [], "moves": 0}
                    i += 1
                else:
                    # We do not need any more games, pop this slot completely
                    active_games.pop(i)
                    roots.pop(i)
                    histories.pop(i)
                    num_concurrent -= 1
                    # Do not increment i because elements shift left
            else:
                i += 1
                
    return all_final_states, all_final_policies, all_final_values