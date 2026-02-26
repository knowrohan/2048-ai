import torch
import numpy as np
import math
from ai.mcts import MCTS
from engine.game import GameEngine

def play_game(model, num_simulations=50, c_puct=1.5, game_idx=None, total_games=None):
    game = GameEngine()
    mcts = MCTS(model, num_simulations, c_puct)
    
    states = []
    policies = []
    moves = 0
    
    while not game.game_over:
        action_probs = mcts.search(game)
        if action_probs is None:
            break
            
        policy = [0.0] * 4
        for a, p in action_probs.items():
            policy[a] = p
            
        # mcts.encode_state returns (1, 1, 4, 4), we want (1, 4, 4) for the buffer
        state_tensor = mcts.encode_state(game.grid).squeeze(0)
        states.append(state_tensor)
        policies.append(policy)
        
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        if sum(probs) == 0:
            break
            
        # Add temperature to exploration if needed, here we sample strictly historically
        action = np.random.choice(actions, p=probs)
        
        game.move(action)
        moves += 1
        
        if game_idx is not None and moves % 50 == 0:
            print(f"  Game {game_idx}/{total_games} - Move {moves} - Score: {game.score} - Max Tile: {np.max(game.grid)}")
        
    max_tile = np.max(game.grid)
    if game_idx is not None:
        print(f"Finished Game {game_idx}/{total_games} - Final Score: {game.score} - Max Tile: {max_tile} - Total Moves: {moves}")
    # Define value in [-1, 1]. A 2048 tile corresponds to a max_tile of 2048. log2(2048) = 11.
    value = (math.log2(max(2, max_tile)) / 11.0) * 2 - 1
    value = min(1.0, value)
    
    values = [value] * len(states)
    
    return states, policies, values
