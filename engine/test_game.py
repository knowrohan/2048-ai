import numpy as np
import pytest
from engine.game import GameEngine

def test_initialization():
    game = GameEngine()
    assert game.score == 0
    assert not game.game_over
    # Two tiles should be spawned
    assert np.count_nonzero(game.grid) == 2

def test_move_left():
    game = GameEngine()
    game.spawn_tile = lambda: None
    game.grid = np.array([
        [2, 2, 4, 8],
        [2, 0, 2, 0],
        [4, 4, 4, 4],
        [2, 4, 8, 16]
    ])
    moved = game.move(0) # Left
    assert moved
    assert np.array_equal(game.grid[0], [4, 4, 8, 0])
    assert np.array_equal(game.grid[1], [4, 0, 0, 0])
    assert np.array_equal(game.grid[2], [8, 8, 0, 0])
    assert np.array_equal(game.grid[3], [2, 4, 8, 16])
    # Total score should be 4 + 4 + 8 + 8 = 24
    assert game.score == 24

def test_move_right():
    game = GameEngine()
    game.spawn_tile = lambda: None
    game.grid = np.array([
        [2, 2, 4, 8],
        [2, 0, 2, 0],
        [4, 4, 4, 4],
        [2, 4, 8, 16]
    ])
    moved = game.move(2) # Right
    assert moved
    assert np.array_equal(game.grid[0], [0, 4, 4, 8])
    assert np.array_equal(game.grid[1], [0, 0, 0, 4])
    assert np.array_equal(game.grid[2], [0, 0, 8, 8])
    assert np.array_equal(game.grid[3], [2, 4, 8, 16])
    assert game.score == 24

def test_game_over():
    game = GameEngine()
    game.grid = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2]
    ])
    game.check_game_over()
    assert game.game_over

def test_valid_moves():
    game = GameEngine()
    game.grid = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 0] # One empty spot
    ])
    valid_moves = game.get_valid_moves()
    # Can move right and down
    assert set(valid_moves) == {2, 3}
