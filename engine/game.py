"""
Core game engine for 2048 logic.
Manages the grid state, tile spawning, move execution, and score tracking.
Implements fast 1D list operations for performance.
"""
import random
import numpy as np

# Direction constants
LEFT = 0
UP = 1
RIGHT = 2
DOWN = 3

_ROW_CACHE = {}

class GameEngine:
    def __init__(self, seed=None, skip_init=False):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Internally use a flat 1D Python list for maximum speed
        self._flat_grid = [0] * 16
        self.score = 0
        self.moves = 0
        self.game_over = False
        
        if not skip_init:
            # Start with two tiles
            self.spawn_tile()
            self.spawn_tile()

    def clone(self):
        sim = GameEngine(skip_init=True)
        sim._flat_grid = self._flat_grid[:]
        sim.score = self.score
        sim.moves = self.moves
        sim.game_over = self.game_over
        return sim
        
    @property
    def grid(self):
        # Expose a 2D numpy array for compatibility with the rest of the codebase
        return np.array(self._flat_grid).reshape((4, 4))
        
    @grid.setter
    def grid(self, value):
        # Accept a 2D numpy array and flatten it internally
        self._flat_grid = value.flatten().tolist()

    def spawn_tile(self):
        empty_indices = [i for i, val in enumerate(self._flat_grid) if val == 0]
        if empty_indices:
            idx = random.choice(empty_indices)
            self._flat_grid[idx] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = [i for i in row if i != 0]
        new_row += [0] * (4 - len(new_row))
        return new_row

    def merge(self, row):
        # We assume row is already compressed (all zeros at the end)
        score_increase = 0
        for i in range(3):
            if row[i] != 0 and row[i] == row[i+1]:
                row[i] *= 2
                score_increase += row[i]
                row[i+1] = 0
        return row, score_increase
        
    def _shift_row(self, row):
        """Helper to compress, merge, and compress again."""
        t_row = tuple(row)
        if t_row in _ROW_CACHE:
            return _ROW_CACHE[t_row]
        compressed = self.compress(row)
        merged, increase = self.merge(compressed)
        final = self.compress(merged)
        _ROW_CACHE[t_row] = (final, increase)
        return final, increase

    def move(self, direction):
        if self.game_over:
            return False
            
        moved = False
        score_increase = 0
        grid = self._flat_grid
        
        if direction == LEFT:
            for i in range(4):
                start = i * 4
                row = grid[start:start+4]
                final_row, inc = self._shift_row(row)
                if row != final_row:
                    grid[start:start+4] = final_row
                    score_increase += inc
                    moved = True
                    
        elif direction == RIGHT:
            for i in range(4):
                start = i * 4
                row = grid[start:start+4][::-1]
                final_row, inc = self._shift_row(row)
                if row != final_row:
                    grid[start:start+4] = final_row[::-1]
                    score_increase += inc
                    moved = True
                    
        elif direction == UP:
            for j in range(4):
                row = [grid[j], grid[4+j], grid[8+j], grid[12+j]]
                final_row, inc = self._shift_row(row)
                if row != final_row:
                    grid[j], grid[4+j], grid[8+j], grid[12+j] = final_row
                    score_increase += inc
                    moved = True
                    
        elif direction == DOWN:
            for j in range(4):
                row = [grid[12+j], grid[8+j], grid[4+j], grid[j]]
                final_row, inc = self._shift_row(row)
                if row != final_row:
                    grid[12+j], grid[8+j], grid[4+j], grid[j] = final_row
                    score_increase += inc
                    moved = True
                    
        if moved:
            self.score += score_increase
            self.moves += 1
            self.spawn_tile()
            
            # Fast game over check without allocating new arrays
            if not self._has_valid_moves():
                self.game_over = True
            
        return moved
        
    def _has_valid_moves(self):
        """Fast check to see if any move is possible."""
        grid = self._flat_grid
        if 0 in grid:
            return True
            
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                val = grid[idx]
                if j < 3 and val == grid[idx + 1]:
                    return True
                if i < 3 and val == grid[idx + 4]:
                    return True
        return False

    def get_valid_moves(self):
        valid_moves = []
        grid = self._flat_grid
        
        # Unrolled validity check for max performance
        # Check LEFT
        moved = False
        for i in range(4):
            start = i * 4
            row = grid[start:start+4]
            final_row, _ = self._shift_row(row[:])
            if row != final_row:
                moved = True
                break
        if moved: valid_moves.append(LEFT)
            
        # Check RIGHT
        moved = False
        for i in range(4):
            start = i * 4
            row = grid[start:start+4][::-1]
            final_row, _ = self._shift_row(row[:])
            if row != final_row:
                moved = True
                break
        if moved: valid_moves.append(RIGHT)
            
        # Check UP
        moved = False
        for j in range(4):
            row = [grid[j], grid[4+j], grid[8+j], grid[12+j]]
            final_row, _ = self._shift_row(row[:])
            if row != final_row:
                moved = True
                break
        if moved: valid_moves.append(UP)
            
        # Check DOWN
        moved = False
        for j in range(4):
            row = [grid[12+j], grid[8+j], grid[4+j], grid[j]]
            final_row, _ = self._shift_row(row[:])
            if row != final_row:
                moved = True
                break
        if moved: valid_moves.append(DOWN)
            
        return valid_moves

    def check_game_over(self):
        if not self._has_valid_moves():
            self.game_over = True
        return self.game_over

    def get_state(self):
        return self.grid.copy()