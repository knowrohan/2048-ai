import random
import numpy as np

class GameEngine:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.grid = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.game_over = False
        
        # Start with two tiles
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        empty_cells = list(zip(*np.where(self.grid == 0)))
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.grid[row, col] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        new_row = [i for i in row if i != 0]
        new_row += [0] * (4 - len(new_row))
        return new_row

    def merge(self, row):
        new_row = row.copy()
        score_increase = 0
        for i in range(3):
            if new_row[i] != 0 and new_row[i] == new_row[i+1]:
                new_row[i] *= 2
                score_increase += new_row[i]
                new_row[i+1] = 0
        return new_row, score_increase

    def move_left(self):
        moved = False
        score_increase = 0
        for i in range(4):
            original_row = self.grid[i].copy()
            compressed_row = self.compress(original_row)
            merged_row, row_score = self.merge(compressed_row)
            final_row = self.compress(merged_row)
            
            self.grid[i] = final_row
            score_increase += row_score
            
            if not np.array_equal(original_row, final_row):
                moved = True
                
        return moved, score_increase

    def move(self, direction):
        """
        0: Left, 1: Up, 2: Right, 3: Down
        """
        if self.game_over:
            return False
            
        rotations = direction
        self.grid = np.rot90(self.grid, rotations)
        moved, score_increase = self.move_left()
        self.grid = np.rot90(self.grid, -rotations)
        
        if moved:
            self.score += score_increase
            self.spawn_tile()
            self.check_game_over()
            
        return moved

    def get_valid_moves(self):
        valid_moves = []
        for direction in range(4):
            # Test move efficiently by copying the grid
            temp_grid = self.grid.copy()
            
            rotations = direction
            self.grid = np.rot90(self.grid, rotations)
            
            moved = False
            for i in range(4):
                original_row = self.grid[i].copy()
                compressed_row = self.compress(original_row)
                merged_row, _ = self.merge(compressed_row)
                final_row = self.compress(merged_row)
                if not np.array_equal(original_row, final_row):
                    moved = True
                    break
                    
            if moved:
                valid_moves.append(direction)
                
            self.grid = temp_grid
            
        return valid_moves

    def check_game_over(self):
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
        return self.game_over

    def get_state(self):
        return self.grid.copy()
