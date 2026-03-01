"""
Pygame-based graphical user interface for the 2048 game.
Handles rendering the game board, colors, tiles, and listening to user inputs.
Provides both human play and AI play (watch) visualization modes.
References `ai.mcts.MCTS` when AI plays.
"""
import pygame
import sys
import numpy as np
import time

# Colors
BACKGROUND_COLOR = (187, 173, 160)
GRID_COLOR = (205, 193, 180)
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (57, 55, 47),
    16384: (54, 52, 44),
    32768: (51, 49, 41),
}

TEXT_COLORS = {
    2: (119, 110, 101),
    4: (119, 110, 101),
    'default': (249, 246, 242)
}

class GameUI:
    def __init__(self, game_engine, width=400, height=500):
        pygame.init()
        self.engine = game_engine
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('2048 AI v1.0')
        self.font = pygame.font.SysFont('arial', 40, bold=True)
        self.score_font = pygame.font.SysFont('arial', 24, bold=True)
        
        self.tile_size = self.width // 4
        self.padding = 10

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw score
        score_text = self.score_font.render(f'Score: {self.engine.score}', True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))
        
        # Draw grid frame
        board_rect = pygame.Rect(0, 100, self.width, self.width)
        pygame.draw.rect(self.screen, BACKGROUND_COLOR, board_rect)
        
        for i in range(4):
            for j in range(4):
                val = self.engine.grid[i][j]
                
                rect = pygame.Rect(
                    j * self.tile_size + self.padding,
                    100 + i * self.tile_size + self.padding,
                    self.tile_size - 2 * self.padding,
                    self.tile_size - 2 * self.padding
                )
                
                color = TILE_COLORS.get(val, TILE_COLORS[2048])
                pygame.draw.rect(self.screen, color, rect, border_radius=5)
                
                if val > 0:
                    text_color = TEXT_COLORS.get(val, TEXT_COLORS['default'])
                    text_surface = self.font.render(str(val), True, text_color)
                    text_rect = text_surface.get_rect(center=rect.center)
                    self.screen.blit(text_surface, text_rect)
                    
        pygame.display.flip()

    def play_human(self):
        clock = pygame.time.Clock()
        while not self.engine.game_over:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.engine.move(0)
                    elif event.key == pygame.K_UP:
                        self.engine.move(1)
                    elif event.key == pygame.K_RIGHT:
                        self.engine.move(2)
                    elif event.key == pygame.K_DOWN:
                        self.engine.move(3)
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        return
            clock.tick(30)
            
        highest_tile = np.max(self.engine.grid)
        print("\n--- Game Over ---")
        print(f"Total Moves: {getattr(self.engine, 'moves', 0)}")
        print(f"Final Score: {self.engine.score}")
        print(f"Highest Tile: {highest_tile}")
        print("\nGame frozen. Close the window to exit.")

        while True:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return
            pygame.time.Clock().tick(30)

    def play_ai(self, model, c_puct=1.5, num_simulations=400, delay=0.3):
        from ai.mcts import MCTS
        import torch
        import torch.nn.functional as F
        device = next(model.parameters()).device
        mcts = MCTS(c_puct=c_puct)
        clock = pygame.time.Clock()
        
        while not self.engine.game_over:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return
            
            # AI Move choice
            root = mcts.search_root(self.engine, model, device)
            
            for _ in range(num_simulations):
                search_path, leaf_game = mcts.search_leaf(self.engine, root)
                if leaf_game.game_over:
                    mcts.backpropagate_leaf(search_path, leaf_game, -1.0, None)
                else:
                    state_tensor = torch.from_numpy(MCTS.encode_state(leaf_game._flat_grid)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        p_logits, v_out = model(state_tensor)
                    p_out = F.softmax(p_logits, dim=1).cpu().numpy()[0]
                    v_out = v_out.cpu().numpy()[0][0]
                    mcts.backpropagate_leaf(search_path, leaf_game, v_out, p_out)
            
            action_probs = mcts.get_action_prob(root, temperature=0.0)
            
            if action_probs:
                # Deterministic move selection for evaluation
                best_action = max(action_probs, key=action_probs.get)
                self.engine.move(best_action)
            else:
                break
                
            time.sleep(delay)
            clock.tick(30)
            
        highest_tile = np.max(self.engine.grid)
        print("\n--- Game Over ---")
        print(f"Total Moves: {getattr(self.engine, 'moves', 0)}")
        print(f"Final Score: {self.engine.score}")
        print(f"Highest Tile: {highest_tile}")
        print("\nGame frozen. Close the window to exit.")

        while True:
            self.draw()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    pygame.quit()
                    return
            pygame.time.Clock().tick(30)
