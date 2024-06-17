import pygame
import numpy as np
from PIL import Image
from tetris import Tetris

class HumanTetris(Tetris):
    def __init__(self, record=False, time_per_piece=5):
        super().__init__(record)
        self.setup_pygame()
        self.time_per_piece = time_per_piece
        self.reset_piece_timer()

    def setup_pygame(self):
        pygame.init()
        self.bar_height = int(0.5 * Tetris.RENDER_SCALE)
        self.game_width = Tetris.BOARD_WIDTH * Tetris.RENDER_SCALE
        self.game_height = Tetris.BOARD_HEIGHT * Tetris.RENDER_SCALE
        self.screen = pygame.display.set_mode((self.game_width, self.game_height + self.bar_height))
        pygame.display.set_caption('Tetris')
        self.clock = pygame.time.Clock()

    def reset_piece_timer(self):
        self.piece_timer = self.time_per_piece * 30  # 30 fps

    def handle_input(self):
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    pygame.quit()
                    exit()
                case pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self._rotate(90)
                        if self._check_collision(self._get_rotated_piece(), [self.current_pos[0], self.current_pos[1]]):
                            self._rotate(-90)
                case pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.play_piece()
                            
        mouse_x, _ = pygame.mouse.get_pos()

        # Clamp hover_x to valid board width range
        piece = self._get_rotated_piece()
        piece_x = [p[0] for p in piece]
        min_x = min(piece_x)
        max_x = max(piece_x)
        valid_pos_size = self.game_width / (Tetris.BOARD_WIDTH - (max_x - min_x)) # % of the board that each position represents
        mouse_grid_x = mouse_x // valid_pos_size # position of the mouse in the board
        self.hover_x = int(mouse_grid_x - min_x)

    def render(self):
        self.screen.fill((0, 0, 0))  # Clear the screen
        
        bar_width = np.clip(self.piece_timer / (self.time_per_piece * 30), 0, 1) * self.game_width
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, self.game_height, bar_width, self.bar_height))

        for y in range(Tetris.BOARD_HEIGHT):
            for x in range(Tetris.BOARD_WIDTH):
                cell = self.board[y][x]
                if cell[0] != Tetris.MAP_EMPTY[0]:
                    color = cell[1]
                    pygame.draw.rect(self.screen, color, pygame.Rect(x * Tetris.RENDER_SCALE + 1, y * Tetris.RENDER_SCALE + 1, Tetris.RENDER_SCALE - 2, Tetris.RENDER_SCALE - 2))
        
        piece = self._get_rotated_piece()
        ghost_y = self.get_ghost_y(piece)
        for x, y in piece:
            ghost_x = x + self.hover_x
            y += ghost_y
            color = tuple(c // 2 for c in Tetris.COLORS[self.current_piece + 1])
            pygame.draw.rect(self.screen, color, pygame.Rect(ghost_x * Tetris.RENDER_SCALE + 1, y * Tetris.RENDER_SCALE + 1, Tetris.RENDER_SCALE - 2, Tetris.RENDER_SCALE - 2))

        pygame.display.flip()

    def get_ghost_y(self, piece):
        ghost_y = 0
        while not self._check_collision(piece, [self.hover_x, ghost_y]):
            ghost_y += 1
        return ghost_y - 1

    def play_piece(self):
        piece = self._get_rotated_piece()
        ghost_y = self.get_ghost_y(piece)
        self.current_pos = [self.hover_x, ghost_y]
        self.play(self.current_pos[0], self.current_rotation, render=False)
        self.reset_piece_timer()

    def run(self):
        while not self.game_over:
            self.handle_input()
            self.piece_timer -= 1
            if self.piece_timer <= 0:
                self.play_piece()
            self.render()
            self.clock.tick(30)

if __name__ == "__main__":
    game = HumanTetris(time_per_piece=5)  # Adjust the time per piece as needed
    game.run()
