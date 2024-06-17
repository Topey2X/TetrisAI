import pygame
import numpy as np
from tetris import Tetris
from agent import DQNAgent


class TetrisAI:
    # Amusingly, this AI will only give up if it has absolutely no moves left
    def __init__(self, tetris):
        n_neurons = [32, 32]
        activations = ["relu", "relu", "linear"]
        weights_file = "checkpoints\_exceptional_1212-2723457.weights.h5" # HARD MODE
        # weights_file = "Best Models\\134282\\17_model.weights.h5" # EASY MODE

        self.tetris: Tetris = tetris
        self.current_state = self.tetris.reset()
        self.agent = DQNAgent(
            self.tetris.get_state_size(),
            n_neurons=n_neurons,
            activations=activations,
            epsilon=0,
            train=False,
        )
        self.agent.load(weights_file)

    def do_move(self):
        next_states = self.tetris.get_next_states()
        if len(next_states) == 0:
            return False
        
        best_state = self.agent.best_state(next_states.values())
        for action, state in next_states.items():
            if state == best_state:
                move = action
                break

        reward, done = self.tetris.play(move[0], move[1])
        self.agent.add_to_memory(self.current_state, next_states[move], reward, done)
        self.current_state = next_states[move]
        return done


class HumanVsTetris(Tetris):
    FPS = 30
    MAX_SPEED = 1 # seconds per move
    MIN_SPEED = 5 # seconds per move
    MAX_SPEED_STEPS = 300

    def __init__(self):
        self.AI_game = Tetris()
        self.setup_pygame()
        super().__init__()

    def setup_pygame(self):
        pygame.init()

        self.bar_height = int(0.5 * Tetris.RENDER_SCALE)
        self.title_height = 30
        self.scoreboard_height = 50
        self.game_width = Tetris.BOARD_WIDTH * Tetris.RENDER_SCALE
        self.game_height = Tetris.BOARD_HEIGHT * Tetris.RENDER_SCALE
        self.game_offset = 10
        self.screen_width = (self.game_width * 2) + (self.game_offset * 3)
        self.screen_height = (
            self.game_height
            + self.bar_height
            + self.title_height
            + self.scoreboard_height
        )
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)

        pygame.display.set_caption("TetrisAI vs Humanity")
        self.clock = pygame.time.Clock()
        self.main_surface = pygame.Surface((self.screen_width, self.screen_height))
        self.game_surface = pygame.Surface((self.game_width, self.game_height))
        self.ai_surface = pygame.Surface((self.game_width, self.game_height))
        
    def reset(self):
        new_seed = self.get_new_seed()
        self.seed = new_seed
        self.AI_game.seed = new_seed
        super().reset()
        self.AI_game.reset()
        self.AI = TetrisAI(self.AI_game)
        self.steps = 0
        self.reset_piece_timer()
        self.hover_x = 3

    def reset_piece_timer(self):
        # Take 300 steps to get to max speed
        self.time_per_piece = self.FPS * ((self.MAX_SPEED_STEPS-self.steps)/self.MAX_SPEED_STEPS) * (self.MIN_SPEED - self.MAX_SPEED) + self.MAX_SPEED
        self.piece_timer = 0  # 30 fps

    def handle_input(self):
        for event in pygame.event.get():
            match event.type:
                case pygame.QUIT:
                    pygame.quit()
                    exit()
                case pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self._rotate(90)
                        if self._check_collision(
                            self._get_rotated_piece(),
                            [self.current_pos[0], self.current_pos[1]],
                        ):
                            self._rotate(-90)
                case pygame.MOUSEBUTTONDOWN:
                    if (event.button == 1) and (self.piece_timer > 0.15*self.FPS):
                        # Prevents the player from adding a second piece too quickly
                        self.play_piece()
                case pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((max(self.screen_width, event.w), max(self.screen_height, event.h)), pygame.RESIZABLE)

        mouse_x, _ = pygame.mouse.get_pos()
        window_width, window_height = self.screen.get_size()
        main_surface_rect = self.main_surface.get_rect(center=(window_width // 2, window_height // 2))
        mouse_x -= main_surface_rect.topleft[0]
        
        if mouse_x >= self.game_width or mouse_x < 0:
            return

        # Clamp hover_x to valid board width range
        piece = self._get_rotated_piece()
        piece_x = [p[0] for p in piece]
        min_x = min(piece_x)
        max_x = max(piece_x)
        valid_pos_size = self.game_width / (
            Tetris.BOARD_WIDTH - (max_x - min_x)
        )  # % of the board that each position represents
        mouse_grid_x = mouse_x // valid_pos_size  # position of the mouse in the board
        self.hover_x = int(mouse_grid_x - min_x)

    def render(self):
        self.screen.fill((50, 50, 50))  # Clear the screen
        self.main_surface.fill((50, 50, 50))  # Clear the board surface
        self.game_surface.fill("black")  # Clear the board surface
        self.ai_surface.fill("black")  # Clear the board surface

        time_left = np.clip((self.time_per_piece - self.piece_timer) / (self.time_per_piece), 0, 1)
        bar_width = time_left * self.screen_width
        if time_left > 0.6:
            color = "green"
        elif time_left > 0.4:
            color = "yellow"
        elif time_left > 0.2:
            color = "orange"
        else:
            color = "red"
        pygame.draw.rect(
            self.main_surface, color, pygame.Rect(0, 0, bar_width, self.bar_height)
        )

        # Draw the Human board
        for y in range(Tetris.BOARD_HEIGHT):
            for x in range(Tetris.BOARD_WIDTH):
                cell = self.board[y][x]
                if cell[0] != Tetris.MAP_EMPTY[0]:
                    color = cell[1]
                    pygame.draw.rect(
                        self.game_surface,
                        color,
                        pygame.Rect(
                            x * Tetris.RENDER_SCALE + 1,
                            y * Tetris.RENDER_SCALE + 1,
                            Tetris.RENDER_SCALE - 2,
                            Tetris.RENDER_SCALE - 2,
                        ),
                    )

        if self.hover_x is not None:
            piece = self._get_rotated_piece()
            ghost_y = self.get_ghost_y(piece)
            for x, y in piece:
                ghost_x = x + self.hover_x
                y += ghost_y
                color = tuple(c // 2 for c in Tetris.COLORS[self.current_piece + 1])
                pygame.draw.rect(
                    self.game_surface,
                    color,
                    pygame.Rect(
                        ghost_x * Tetris.RENDER_SCALE + 1,
                        y * Tetris.RENDER_SCALE + 1,
                        Tetris.RENDER_SCALE - 2,
                        Tetris.RENDER_SCALE - 2,
                    ),
                )

        # Draw the AI board
        for y in range(Tetris.BOARD_HEIGHT):
            for x in range(Tetris.BOARD_WIDTH):
                cell = self.AI_game.board[y][x]
                if cell[0] != Tetris.MAP_EMPTY[0]:
                    color = cell[1]
                    pygame.draw.rect(
                        self.ai_surface,
                        color,
                        pygame.Rect(
                            x * Tetris.RENDER_SCALE + 1,
                            y * Tetris.RENDER_SCALE + 1,
                            Tetris.RENDER_SCALE - 2,
                            Tetris.RENDER_SCALE - 2,
                        ),
                    )
        self.main_surface.blits(
            [
                (
                    self.game_surface,
                    (self.game_offset, self.bar_height + self.title_height),
                ),
                (
                    self.ai_surface,
                    (
                        self.game_width + self.game_offset + self.game_offset,
                        self.bar_height + self.title_height,
                    ),
                ),
            ]
        )

        # Additional text and trackers
        font = pygame.font.SysFont(None, 32)

        # Human title
        human_title_surface = font.render("Human", True, "white")
        
        # AI title
        ai_title_surface = font.render("AI", True, "white")
        
        # Human score and lines cleared
        font = pygame.font.SysFont(None, 24)
        human_score_surface = font.render(f"Score: {self.score}", True, "white")
        human_lines_surface = font.render(
            f"Lines Cleared: {self.clearedLines}", True, "white"
        )
        # AI score and lines cleared
        ai_score_surface = font.render(f"Score: {self.AI_game.score}", True, "white")
        ai_lines_surface = font.render(
            f"Lines Cleared: {self.AI_game.clearedLines}", True, "white"
        )
        self.main_surface.blits(
            [
                ( # Human title
                    human_title_surface,
                    (
                        self.game_width // 2 - human_title_surface.get_width() // 2,
                        self.bar_height + 5,
                    ),
                ),
                ( # AI title
                    ai_title_surface,
                    (
                        self.game_width
                        + self.game_offset
                        + self.game_width // 2
                        - ai_title_surface.get_width() // 2,
                        self.bar_height + 5,
                    ),
                ),
                ( # Human Score and Lines Cleared
                    human_score_surface, (10, self.game_height + self.bar_height + self.title_height + 5)
                ),
                (
                    human_lines_surface, (10, self.game_height + self.bar_height + self.title_height + 30)
                ),
                ( # AI Score and Lines Cleared
                    ai_score_surface,
                    (
                        self.game_width + self.game_offset + 10,
                        self.game_height + self.bar_height + self.title_height + 5,
                    ),
                ),
                (
                    ai_lines_surface,
                    (
                        self.game_width + self.game_offset + 10,
                        self.game_height + self.bar_height + self.title_height + 30,
                    ),
                ),
            ]
        )

        window_width, window_height = self.screen.get_size()
        main_surface_rect = self.main_surface.get_rect(center=(window_width // 2, window_height // 2))
        self.screen.blit(self.main_surface, main_surface_rect.topleft)

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
        self.AI.do_move()
        self.reset_piece_timer()
        self.steps += 1
        self.hover_x = 3
        
    def show_game_over_screen(self):
        self.screen.fill((50, 50, 50))
        self.main_surface.fill((50, 50, 50))
        font = pygame.font.SysFont(None, 72)
        if self.AI_game.game_over and self.game_over:
            text_surface = font.render("It's a Tie!", True, "white")
        elif self.AI_game.game_over:
            text_surface = font.render("Human Wins!", True, "white")
        else:
            text_surface = font.render("AI Wins!", True, "white")
            
        
        text_rect = text_surface.get_rect(center=(self.screen_width // 2, self.screen_height // 2 - 100))
        self.main_surface.blit(text_surface, text_rect)
        
        font_small = pygame.font.SysFont(None, 36)
        reset_text = font_small.render("Press [SPACE] to Restart", True, "white")
        reset_rect = reset_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 100))
        self.main_surface.blit(reset_text, reset_rect)
        
        # Display scores
        human_score = self.score
        ai_score = self.AI_game.score
        if human_score > ai_score:
            human_score_color = "green"
            ai_score_color = "white"
        elif ai_score > human_score:
            human_score_color = "white"
            ai_score_color = "green"
        else:  # tie
            human_score_color = ai_score_color = "white"

        human_score_text = font_small.render(f'Human Score: {human_score}', True, human_score_color)
        ai_score_text = font_small.render(f'AI Score: {ai_score}', True, ai_score_color)

        human_score_rect = human_score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        ai_score_rect = ai_score_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2 + 50))

        self.main_surface.blit(human_score_text, human_score_rect)
        self.main_surface.blit(ai_score_text, ai_score_rect)
        
        window_width, window_height = self.screen.get_size()
        main_surface_rect = self.main_surface.get_rect(center=(window_width // 2, window_height // 2))
        self.screen.blit(self.main_surface, main_surface_rect.topleft)
        
        pygame.display.flip()
        
        # Wait for user input to restart or quit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.reset()
                        waiting = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        exit()

    def run(self):
        while True:
            while not (self.game_over or self.AI_game.game_over):
                self.handle_input()
                self.piece_timer += 1
                if self.piece_timer >= self.time_per_piece:
                    self.play_piece()
                self.render()
                self.clock.tick(self.FPS)
            self.show_game_over_screen()


if __name__ == "__main__":
    game = HumanVsTetris()
    game.run()
