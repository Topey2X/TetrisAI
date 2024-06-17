import pygame
import sys
from tetris import Tetris

# Initialize Pygame
pygame.init()

# Define constants
CELL_SIZE = 30
BOARD_WIDTH = Tetris.BOARD_WIDTH * CELL_SIZE
BOARD_HEIGHT = Tetris.BOARD_HEIGHT * CELL_SIZE
SCREEN_WIDTH = BOARD_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT + 100  # Extra space for score display

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Initialize the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Tetris")

# Initialize font
font = pygame.font.Font(None, 36)

# Create an instance of the Tetris game
game = Tetris()

def draw_board(board):
    for y in range(Tetris.BOARD_HEIGHT):
        for x in range(Tetris.BOARD_WIDTH):
            cell_value = board[y][x][0]
            cell_color = Tetris.COLORS[cell_value]
            pygame.draw.rect(screen, cell_color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, WHITE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def draw_text(text, position, color=WHITE):
    label = font.render(text, 1, color)
    screen.blit(label, position)

def main():
    clock = pygame.time.Clock()
    running = True

    while running:
        screen.fill(BLACK)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    game.move_left()
                elif event.key == pygame.K_RIGHT:
                    game.move_right()
                elif event.key == pygame.K_DOWN:
                    game.move_down()
                elif event.key == pygame.K_UP:
                    game.rotate()

        # Update game state
        game.update()

        # Draw the current state of the board
        draw_board(game.board)

        # Draw score
        draw_text(f"Score: {game.score}", (10, BOARD_HEIGHT + 10))

        # Refresh the screen
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
