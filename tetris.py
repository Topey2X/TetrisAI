import random
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from time import sleep
import os

class Tetris:
    MAP_EMPTY = (0, (0, 0, 0))  # (state, color)
    MAP_BLOCK = 1
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    RENDER_SCALE = 30

    TETROMINOS = {
        0: {
            0: [(0,0), (1,0), (2,0), (3,0)],
            90: [(1,0), (1,1), (1,2), (1,3)],
            180: [(3,0), (2,0), (1,0), (0,0)],
            270: [(1,3), (1,2), (1,1), (1,0)],
        },
        1: {
            0: [(1,0), (0,1), (1,1), (2,1)],
            90: [(0,1), (1,2), (1,1), (1,0)],
            180: [(1,2), (2,1), (1,1), (0,1)],
            270: [(2,1), (1,0), (1,1), (1,2)],
        },
        2: {
            0: [(1,0), (1,1), (1,2), (2,2)],
            90: [(0,1), (1,1), (2,1), (2,0)],
            180: [(1,2), (1,1), (1,0), (0,0)],
            270: [(2,1), (1,1), (0,1), (0,2)],
        },
        3: {
            0: [(1,0), (1,1), (1,2), (0,2)],
            90: [(0,1), (1,1), (2,1), (2,2)],
            180: [(1,2), (1,1), (1,0), (2,0)],
            270: [(2,1), (1,1), (0,1), (0,0)],
        },
        4: {
            0: [(0,0), (1,0), (1,1), (2,1)],
            90: [(0,2), (0,1), (1,1), (1,0)],
            180: [(2,1), (1,1), (1,0), (0,0)],
            270: [(1,0), (1,1), (0,1), (0,2)],
        },
        5: {
            0: [(2,0), (1,0), (1,1), (0,1)],
            90: [(0,0), (0,1), (1,1), (1,2)],
            180: [(0,1), (1,1), (1,0), (2,0)],
            270: [(1,2), (1,1), (0,1), (0,0)],
        },
        6: {
            0: [(1,0), (2,0), (1,1), (2,1)],
            90: [(1,0), (2,0), (1,1), (2,1)],
            180: [(1,0), (2,0), (1,1), (2,1)],
            270: [(1,0), (2,0), (1,1), (2,1)],
        }
    }

    COLORS = {
        0: (0, 0, 0),    # Empty
        1: (255, 0, 0),  # I
        2: (0, 255, 0),  # T
        3: (0, 0, 255),  # L
        4: (255, 255, 0),# J
        5: (255, 165, 0),# Z
        6: (0, 255, 255),# S
        7: (200, 0, 200),# O
    }

    def __init__(self, record = False):
        x_mask = (np.arange(Tetris.BOARD_WIDTH*Tetris.RENDER_SCALE) % Tetris.RENDER_SCALE == 0)
        y_mask = (np.arange(Tetris.BOARD_HEIGHT*Tetris.RENDER_SCALE) % Tetris.RENDER_SCALE == 0)
        x_grid, y_grid = np.meshgrid(x_mask, y_mask)
        self._grid_mask = x_grid | y_grid
        self._recorder = None
        if record:
            while True:
                self._file_name = f"temp{random.randint(0, 1000)}.avi"
                if not os.path.exists(self._file_name):
                    break
            self._recorder = cv2.VideoWriter(self._file_name, cv2.VideoWriter_fourcc(*'MJPG'), 60.0, (Tetris.BOARD_WIDTH * Tetris.RENDER_SCALE, Tetris.BOARD_HEIGHT * Tetris.RENDER_SCALE))
        
        self.reset()

    def reset(self):
        self.board = [[Tetris.MAP_EMPTY] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.bag = list(range(len(Tetris.TETROMINOS)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        self.clearedLines = 0
        return self._get_board_props(self.board)

    def _get_rotated_piece(self):
        return Tetris.TETROMINOS[self.current_piece][self.current_rotation]

    def _get_complete_board(self):
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = (self.current_piece + 1, Tetris.COLORS[self.current_piece + 1])
        return board

    def get_game_score(self):
        return self.score
    
    def get_lines_cleared(self):
        return self.clearedLines
    
    def _new_round(self):
        if len(self.bag) == 0:
            self.bag = list(range(len(Tetris.TETROMINOS)))
            random.shuffle(self.bag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True

    def _check_collision(self, piece, pos):
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x][0] == Tetris.MAP_BLOCK:
                return True
        return False

    def _rotate(self, angle):
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r

    def _add_piece_to_board(self, piece, pos):
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = (Tetris.MAP_BLOCK, Tetris.COLORS[self.current_piece + 1])
        return board

    def _clear_lines(self, board):
        lines_to_clear = [index for index, row in enumerate(board) if sum(1 for cell in row if cell[0] != Tetris.MAP_EMPTY[0]) == Tetris.BOARD_WIDTH]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            for _ in lines_to_clear:
                board.insert(0, [Tetris.MAP_EMPTY for _ in range(Tetris.BOARD_WIDTH)])
        return len(lines_to_clear), board

    def _number_of_holes(self, board):
        holes = 0

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i][0] != Tetris.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x[0] == Tetris.MAP_EMPTY[0]])

        return holes

    def _bumpiness(self, board):
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i][0] != Tetris.MAP_BLOCK:
                i += 1
            min_ys.append(i)
        
        for i in range(len(min_ys) - 1):
            bumpiness = abs(min_ys[i] - min_ys[i+1])
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += abs(min_ys[i] - min_ys[i+1])

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            i = 0
            while i < Tetris.BOARD_HEIGHT and col[i][0] == Tetris.MAP_EMPTY[0]:
                i += 1
            height = Tetris.BOARD_HEIGHT - i
            sum_height += height
            if height > max_height:
                max_height = height
            elif height < min_height:
                min_height = height

        return sum_height, max_height, min_height

    def _get_board_props(self, board):
        lines, board = self._clear_lines(board)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_next_states(self):
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        for rotation in rotations:
            piece = Tetris.TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states

    def get_state_size(self):
        return 4

    def play(self, x, rotation, render=False, render_delay=None):
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render and render_delay is not None:
                self.render()
                if render_delay > 0:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1
        
        if render_delay is None and render:
            self.render()

        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
        clearedLines = lines_cleared
        self.clearedLines += clearedLines
        self.score += score

        self._new_round()
        if self.game_over:
            score -= 2

        return score, self.game_over

    def render(self, wait_ms=1):
        img = [cell[1] for row in self._get_complete_board() for cell in row]
        img = np.array(img).reshape(Tetris.BOARD_HEIGHT, Tetris.BOARD_WIDTH, 3).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * Tetris.RENDER_SCALE, Tetris.BOARD_HEIGHT * Tetris.RENDER_SCALE), resample=Image.NEAREST)
        img = np.array(img)
        
        img[self._grid_mask] = [0, 0, 0]
                
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 255, 255)
        thickness = 2
        line_type = cv2.LINE_AA
                
        cv2.putText(img, 'Score:', (10, 30), font, font_scale, color, thickness, line_type)
        cv2.putText(img, str(self.score), (120, 30), font, font_scale, color, thickness, line_type)
        cv2.putText(img, 'Lines Cleared:', (10, 70), font, font_scale, color, thickness, line_type)
        cv2.putText(img, str(self.clearedLines), (240, 70), font, font_scale, color, thickness, line_type)

        if self.game_over:
            text, font_scale, thickness, bg_color, padding = 'Game Over!', 2, 4, (0, 0, 0), 10
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x, text_y = (Tetris.BOARD_WIDTH * Tetris.RENDER_SCALE - text_size[0]) // 2, (Tetris.BOARD_HEIGHT * Tetris.RENDER_SCALE + text_size[1]) // 2
            cv2.rectangle(img, (text_x - padding, text_y - text_size[1] - padding), (text_x + text_size[0] + padding, text_y + padding), bg_color, cv2.FILLED)
            cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        if self._recorder:
            self._recorder.write(img)
            if self.game_over:
                self._recorder.release()
                self._recorder = None
                os.rename(self._file_name, f"tetris_{self.score}.avi")
        else:
            cv2.imshow('Tetris', img)
            cv2.waitKey(wait_ms)
        
if __name__ == "__main__":
    raise Exception("Unimplemented")