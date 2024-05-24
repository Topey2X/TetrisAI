import gym
from gym import spaces
import numpy as np
from tetris import Tetris

class TetrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TetrisEnv, self).__init__()
        self.tetris = Tetris()
        self.action_space = spaces.Discrete(40)  # 10 positions * 4 rotations
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.tetris.get_state_size(),), dtype=np.float32
        )
    
    def reset(self):
        self.tetris.reset()
        return self._get_obs()
    
    def step(self, action):
        x = action % 10
        rotation = (action // 10) * 90
        reward, done = self.tetris.play(x, rotation)
        return self._get_obs(), reward, done, {}
    
    def render(self, mode='human', close=False):
        self.tetris.render()
    
    def _get_obs(self):
        # Assuming the observation is derived from the board properties
        board_props = self.tetris._get_board_props(self.tetris.board)
        return np.array(board_props, dtype=np.float32)