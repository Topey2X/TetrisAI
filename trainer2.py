from datetime import datetime
from statistics import mean
import random
from tqdm import tqdm
import torch

from model2 import DQNAgent
from tetris import Tetris
import os

# Run DQN with Tetris
def dqn():
    env = Tetris()
    episodes = 3500
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    # render_every = 500
    render_every = None
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']


    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    # log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    scores = []
    clearedLines = []

    #rendering = False

    for episode in tqdm(range(episodes), desc="Training", unit="episodes"):
        #if episode == 3000:
        #    rendering = True

        current_state = env.reset()
        done = False
        steps = 0

        if render_every and episode % render_every == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=False,
                                    render_delay=render_delay)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1
        scores.append(env.get_game_score())
        clearedLines.append(env.get_lines_cleared)

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)

        # Logs
        if log_every and episode and episode % log_every == 0:            
            # Save Weights
            os.makedirs('ckpts', exist_ok=True)
            agent.save(f'ckpts/{scores[-1]}_model.weights.h5')

if __name__ == "__main__":
    dqn()
