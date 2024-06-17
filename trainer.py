import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
from tqdm import tqdm
from agent import DQNAgent
from tetris import Tetris
from os import makedirs, path

EXCEPTIONAL_SCORE_THRESHOLD = 100000

# Run DQN with Tetris
def dqn():
    env = Tetris()
    episodes = 10000
    max_steps = None
    epsilon_stop_episode = 500
    mem_size = 20000
    discount = 0.99
    batch_size = 512
    render_every = None
    log_every = None
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']
    # previous_weights = None
    previous_weights = "BEST_MODEL.weights.h5"


    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations, epsilon=0.01, epsilon_min=0.0001,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    if previous_weights and path.exists(previous_weights):
        agent.load(previous_weights)
        print(f"Loaded weights from {previous_weights}")

    scores = []
    clearedLines = []

    try:
        for episode in tqdm(range(episodes), desc="Training", unit="episodes"):
            current_state = env.reset()
            done = False
            steps = 0

            render = (render_every is not None) and (episode % render_every == 0)

            # Game
            while not done and (not max_steps or steps < max_steps):
                next_states = env.get_next_states()
                best_state = agent.best_state(next_states.values())
                
                best_action = None
                for action, state in next_states.items():
                    if state == best_state:
                        best_action = action
                        break

                reward, done = env.play(best_action[0], best_action[1], render=render,
                                        render_delay=render_delay)
                
                agent.add_to_memory(current_state, next_states[best_action], reward, done)
                current_state = next_states[best_action]
                steps += 1
            scores.append(env.get_game_score())
            clearedLines.append(env.get_lines_cleared())

            # Train
            if episode % train_every == 0:
                agent.train(batch_size=batch_size)

            # Logs
            if log_every and episode and episode % log_every == 0:            
                # Save Weights
                makedirs("checkpoints", exist_ok=True)
                agent.save(f"checkpoints\\{episode}-{mean(scores[-1])}.weights.h5")
            
            # Exceptional Results
            if scores[-1] > EXCEPTIONAL_SCORE_THRESHOLD:            
                # Save Weights
                makedirs("checkpoints", exist_ok=True)
                agent.save(f"checkpoints\\_exceptional_{episode}-{scores[-1]}.weights.h5")
                
    except KeyboardInterrupt:
        print("Training Interrupted!")
    
    # Save Weights
    makedirs("checkpoints", exist_ok=True)
    agent.save(f"checkpoints\\_final_{episodes}-{mean(scores[-1])}.weights.h5")
    

    # Plot Scores and Cleared Lines
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Scores', color=color)
    ax1.plot(scores, color=color, label='Scores')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Lines Cleared', color=color)
    ax2.plot(clearedLines, color=color, label='Lines Cleared')
    ax2.tick_params(axis='y', labelcolor=color)
    
    x = np.arange(len(scores))
    y = scores
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    ax1.plot(x, p(x), color='orange', label='Best Fit')

    fig.tight_layout()
    plt.title('Training Progress')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    dqn()
