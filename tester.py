from agent import DQNAgent
from tetris import Tetris

MIN_OUTCOME_SCORE = 100000
RENDER_LAST_FRAME_ONLY = True

# Run DQN with Tetris
def eval():
    env = Tetris()
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    replay_start_size = 2000
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations, epsilon=0,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size, train=False)
    agent.load("ckpts\\134282_model.weights.h5")
    
    
    while True:
        current_state = env.reset()
        done = False
        steps = 0

        # Game
        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=(not RENDER_LAST_FRAME_ONLY),
                                    render_delay=render_delay)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1
            
        if (MIN_OUTCOME_SCORE is None) or (env.get_game_score() > MIN_OUTCOME_SCORE):
            break

    if RENDER_LAST_FRAME_ONLY:
        env.render(wait_ms=0)


if __name__ == "__main__":
    eval()
