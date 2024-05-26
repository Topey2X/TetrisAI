from agent import DQNAgent
from tetris import Tetris

MIN_OUTCOME_SCORE = None # None or a minimum score to reach before stopping
RENDER_GAME = True # Whether to show the game while running, or just the final screen.
RECORD_GAME = False # Saves a '.avi' video file instead of displaying the game (overrides RENDER_GAME)
WEIGHTS_FILE = "BEST_MODEL.weights.h5"

# Run DQN with Tetris
def eval():
    env = Tetris(RECORD_GAME)
    max_steps = None
    n_neurons = [32, 32]
    render_delay = 0 # set to None for max speed rendering
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(), n_neurons=n_neurons, activations=activations, epsilon=0, train=False)
    agent.load(WEIGHTS_FILE)
    
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

            reward, done = env.play(best_action[0], best_action[1], render=RENDER_GAME,
                                    render_delay=render_delay)
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1
            
            if RECORD_GAME:
                print(f"\rScore: {env.get_game_score()}", end="")
            
        if (MIN_OUTCOME_SCORE is None) or (env.get_game_score() > MIN_OUTCOME_SCORE):
            break

    env.render(wait_ms=0)


if __name__ == "__main__":
    eval()
