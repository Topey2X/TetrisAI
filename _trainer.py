# Trains and runs our model for car
import torch, random, math
import numpy as np
import matplotlib.pyplot as plt
from gym_simpletetris.envs import TetrisEnv
from model import *
import os
from datetime import datetime

np.float = float


def train(starting_policy=None):
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    working_dir = os.path.join("models", timestamp)
    os.makedirs(working_dir)

    # env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=False, isDiscrete=True)
    # env = TetrisEnv(renders=False, isDiscrete=True)
    env = TetrisEnv(
        height=20,                       # Height of Tetris grid
        width=10,                        # Width of Tetris grid
        obs_type='grayscale',                  # ram | grayscale | rgb
        extend_dims=False,               # Extend ram or grayscale dimensions
    #    render_mode='human',         # Unused parameter
        reward_step=True,               # See reward table
        penalise_height=True,           # See reward table
        penalise_height_increase=True,  # See reward table
        advanced_clears=True,           # See reward table
        high_scoring=True,              # See reward table
        penalise_holes=True,            # See reward table
        penalise_holes_increase=True,   # See reward table
        lock_delay=0,                    # Lock delay as number of steps
        step_reset=False                 # Reset lock delay on step downwards
    )
    # set manual seeds so we get same behaviour everytime - so that when you change your hyper parameters you can attribute the effect to those changes
    env.action_space.seed(0)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    episode_batch_score = 0
    episode_reward = 0
    report_every = 100
    agent = DQN_Solver(env)  # create DQN agent
    if starting_policy is not None:  # load current policy if given
        checkpoint = torch.load(starting_policy)
        agent.policy_network.load_state_dict(checkpoint["policy"])
        agent.policy_network.eval()
        agent.learn_count += 1000

    plt.clf()

    print("START TRAINING")
    try:
        for i in range(EPISODES):
            state = (
                env.reset()
            )  # this needs to be called once at the start before sending any actions
            while True:
                # sampling loop - sample random actions and add them to the replay buffer
                action = agent.choose_action(state)
                state_, reward, done, info = env.step(action)

                if done:
                    pass  # timed out

                # (Remember this outcome)
                agent.memory.add(state, action, reward, state_, done)

                # only start learning once replay memory reaches REPLAY_START_SIZE
                if agent.memory.mem_count > REPLAY_START_SIZE:
                    agent.learn()

                state = state_
                episode_batch_score += reward
                episode_reward += reward

                if done:
                    break

            episode_history.append(i)
            episode_reward_history.append(episode_reward)
            episode_reward = 0.0

            # save our model every batches of 100 episodes so we can load later. (note: you can interrupt the training any time and load the latest saved model when testing)
            if ((i + 1) % report_every) == 0:
                if agent.memory.mem_count > REPLAY_START_SIZE:
                    torch.save(
                        {"policy": agent.policy_network.state_dict()},
                        f"{working_dir}/policy_network{i // 100}.pkl",
                    )
                    print(
                        f"EPISODES {i-report_every+1}-{i+1}: Average total reward per episode batch: {episode_batch_score/ float(100)}"
                    )
                    episode_batch_score = 0
                else:
                    print(
                        f"EPISODES {i-report_every+1}-{i+1}: waiting for buffer to fill..."
                    )
                    episode_batch_score = 0
    except KeyboardInterrupt:
        pass  # still show the final training

    torch.save(
        {"policy": agent.policy_network.state_dict()},
        f"{working_dir}/policy_network_final.pkl",
    )

    plt.plot(episode_history, episode_reward_history)
    plt.show()


if __name__ == "__main__":
    train()
    # train("models/240415-144333/policy_network_final.pkl")
