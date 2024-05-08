import torch, random, math
import numpy as np
import matplotlib.pyplot as plt
from simple_driving.envs.simple_driving_env import *
from Tetris_DQN_Solver import *

############################################################################################

# Train network
def train:
    #env = gym.make("SimpleDriving-v0", apply_api_compatibility=True, renders=True, isDiscrete=True)
    env = gym.make('SimpleTetris-v0',
        height=20,                       # Height of Tetris grid
        width=10,                        # Width of Tetris grid
        obs_type='ram',            # ram | grayscale | rgb
        extend_dims=False,               # Extend ram or grayscale dimensions
    #    render_mode='human',         # Unused parameter
        reward_step=False,               # See reward table
        penalise_height=False,           # See reward table
        penalise_height_increase=False,  # See reward table
        advanced_clears=False,           # See reward table
        high_scoring=False,              # See reward table
        penalise_holes=False,            # See reward table
        penalise_holes_increase=False,   # See reward table
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
    agent = DQN_Solver(env)  # create DQN agent
    plt.clf()

    for i in range(EPISODES):
        state = env.reset()  # this needs to be called once at the start before sending any actions
        while True:
            # sampling loop - sample random actions and add them to the replay buffer
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            #if reached_goal == True:
            #    reward += 100
            #else:
            #    reward -= 100

            
            

            ####### add sampled experience to replay buffer ##########
            agent.memory.add(state, action, reward, state_, done)
            ##########################################################

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
        if i % 100 == 0 and agent.memory.mem_count > REPLAY_START_SIZE:
            torch.save(agent.policy_network.state_dict(), "C:\\Users\\liamx\\OneDrive - UTS\\UTS Class Files\\UTS Info\\4th Year\\Semester 1\\AI in Robotics\\Quiz 3/policy_network.pkl")
            print("average total reward per episode batch since episode ", i, ": ", episode_batch_score/ float(100))
            episode_batch_score = 0
        elif agent.memory.mem_count < REPLAY_START_SIZE:
            print("waiting for buffer to fill...")
            episode_batch_score = 0
            
    torch.save(agent.policy_network.state_dict(), "latest_model.pkl")

    plt.plot(episode_history, episode_reward_history)
    plt.show()

if __name__ == "__main__":
    
