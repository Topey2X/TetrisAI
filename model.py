## DQN neural net model.

import torch, random, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Hyper parameters that will be used in the DQN algorithm

EPISODES = 100000                 # number of episodes to run the training for
LEARNING_RATE = 0.0001         # the learning rate for optimising the neural network weights
MEM_SIZE = 100000                # maximum size of the replay memory - will start overwritting values once this is exceed
REPLAY_START_SIZE = 20000       # The amount of samples to fill the replay memory with before we start learning
BATCH_SIZE = 128                 # Number of random samples from the replay memory we use for training each iteration
GAMMA = 0.99                    # Discount factor
EPS_START = 0.2                 # Initial epsilon value for epsilon greedy action sampling
EPS_END = 0.0001                # Final epsilon value
EPS_DECAY = 4 * MEM_SIZE        # Amount of samples we decay epsilon over
MEM_RETAIN = 0.1                # Percentage of initial samples in replay memory to keep - for catastrophic forgetting
NETWORK_UPDATE_ITERS = 5000     # Number of samples 'C' for slowly updating the target network \hat{Q}'s weights with the policy network Q's weights

FC1_DIMS = 512                   # Number of neurons in our MLP's first hidden layer
FC2_DIMS = 512                   # Number of neurons in our MLP's second hidden layer
# FC3_DIMS = 128                   # Number of neurons in our MLP's third hidden layer

# metrics for displaying training status
best_reward = 0
average_reward = 0
episode_history = []
episode_reward_history = []
np.bool = np.bool_ # backwards compatability.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# for creating the policy and target networks - same architecture
class Network(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.input_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        # build an MLP with 2 hidden layers
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(*self.input_shape, FC1_DIMS),   # input layer
            torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC1_DIMS, FC2_DIMS),    # hidden layer
            torch.nn.ReLU(),     # this is called an activation function
            # torch.nn.Linear(FC2_DIMS, FC3_DIMS),    # hidden layer
            # torch.nn.ReLU(),     # this is called an activation function
            torch.nn.Linear(FC2_DIMS, self.action_space)    # output layer
            )

        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()  # loss function

    def forward(self, x):
        return self.layers(x)

# handles the storing and retrival of sampled experiences
class ReplayBuffer:
    def __init__(self, env):
        self.mem_count = 0
        self.states = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64)
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, *env.observation_space.shape),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=np.bool)

    def add(self, state, action, reward, state_, done):
        # if memory count is higher than the max memory size then overwrite previous values
        if self.mem_count < MEM_SIZE:
            mem_index = self.mem_count
        else:
            mem_index = int(self.mem_count % ((1-MEM_RETAIN) * MEM_SIZE) + (MEM_RETAIN * MEM_SIZE))  # avoid catastrophic forgetting, retain first 10% of replay buffer

        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1

    # returns random samples from the replay buffer, number is equal to BATCH_SIZE
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)

        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self, env):
        self.memory = ReplayBuffer(env)
        self.policy_network = Network(env).to(device)  # Move policy network to the designated device
        self.target_network = Network(env).to(device)  # Move target network to the designated device
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.learn_count = 0

    def choose_action(self, observation):
        if self.memory.mem_count > REPLAY_START_SIZE:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.learn_count / EPS_DECAY)
        else:
            eps_threshold = 1.0

        if random.random() < eps_threshold:
            return np.random.choice(np.array(
                    range(3)),
                    p=[0.25,0.5,0.25] #force driving, allow steering, ban not moving
                )

        state = torch.tensor(observation, device=device).float().detach().unsqueeze(0)
        self.policy_network.eval()
        with torch.no_grad():
            q_values = self.policy_network(state)
        return torch.argmax(q_values).item()

    def learn(self):
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        states_ = torch.tensor(states_, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.bool, device=device)
        batch_indices = torch.arange(BATCH_SIZE, dtype=torch.int64, device=device)

        self.policy_network.train(True)
        q_values = self.policy_network(states)
        q_values = q_values[batch_indices, actions]

        self.target_network.eval()
        with torch.no_grad():
            q_values_next = self.target_network(states_)
        q_values_next_max = torch.max(q_values_next, dim=1)[0]
        q_target = rewards + GAMMA * q_values_next_max * dones

        loss = self.policy_network.loss(q_values, q_target)
        self.policy_network.optimizer.zero_grad()
        loss.backward()
        self.policy_network.optimizer.step()
        self.learn_count += 1

        if self.learn_count % NETWORK_UPDATE_ITERS == NETWORK_UPDATE_ITERS - 1:
            self.update_target_network()

    def update_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def returning_epsilon(self):
        return self.exploration_rate
    
if __name__ == "__main__":
    print("Idiot, run the trainer, not the model ...")
