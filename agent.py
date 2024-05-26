import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Sets up a Deep Q Network Agent with the following default structure:
#   - Memory Size: 10000
#   - Discount: 0.95
#   - Epsilon: 1
#   - Epsilon Min: 0
#   - Epsilon Stop Episode: 500
#   - Neurons: [32, 32]
#   - Activations: ['relu', 'relu', 'linear']
#   - Loss: 'mse'
#   - Optimizer: 'adam'
#   - Replay Start Size: 3333
#   - Train: False

# This network tries to predict the final output score of a given state. It's used to choose the best possible action based on the resulting state and the predicted reward from that state.
# This allows the network to behave with the future outcome in mind, allowing tetris (4 rows cleared at once) to be favoured and set up over immediate rewards (1 row cleared at once).
# The network is provided with 4 inputs:
#   - The number of holes in the board
#   - The height of the highest column
#   - The number of rows cleared
#   - The bumpiness of the board
# The network is trained by playing the game and storing the resulting states, actions, rewards, and next states in a memory buffer.

class DQNAgent:
    def __init__(self, state_size, mem_size=10000, discount=0.95, epsilon=1, epsilon_min=0, epsilon_stop_episode=500,
                 n_neurons=[32, 32], activations=['relu', 'relu', 'linear'], loss='mse', optimizer='adam', 
                 replay_start_size=None, train=False):

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / epsilon_stop_episode
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss_function = nn.MSELoss()
        self.optimizer_type = optimizer
        if not replay_start_size:
            replay_start_size = mem_size // 3
        self.replay_start_size = replay_start_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        layers = []
        input_size = self.state_size
        # Build the model based on the number of neurons and activations
        for i, neurons in enumerate(self.n_neurons):
            layers.append(nn.Linear(input_size, neurons))
            if self.activations[i] == 'relu':
                layers.append(nn.ReLU())
            elif self.activations[i] == 'linear':
                layers.append(nn.Identity())
            input_size = neurons
        layers.append(nn.Linear(input_size, 1))
        if self.activations[-1] == 'relu':
            layers.append(nn.ReLU())
        elif self.activations[-1] == 'linear':
            layers.append(nn.Identity())

        model = nn.Sequential(*layers)
        if self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(model.parameters())
        return model

    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append((current_state, next_state, reward, done))

    def random_value(self):
        return random.random()

    def predict_value(self, state):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            return self.model(state).item()

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            state = np.reshape(state, [1, self.state_size])
            return self.predict_value(state)

    def best_state(self, states):
        max_value = None
        best_state = None

        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if max_value is None or value > max_value:
                    max_value = value
                    best_state = state

        return best_state

    def train(self, batch_size=32):
        if len(self.memory) >= self.replay_start_size and len(self.memory) >= batch_size:
            batch = random.sample(self.memory, batch_size)

            next_states = np.array([x[1] for x in batch])
            next_qs = [self.predict_value(np.reshape(state, [1, self.state_size])) for state in next_states]

            x = []
            y = []

            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

            self.model.train()
            x = torch.FloatTensor(x).to(self.device)
            y = torch.FloatTensor(y).to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x).squeeze()
            loss = self.loss_function(outputs, y)
            loss.backward()
            self.optimizer.step()

            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay

    def save(self, save_dir):
        torch.save(self.model.state_dict(), save_dir)

    def load(self, load_dir):
        self.model.load_state_dict(torch.load(load_dir))
        self.model.eval()
