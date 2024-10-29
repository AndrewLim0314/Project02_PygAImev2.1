import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)  # Second hidden layer
        self.fc3 = nn.Linear(128, output_dim)  # Output layer
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)  # Optimizer
        self.memory = deque(maxlen=10000)  # Replay buffer
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay for epsilon
        self.epsilon_min = epsilon_min  # Minimum epsilon

    def forward(self, x):
        """ Forward pass through the network. """
        x = torch.relu(self.fc1(x))  # Activation for first layer
        x = torch.relu(self.fc2(x))  # Activation for second layer
        return self.fc3(x)  # Output Q-values

    def remember(self, state, action, reward, next_state, done):
        """ #Store the experience in the replay buffer.
"""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state):
        """ Select an action based on the current state. """
        if random.random() < self.epsilon:
            return random.randrange(self.fc3.out_features)  # Random action
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor
            with torch.no_grad():  # Disable gradient computation
                q_values = self.forward(state_tensor)  # Get Q-values
            return torch.argmax(q_values).item()  # Return action with highest Q-value



    def replay(self, batch_size):
        """ #Experience replay to update the model. """
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)  # Sample a minibatch from memory

        # Convert lists of numpy arrays to a single numpy array before converting to a tensor
        state_batch = torch.FloatTensor(np.array([s[0] for s in minibatch]))
        action_batch = torch.LongTensor(np.array([s[1] for s in minibatch]))
        reward_batch = torch.FloatTensor(np.array([s[2] for s in minibatch]))
        next_state_batch = torch.FloatTensor(np.array([s[3] for s in minibatch]))
        done_batch = torch.FloatTensor(np.array([s[4] for s in minibatch]))

        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.forward(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (0.99 * next_q_values * (1 - done_batch))

        # Get current Q-values
        current_q_values = self.forward(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()

        # Compute loss and update the model
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()  # Backpropagate the loss
        self.optimizer.step()  # Update the weights

        # Decay epsilon after every episode
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay